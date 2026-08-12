# GUI for GLMakie
using GLMakie

include("ThermalExport.jl")

export sill_intrusion_1D, compute_zircon_ages, volume_averaged_age

include("ThermalCode_1D.jl")

# populated with the latest run's data at the end of each simulation, so it's reachable
# from the REPL: tracer T-t histories (`QMagma.tracers_out`, for ZirconGrowth.jl), tracers
# removed by eruption (`QMagma.erupted_tracers_out`), and 1D profiles / temporal evolution
# (`QMagma.last_run_out`)
tracers_out = Tracer[]
erupted_tracers_out = Tracer[]
last_run_out = Dict{Symbol,Any}()

# Few helpers (height=18 keeps each widget within its Fixed(18) grid row, otherwise the
# default widget heights are taller than the row and stick out past the colored Box behind
# it; Textbox additionally needs fontsize/textpadding reduced from their defaults, since the
# default 8px top+bottom textpadding alone exceeds the 18px row, clipping the displayed text):
add_textbox(fig, label, value) = [Label(fig, label), Textbox(fig, stored_string = string(value), validator = typeof(value), height=18, fontsize=11, textpadding=(4,4,2,2))]
add_togglebox(fig, label, active) = [Label(fig, label), Toggle(fig, active=active, height=18)]
get_valuebox(box::Vector) = parse(box[2].validator.val, box[2].stored_string.val)

function gui_flux_history(mode; base_m_per_yr, peak_m_per_yr,
                          start_kyr, end_kyr, table_path="")
    if mode == "CSV table"
        isempty(strip(table_path)) && throw(ArgumentError("Flux CSV path must not be empty"))
        return load_flux_history(expanduser(strip(table_path)))
    end
    symbol = mode == "Constant" ? :constant :
             mode == "Linear ramp" ? :ramp :
             mode == "Pulse" ? :pulse :
             throw(ArgumentError("unknown flux mode: $mode"))
    return FluxHistory(symbol;
        base=base_m_per_yr/SecYear,
        peak=peak_m_per_yr/SecYear,
        t_start=start_kyr*1000SecYear,
        t_end=end_kyr*1000SecYear)
end

"""
    eruption_control_state(trigger)

Return which eruption controls apply to a selected eruption trigger. The sill radius,
maximum eruption depth, and collapse mechanism apply to every active trigger: radius
sets reported erupted volume, depth gates every trigger, and collapse acts after every
eruption.
"""
function eruption_control_state(trigger)
    trigger in ("None", "Melt thickness", "Elastic box model", "D&H 3-phase") ||
        throw(ArgumentError("unknown eruption trigger: $trigger"))
    active = trigger != "None"
    pressure = trigger in ("Elastic box model", "D&H 3-phase")
    return (; threshold = trigger == "Melt thickness", radius = active,
              pressure, shear_modulus = pressure, magma_compressibility = pressure,
              max_depth = active, collapse = active)
end

function eruption_fires(trigger; thickness, threshold, overpressure, pressure_critical,
                        h_erupt, z_lo, z_hi, z_erupt_max, near_boundary, Δz)
    trigger in ("Melt thickness", "Elastic box model", "D&H 3-phase") ||
        throw(ArgumentError("unknown active eruption trigger: $trigger"))
    triggered = trigger == "Melt thickness" ? thickness >= threshold :
                trigger == "Elastic box model" ? overpressure >= pressure_critical :
                h_erupt > 0
    depth_ok = abs((z_lo + z_hi)/2) <= z_erupt_max
    return triggered && depth_ok && !near_boundary && h_erupt > 2Δz
end

function collapse_surface_subsidence(method, h_erupt)
    method in (:caldera, :elastic, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    return method in (:caldera, :hybrid) ? h_erupt : 0.0
end

function set_textbox_enabled!(box, enabled)
    label, textbox = box
    label.color = enabled ? :black : (:gray, 0.55)
    textbox.textcolor = enabled ? :black : (:gray, 0.65)
    textbox.boxcolor = enabled ? :transparent : (:gray, 0.15)
    textbox.boxcolor_hover = enabled ? :transparent : (:gray, 0.15)
    textbox.boxcolor_focused = enabled ? :transparent : (:gray, 0.15)
    textbox.bordercolor = enabled ? (:gray, 0.8) : (:gray, 0.45)
    textbox.bordercolor_hover = enabled ? (:gray, 0.55) : (:gray, 0.45)
    enabled || GLMakie.Makie.defocus!(textbox)
    return nothing
end

function bind_textbox_enabled!(box, enabled)
    textbox = box[2]
    on(enabled) do active
        set_textbox_enabled!(box, active)
    end
    on(textbox.focused) do focused
        !enabled[] && focused && GLMakie.Makie.defocus!(textbox)
    end
    set_textbox_enabled!(box, enabled[])
    return nothing
end

function set_menu_enabled!(menu, enabled)
    menu.textcolor = enabled ? :black : (:gray, 0.55)
    menu.dropdown_arrow_color = enabled ? (:black, 0.2) : (:gray, 0.45)
    enabled || (menu.is_open[] = false)
    return nothing
end

function bind_menu_enabled!(menu, enabled)
    on(enabled) do active
        set_menu_enabled!(menu, active)
    end
    on(menu.is_open) do open
        !enabled[] && open && (menu.is_open[] = false)
    end
    set_menu_enabled!(menu, enabled[])
    return nothing
end

"""
    sill_intrusion_1D(; size=nothing)

Interactive GLMakie App for 1D thermal intrusion model. `size` is the size of the window
in pixels; if `nothing` (default), it's chosen automatically to fit within the primary
monitor's available height, since a fixed pixel height can be taller than some screens
(clipping the bottom of the control panel) and there's no scrollable layout to fall back on.
"""
function sill_intrusion_1D(; size=nothing)
    GLMakie.activate!()
    GLMakie.closeall() # close any open screen

    if size === nothing
        win_w = 1500
        win_h = 900
        try
            monitor = GLMakie.GLFW.GetPrimaryMonitor()
            vidmode = GLMakie.GLFW.GetVideoMode(monitor)
            # GetVideoMode reports the full screen height; subtract a fixed margin for
            # the menu bar/dock/title bar (not otherwise queryable here) rather than a
            # percentage, since the menu bar's height doesn't scale with screen size
            win_h = min(win_h, vidmode.height - 130)
        catch e
            @warn "Could not query monitor size; using default window height" exception=e
        end
        size = (win_w, win_h)
    end

    fig = Figure(size=size)

    time_val = Observable(0.0)
    stop_requested = Observable(false)
    sim_running = Observable(false)
    zircon_running = Observable(false)
    last_run = Dict{Symbol,Any}()
    last_matparam = nothing

    Label(fig[0, 1:3], text = "1D Sill Injection", fontsize = 30)

    fig[1, 1] = depth_fig = GridLayout()
    ax1 =  Axis(depth_fig[1:3, 1], xlabel="Temperature [ᵒC]", ylabel="Depth [km]", title = @lift("t = $(round($time_val, digits = 2)) kyrs"))
    ax2 =  Axis(depth_fig[1:3, 2], xlabel="Melt fraction ϕ")

    fig[1, 2] = timeseries_fig = GridLayout()
    ax3 =  Axis(timeseries_fig[1, 1], xlabel="Time [kyrs]", ylabel="Maximum Temperature [ᵒC]",ytickcolor=:red,ylabelcolor=:red,yticklabelcolor=:red)
    ax4 =  Axis(timeseries_fig[1, 1], ylabel="Maximum melt fraction ϕ",ytickcolor=:blue,ylabelcolor=:blue,yticklabelcolor=:blue,  yaxisposition = :right)
    ax5 =  Axis(timeseries_fig[2, 1], xlabel="Time [kyrs]", ylabel="Cumulative erupted volume [km³]",ytickcolor=:darkgreen,ylabelcolor=:darkgreen,yticklabelcolor=:darkgreen)
    ax5b = Axis(timeseries_fig[2, 1], ylabel="Erupted volume per event [km³]",ytickcolor=:orange,ylabelcolor=:orange,yticklabelcolor=:orange,  yaxisposition = :right)

    # bottom row, full width under temp / melt-frac / eruptions: the D&H chamber's H₂O
    # speciation (dissolved vs exsolved, left) overlaid with the mush-mean melt fraction
    # ϕ_mush (right) that drives the split. Only populated when the D&H 3-phase trigger runs.
    fig[2, 1:2] = chamber_fig = GridLayout()
    ax6  = Axis(chamber_fig[1, 1], xlabel="Time [kyrs]", ylabel="H₂O mass fraction [-]",
                title="D&H chamber: H₂O speciation vs mush melt fraction",
                ytickcolor=:teal, ylabelcolor=:teal, yticklabelcolor=:teal)
    ax6b = Axis(chamber_fig[1, 1], ylabel="ϕ_mush [-]",
                ytickcolor=:gray30, ylabelcolor=:gray30, yticklabelcolor=:gray30, yaxisposition = :right)

    linkxaxes!(ax3, ax4)
    linkxaxes!(ax3, ax5)
    linkxaxes!(ax3, ax5b)
    linkxaxes!(ax3, ax6)
    linkxaxes!(ax3, ax6b)

    # keep the full-width H₂O panel from collapsing under the taller row-1 panels
    rowsize!(fig.layout, 2, Relative(0.28))

    fig[1:2, 3] = grid = GridLayout(tellwidth = false)
    rowgap!(grid, 0)

    grid[1, 1] = but            = Button(fig, label = "  RUN SIMULATION  ", buttoncolor = :lightgreen, height=18, fontsize=11)
    grid[1, 2] = but_stop       = Button(fig, label = "  STOP  ", buttoncolor = :red, height=18, fontsize=11)

    Box(grid[2:3, 1:4], color = :lightgrey, cornerradius = 10)
    grid[2, 1:2] = Δz_box       = add_textbox(fig,"Δz [m]:",20)
    grid[2, 3:4] = nt_box       = add_textbox(fig,"# steps nt:",3000)
    grid[3, 1:2] = Δt_yrs_box   = add_textbox(fig,"Δt [yrs]:",100.0)

    Box(grid[4:5, 1:4], color = :lightblue, cornerradius = 10)
    grid[4, 1:2] = H_box        = add_textbox(fig,"Crust [km]:",40.0)
    grid[4, 3:4] = Ttop_box     = add_textbox(fig,"Ttop [ᵒC]:",0.0)
    grid[5, 1:2] = γ_box        = add_textbox(fig,"Geotherm [ᵒC/km]:",20.0)
    # η_r (wall relaxation viscosity) is computed analytically from the country-rock T each
    # step (wall_relaxation_viscosity), so no GUI knob for it — grid[5,3:4] left free.

    Box(grid[6:11, 1:4], color = :lightyellow, cornerradius = 10)
    grid[6, 1:2] = Tsill_box    = add_textbox(fig,"Sill T [ᵒC]:",1200.0)
    grid[6, 3:4] = Sill_thick_box = add_textbox(fig,"Sill thick [m]:",100.0)
    grid[7, 1:4] = menu_flux = Menu(fig,
        options = ["Constant", "Linear ramp", "Pulse", "CSV table"],
        default = "Constant", height=18, fontsize=11)
    grid[8, 1:2] = flux_base_box = add_textbox(fig,"Base flux [m/yr]:",0.1)
    grid[8, 3:4] = flux_peak_box = add_textbox(fig,"Peak/end flux [m/yr]:",0.2)
    grid[9, 1:2] = flux_start_box = add_textbox(fig,"Flux start [kyr]:",50.0)
    grid[9, 3:4] = flux_end_box = add_textbox(fig,"Flux end [kyr]:",100.0)
    grid[10, 1:4] = flux_table_box = [Label(fig, "Flux CSV path:"),
        Textbox(fig, stored_string="flux.csv", height=18, fontsize=11,
                textpadding=(4,4,2,2))]
    grid[11, 1:2] = Sill_interval_top_box = add_textbox(fig,"Top inj. [km]:",10.0)
    grid[11, 3:4] = Sill_interval_bot_box = add_textbox(fig,"Bottom inj. [km]:",20.0)

    flux_base_enabled = Observable(true)
    flux_peak_enabled = Observable(false)
    flux_times_enabled = Observable(false)
    flux_table_enabled = Observable(false)
    bind_textbox_enabled!(flux_base_box, flux_base_enabled)
    bind_textbox_enabled!(flux_peak_box, flux_peak_enabled)
    bind_textbox_enabled!(flux_start_box, flux_times_enabled)
    bind_textbox_enabled!(flux_end_box, flux_times_enabled)
    bind_textbox_enabled!(flux_table_box, flux_table_enabled)

    function update_flux_controls!(mode)
        table = mode == "CSV table"
        variable = mode in ("Linear ramp", "Pulse")
        flux_base_enabled[] = !table
        flux_peak_enabled[] = variable
        flux_times_enabled[] = variable
        flux_table_enabled[] = table
        return nothing
    end
    on(update_flux_controls!, menu_flux.selection)
    update_flux_controls!(menu_flux.selection[])

    Box(grid[12:14, 1:4], color = (:red,0.3), cornerradius = 10 )
    grid[12, 1:2] = Ql_box = add_textbox(fig,"Latent heat [kJ/kg]:",255.0)
    grid[13, 1:4] = menu_conduct = Menu(fig, options = ["T-dependent conductivity", "Constant conductivity 3 W/m/K"], default = "Constant conductivity 3 W/m/K", height=18, fontsize=11)
    grid[14, 1:4] = menu_melting = Menu(fig, options = ["MeltingParam_Assimilation", "MeltingParam_Basalt", "MeltingParam_Rhyolite"], default = "MeltingParam_Basalt", height=18, fontsize=11)

    Box(grid[15:16, 1:4], color = (:orange,0.3), cornerradius = 10 )
    grid[15, 1:4] = Label(fig, "Method:")
    grid[16, 1:4] = menu_method = Menu(fig, options = ["Discrete sills", "Q_magma", "Both (compare)"], default = "Both (compare)", height=18, fontsize=11)

    Box(grid[17:21, 1:4], color = (:purple,0.2), cornerradius = 10 )
    # eruption is two independent choices: the TRIGGER (when the column erupts) and the
    # COLLAPSE kinematics (how the column closes the vent afterwards). Any trigger can be
    # paired with any collapse.
    grid[17, 1:4] = Label(fig, "Eruption  trigger  |  collapse:")
    grid[18, 1:2] = menu_trigger  = Menu(fig, options = ["None", "Melt thickness", "Elastic box model", "D&H 3-phase"], default = "None", height=18, fontsize=11)
    grid[18, 3:4] = menu_collapse = Menu(fig, options = ["Hybrid", "Caldera", "Elastic"], default = "Hybrid", height=18, fontsize=11)
    grid[19, 1:2] = Eruption_thick_box = add_textbox(fig,"Threshold [m]:",500.0)
    grid[19, 3:4] = Sill_radius_box    = add_textbox(fig,"Sill radius [km]:",5.0)
    grid[20, 1:2] = dPc_box            = add_textbox(fig,"ΔP crit [MPa]:",20.0)
    grid[20, 3:4] = mu_box             = add_textbox(fig,"μ shear [GPa]:",10.0)
    grid[21, 1:2] = beta_box           = add_textbox(fig,"β magma [1/GPa]:",0.1)
    grid[21, 3:4] = zmax_box           = add_textbox(fig,"Max erupt depth [km]:",15.0)

    # Grey inactive eruption inputs and prevent focus/opening, so the selected trigger
    # is the only source of applicable settings.
    threshold_enabled = Observable(false)
    radius_enabled    = Observable(false)
    pressure_enabled  = Observable(false)
    shear_enabled     = Observable(false)
    compressibility_enabled = Observable(false)
    depth_enabled     = Observable(false)
    collapse_enabled  = Observable(false)
    bind_textbox_enabled!(Eruption_thick_box, threshold_enabled)
    bind_textbox_enabled!(Sill_radius_box, radius_enabled)
    bind_textbox_enabled!(dPc_box, pressure_enabled)
    bind_textbox_enabled!(mu_box, shear_enabled)
    bind_textbox_enabled!(beta_box, compressibility_enabled)
    bind_textbox_enabled!(zmax_box, depth_enabled)
    bind_menu_enabled!(menu_collapse, collapse_enabled)

    function update_eruption_controls!(trigger)
        controls = eruption_control_state(trigger)
        threshold_enabled[] = controls.threshold
        radius_enabled[] = controls.radius
        pressure_enabled[] = controls.pressure
        shear_enabled[] = controls.shear_modulus
        compressibility_enabled[] = controls.magma_compressibility
        depth_enabled[] = controls.max_depth
        collapse_enabled[] = controls.collapse
        return nothing
    end
    on(update_eruption_controls!, menu_trigger.selection)
    update_eruption_controls!(menu_trigger.selection[])

    Box(grid[22:25, 1:4], color = (:green,0.3), cornerradius = 10 )
    grid[22, 1:2] = filename = [Label(fig, "filename:"), Textbox(fig, stored_string = "sim1", height=18, fontsize=11, textpadding=(4,4,2,2))]
    grid[23, 1:2] = but_save =  Button(fig, label = "  SAVE SCREENSHOT  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)
    grid[23, 3:4] = but_save_data = Button(fig, label = "  SAVE JLD2 + VTK  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)
    grid[24, 1:2] = record_toggle = add_togglebox(fig,"Record movie:",false)
    grid[25, 1:4] = but_zircon = Button(fig, label = "  COMPUTE ZIRCON AGES  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)

    for r in 1:25
        rowsize!(grid, r, Fixed(18))
    end

    on(but_save.clicks) do n
        png_name = filename[2].stored_string.val * ".png"
        save(png_name, fig)
        println("Save screenshot to $(joinpath(pwd(), png_name))")
    end

    on(but_save_data.clicks) do n
        if isempty(last_run)
            println("No simulation data to save yet - run the simulation first")
        else
            base_name = filename[2].stored_string.val
            jld2_name = base_name * ".jld2"
            jldsave(jld2_name; last_run...)
            vtk_names = String[]
            matparam = last_matparam
            isnothing(matparam) && error("No material parameters available for 2D/3D melt-fraction export")
            sigma = last_run[:gaussian_sigma]
            x2 = range(-3sigma, 3sigma; length=41)
            x3 = range(-3sigma, 3sigma; length=21)
            for (label, temperature_key) in (("discrete", :T), ("Qmagma", :T_Qmagma))
                haskey(last_run, temperature_key) || continue
                temperature = last_run[temperature_key]
                if temperature_key === :T
                    append!(vtk_names, export_thermal_structure(base_name * "_" * label, last_run[:z];
                        fields=(temperature, melt_fraction=last_run[:phi], rocks=last_run[:rocks]),
                        formats=(:vtk,)))
                else
                    append!(vtk_names, export_thermal_structure(base_name * "_" * label, last_run[:z];
                        fields=(temperature, melt_fraction=last_run[:phi_Qmagma]), formats=(:vtk,)))
                end
                T2 = gaussian_thermal_structure(temperature, last_run[:T_background], x2; sigma)
                T3 = gaussian_thermal_structure(temperature, last_run[:T_background], x3; y=x3, sigma)
                ϕ2 = melt_fraction_from_temperature(T2, matparam)
                ϕ3 = melt_fraction_from_temperature(T3, matparam)
                fields2 = (temperature=T2, melt_fraction=ϕ2)
                fields3 = (temperature=T3, melt_fraction=ϕ3)
                if temperature_key === :T
                    rocks = last_run[:rocks]
                    rocks2 = (abs.(x2) .<= sigma) .* reshape(rocks, 1, :)
                    rocks3 = ((x3.^2 .+ (x3').^2) .<= sigma^2) .* reshape(rocks, 1, 1, :)
                    fields2 = (; fields2..., rocks=rocks2)
                    fields3 = (; fields3..., rocks=rocks3)
                end
                append!(vtk_names, export_thermal_structure(base_name * "_" * label * "_2d", last_run[:z];
                    x=x2, fields=fields2, formats=(:vtk,)))
                append!(vtk_names, export_thermal_structure(base_name * "_" * label * "_3d", last_run[:z];
                    x=x3, y=x3, fields=fields3, formats=(:vtk,)))
            end
            println("Saved data to $(joinpath(pwd(), jld2_name)) and $(join(vtk_names, ", "))")
        end
    end

    on(but_stop.clicks) do n
        stop_requested[] = true
        println("Stopping simulation...")
    end

    on(but_zircon.clicks) do n
        if sim_running[]
            println("Simulation is still finishing up - wait until it stops completely before computing zircon ages")
        elseif zircon_running[]
            println("Zircon ages are already being computed")
        elseif isempty(tracers_out)
            println("No tracer data yet - run the simulation first")
        else
            zircon_running[] = true
            reservoir = copy(tracers_out)
            cargo = copy(erupted_tracers_out)
            Threads.nthreads() == 1 && println("Zircon calculation uses the only Julia thread; restart with `julia -t auto` to keep the GUI responsive")
            println("Computing zircon ages for $(length(reservoir)) reservoir tracers + $(length(cargo)) erupted-cargo tracers on $(Threads.nthreads()) thread(s)...")
            worker = Threads.@spawn begin
                t_ref = maximum((tr.time_vec[end] for tr in reservoir if length(tr.time_vec) >= 2); init=0.0)
                zircon_result = compute_zircon_ages(reservoir; nx=50, t_ref_Myr=t_ref)
                cargo_result  = isempty(cargo) ? nothing : compute_zircon_ages(cargo; nx=50, t_ref_Myr=t_ref)
                return zircon_result, cargo_result
            end

            # Keep all GLMakie calls on the GUI task. `@async` alone is cooperative and
            # cannot make CPU-bound work responsive; it only waits for the worker here.
            @async try
                zircon_result, cargo_result = fetch(worker)
                if !isempty(zircon_result.age_years)
                    last_run[:zircon_age_years]    = zircon_result.age_years
                    last_run[:zircon_radius_um]    = zircon_result.zircon_radius_um
                    cargo_result !== nothing && (last_run[:zircon_age_years_erupted] = cargo_result.age_years)

                    age_ka = zircon_result.age_years ./ 1e3
                    n = length(age_ka)
                    cargo_ka = (cargo_result === nothing || isempty(cargo_result.age_years)) ? Float64[] : cargo_result.age_years ./ 1e3
                    ne = length(cargo_ka)

                    zircon_fig = Figure(size=(1100,400))
                    zircon_ax  = Axis(zircon_fig[1,1], xlabel="Zircon age [ka]", ylabel="Density",
                                       title="Zircon age distribution (reservoir n=$n, erupted n=$ne)")
                    # ages are ≥0 by construction (age = t_end - growth_midpoint, plus a ≥0
                    # common-clock offset). Constrain the KDE support to [0, ub] so the kernel
                    # tail can't bleed into negative "ages" - there is no pre-run zircon growth.
                    ub = maximum(vcat(age_ka, cargo_ka)) * 1.05
                    density!(zircon_ax, age_ka; boundary=(0.0, ub), color=(:steelblue,0.4), strokecolor=:steelblue, strokewidth=2, label="reservoir (n=$n)")
                    if ne > 1
                        density!(zircon_ax, cargo_ka; boundary=(0.0, ub), color=(:firebrick,0.3), strokecolor=:firebrick, strokewidth=2, label="erupted (n=$ne)")
                    end
                    xlims!(zircon_ax, 0, ub)
                    axislegend(zircon_ax)

                    age_sorted = sort(age_ka)
                    cum_prob   = (1:n) ./ n .* 100
                    cdf_ax = Axis(zircon_fig[1,2], xlabel="Zircon age [ka]", ylabel="Cumulative probability [%]",
                                  title="Zircon age spectrum (ranked order)")
                    stairs!(cdf_ax, age_sorted, cum_prob; step=:post, color=:steelblue, label="reservoir (n=$n)")
                    if ne > 1
                        cargo_sorted = sort(cargo_ka)
                        stairs!(cdf_ax, cargo_sorted, (1:ne) ./ ne .* 100; step=:post, color=:firebrick, label="erupted (n=$ne)")
                    end
                    ylims!(cdf_ax, 0, 100)
                    ne > 1 && axislegend(cdf_ax, position=:rb)

                    # display in its own window first: saving an undisplayed Figure
                    # directly can make GLMakie reuse/reconfigure the main GUI's existing
                    # screen instead of opening an independent one, replacing it on-screen.
                    display(GLMakie.Screen(), zircon_fig; title="Zircon Ages")

                    zircon_name = filename[2].stored_string.val * "_zircon_ages.png"
                    save(zircon_name, zircon_fig)
                    println("Saved zircon age density + cumulative probability plot to $(joinpath(pwd(), zircon_name))")
                    ne > 0 && println("Erupted zircon cargo: $ne datable ages (of $(length(cargo)) extracted tracers)")
                else
                    println("No tracers had enough recorded history to compute zircon ages; skipped zircon age plot")
                end
            catch err
                @error "Zircon age computation failed" exception=(err, catch_backtrace())
            finally
                zircon_running[] = false
            end
        end
    end


    rowsize!(timeseries_fig, 2, Relative(1/2))

    SecYear = 3600*24*365.25
    # Start the simulation
    on(but.clicks) do n
        stop_requested[] = false
        sim_running[] = true
        # Retrieve data from GUI
        SecYear     = 3600*24*365.25
        Δz          = get_valuebox(Δz_box)
        H           = get_valuebox(H_box)
        H > 0 || throw(ArgumentError("Crustal thickness must be positive"))
        Δz > 0 || throw(ArgumentError("Grid spacing must be positive"))
        nz          = round(Int, H*1e3/Δz) + 1
        nz >= 2 || throw(ArgumentError("Grid spacing must not exceed crustal thickness"))
        nt          = get_valuebox(nt_box)
        γ           = get_valuebox(γ_box)
        Tsill       = get_valuebox(Tsill_box)
        Ttop        = get_valuebox(Ttop_box)
        Δt          = get_valuebox(Δt_yrs_box)*SecYear
        Silltop     = get_valuebox(Sill_interval_top_box)
        Sillbot     = get_valuebox(Sill_interval_bot_box)
        Sillthick   = get_valuebox(Sill_thick_box)
        0 <= Silltop < Sillbot <= H || throw(ArgumentError(
            "Injection depths must satisfy 0 ≤ top < bottom ≤ crustal thickness"))
        Sillthick > 0 || throw(ArgumentError("Sill thickness must be positive"))
        ȧ = gui_flux_history(menu_flux.selection[];
            base_m_per_yr=get_valuebox(flux_base_box),
            peak_m_per_yr=get_valuebox(flux_peak_box),
            start_kyr=get_valuebox(flux_start_box),
            end_kyr=get_valuebox(flux_end_box),
            table_path=flux_table_box[2].stored_string.val)
        Ql          = get_valuebox(Ql_box)*1e3
        method      = menu_method.selection[]
        run_discrete = method=="Discrete sills" || method=="Both (compare)"
        run_Qmagma   = method=="Q_magma"        || method=="Both (compare)"

        # eruption = trigger (when) + collapse (how the vent closes); chosen independently
        trigger_method  = menu_trigger.selection[]
        if trigger_method == "D&H 3-phase" &&
           menu_melting.selection[] != "MeltingParam_Rhyolite"
            throw(ArgumentError(
                "D&H 3-phase uses the Liu silicic H₂O law; select MeltingParam_Rhyolite"))
        end
        collapse_method = menu_collapse.selection[]
        erupt_mode      = collapse_method == "Elastic" ? :elastic :
                          collapse_method == "Caldera" ? :caldera : :hybrid
        Eruption_thick  = get_valuebox(Eruption_thick_box)
        # lateral extent of the sill/chamber the 1D column represents: erupted volumes
        # are A_sill * melt thickness, and the elastic box model stores recharge in the
        # chamber's elastic compliance
        R_sill  = get_valuebox(Sill_radius_box)*1e3          # [m]
        A_sill  = pi*R_sill^2                                # [m^2]
        ΔPc     = get_valuebox(dPc_box)*1e6                  # critical overpressure [Pa]
        μ_shear = get_valuebox(mu_box)*1e9                   # host-rock shear modulus [Pa]
        β_magma = get_valuebox(beta_box)/1e9                 # magma compressibility [1/Pa]
        β_eff   = β_magma + 3/(4*μ_shear)                    # spherical-chamber storage [1/Pa]
        z_erupt_max = get_valuebox(zmax_box)*1e3             # max chamber-centroid depth for eruption (all triggers) [m]
        # D&H 3-phase overpressure trigger: one lumped chamber state per model. η_r (wall
        # relaxation viscosity) is no longer a knob — it is recomputed each step from the
        # country-rock T (wall_relaxation_viscosity); the struct default is just a placeholder.
        erupt_params = EruptionParams(ΔP_crit=ΔPc, ϕ_erupt=0.5,
            z_erupt_max=z_erupt_max, z_gas_max=10e3, β_r=1.0/β_eff)


        conductivity = T_Conductivity_Whittington()
        heatcapacity = T_HeatCapacity_Whittington()
        if menu_conduct.selection[]=="Constant conductivity 3 W/m/K"
            conductivity = ConstantConductivity(k=3.0)
            heatcapacity = ConstantHeatCapacity()
        end

        melting = MeltingParam_Smooth3rdOrder()
        if menu_melting.selection[]=="MeltingParam_Assimilation"
            melting = MeltingParam_Assimilation()
        elseif menu_melting.selection[]=="MeltingParam_Rhyolite"
            melting = MeltingParam_Smooth3rdOrder(a=3043.0,b=−10552.0, c=12204.9,d=−4709.0)
        end


        @info "parameters" nz, H, γ, Tsill, Ttop, nt
        Tbot = Ttop +   H*γ

        # setup model. init_model assembles the single MatParam every entry point shares;
        # the host-rock thermal density comes from the same parameter object as the chamber's
        # crustal density, and check_density_consistency refuses a run where the two differ.
        Params, BC, N, Δ, T, z = init_model(nz=nz, L=H*1e3, Geotherm=γ, Ttop=Ttop, Tbot=Tbot, Δt=Δt,
                                            ρ=erupt_params.ρ_crust, Q_L=Ql,
                                            Conductivity=conductivity, HeatCapacity=heatcapacity,
                                            Melting=melting)
        MatParam = Params.MatParam
        last_matparam = MatParam
        check_density_consistency(MatParam, erupt_params)
        Δz = Δ[1]
        T_background = copy(T)

        # second model, evolved with an equivalent steady volumetric source Q_magma
        # instead of discrete sill injection (compared side-by-side in the plots below)
        Params_Q = deepcopy(Params)
        T_Q      = deepcopy(T)
        Params_Q.Told .= T_Q
        # One accretion history drives both emplacement models. Each step's exactly
        # integrated thickness becomes whole sills in the discrete branch and a step-mean
        # source rate in the Q_magma branch.
        A_inj    = 0.0                             # cumulative injected thickness [m]

        rocks = zero(T) # will later contain locations with injected sills
        # same indicator for the smeared branch: magma arrives spread over the injection
        # zone rather than as a sill, and is advected by the same host-rock velocity
        rocks_Q = zero(T_Q)
        zone_lo, zone_hi = -Sillbot*1e3, -Silltop*1e3

        # injection-zone boundary markers, advected by Params_Q.w so the dashed lines
        # on ax2 show how far the host rock at the zone edges has moved under Q_magma
        zone_markers = [-Silltop*1e3, -Sillbot*1e3]

        # Each model owns its tracer population. Sharing one population would make a
        # Q_magma event impossible to reconcile while the same tracers follow discrete
        # sill displacements (and vice versa).
        tracers = run_discrete ? init_tracers(Silltop, Sillbot) : Tracer[]
        tracers_Q = run_Qmagma ? init_tracers(Silltop, Sillbot) : Tracer[]
        erupted_tracers = Tracer[]
        erupted_tracers_Q = Tracer[]
        rng = Random.default_rng()

        # create initial plot
        PlotData = (;ax1, ax2, fig)
        println("Running simulation $n")

        # timestepping
        F = zero(T)
        time = 0.0
        timevec =Observable([0.0, 1.0])
        Tmaxvec =Observable([0.0, 1.0])

        Tplot   = Observable(T)
        ϕplot   = Observable(Params.ϕ)
        TQplot  = Observable(T_Q)
        ϕQplot  = Observable(Params_Q.ϕ)

        empty!(ax1)
        if run_discrete
            lines!(ax1, Tplot,  z/1e3, color=:red, label="discrete sills")
        end
        if run_Qmagma
            lines!(ax1, TQplot, z/1e3, color=:orange, linestyle=:dash, label="Q_magma")
        end
        if run_discrete && run_Qmagma
            axislegend(ax1, position=:lb)
        end
        ax1.limits=(minimum(T)-10, Tsill+100, extrema(z/1e3)...)
        empty!(ax2)
        if run_discrete
            lines!(ax2, ϕplot,  z/1e3, color=:blue)
        end
        if run_Qmagma
            lines!(ax2, ϕQplot, z/1e3, color=:purple, linestyle=:dash)
        end
        ax2.limits=(0,1,extrema(z/1e3)...)
        xlims!(ax3, 0, nt*Δt/SecYear/1e3)
        xlims!(ax4, 0, nt*Δt/SecYear/1e3)

        # Get initial sparsity pattern of matrix
        nz          = N[1]
        J1          = Tridiagonal(ones(nz-1), ones(nz), ones(nz-1))
        J1[1,2] =   0; J1[2,1]=0; J1[nz-1,nz]=0; J1[nz,nz-1]=0
        Jac         =   sparse(Float64.(abs.(J1).>0))
        colors      =   matrix_colors(Jac)

        time_vec  = Float64[]
        flux_vec  = Float64[]
        Tmax_vec  = Float64[]
        ϕmax_vec  = Float64[]
        TQmax_vec = Float64[]
        ϕQmax_vec = Float64[]

        # Cumulative physically withdrawn volume [km³] and realized event history.
        erupted_volume = 0.0
        erupted_volume_vec = Float64[]
        eruption_event_time_vec   = Float64[]   # time [kyrs] of each individual eruption
        eruption_event_volume_vec = Float64[]   # volume [km^3] of that single event
        collapse_event_time_vec = Float64[]     # time [kyrs] of each physical column closure
        collapse_event_thickness_vec = Float64[] # closure amplitude [m]
        surface_subsidence = 0.0
        surface_subsidence_vec = Float64[]
        eruption_trigger_time_vec   = Float64[] # D&H drainage times [kyrs], including sub-grid drainage
        eruption_trigger_volume_vec = Float64[] # drained volume [km³] during each thermal step
        # same bookkeeping for the Q_magma model, which erupts independently based on
        # its own melt fraction
        erupted_volume_Q = 0.0
        erupted_volume_Q_vec = Float64[]
        eruption_event_time_Q_vec   = Float64[]
        eruption_event_volume_Q_vec = Float64[]
        collapse_event_time_Q_vec = Float64[]
        collapse_event_thickness_Q_vec = Float64[]
        surface_subsidence_Q = 0.0
        surface_subsidence_Q_vec = Float64[]
        eruption_trigger_time_Q_vec   = Float64[]
        eruption_trigger_volume_Q_vec = Float64[]
        eruption_events = EruptionEvent[]
        eruption_events_Q = EruptionEvent[]

        enthalpy_budget = EnthalpyBudget(column_enthalpy(T, z, MatParam, Params.Phases))
        enthalpy_budget_Q = EnthalpyBudget(column_enthalpy(T_Q, z, MatParam, Params_Q.Phases))
        enthalpy_budget_vec = NamedTuple[]
        enthalpy_budget_Q_vec = NamedTuple[]

        # melt content of the starting geotherm: Params.ϕ is only filled by the first
        # nonlinear solve, and the budget needs the state it starts from
        compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
        compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
        mass_budget = MassBudget(integrated_content(rocks, z), melt_thickness(Params.ϕ, z, z[1], z[end]))
        mass_budget_Q = MassBudget(integrated_content(rocks_Q, z), melt_thickness(Params_Q.ϕ, z, z[1], z[end]))
        mass_budget_vec = NamedTuple[]
        mass_budget_Q_vec = NamedTuple[]

        # D&H chamber diagnostics vs time (per model). Only pushed when the D&H trigger runs,
        # so length matches time_vec in that mode; empty otherwise. Exported in last_run (JLD2)
        # as the H₂O-evolution history for the inversion workflow.
        dP_vec=Float64[]; mdiss_vec=Float64[]; Xg_vec=Float64[]; phig_vec=Float64[]; rhogas_vec=Float64[]; etar_vec=Float64[]; phimush_vec=Float64[]
        dP_Q_vec=Float64[]; mdiss_Q_vec=Float64[]; Xg_Q_vec=Float64[]; phig_Q_vec=Float64[]; rhogas_Q_vec=Float64[]; etar_Q_vec=Float64[]; phimush_Q_vec=Float64[]

        # chamber overpressure [Pa] for the elastic box model, one per model: recharge
        # into an existing mush inflates the chamber against the elastic walls
        # (storage β_eff); eruption relaxes it back to lithostatic
        P_over   = 0.0
        P_over_Q = 0.0
        # D&H 3-phase trigger carries a lumped chamber state per model instead (evolved
        # every timestep by step_overpressure!); reset fresh for this run
        erupt_state   = EruptionState()
        erupt_state_Q = EruptionState()

        Sill_z0 = NaN

        # perform timestepping
        crust_added = 0.0
        crust_added_numerics = integrated_content(rocks, z)/1e3
        F_Q = zero(T_Q)

        recording = record_toggle[2].active[]
        movie_name = filename[2].stored_string.val * ".mp4"
        # VideoStream reconfigures the Figure's existing screen (it's the same one used for
        # the live display); without `visible=true` it hides the window after the first
        # recorded frame.
        vstream = recording ? VideoStream(fig; framerate=24, visible=true) : nothing
        if recording
            println("Recording movie to $(joinpath(pwd(), movie_name))")
        end

        # Redrawing every axis (empty! + full replot) every timestep is the real cost of
        # the run - the physics itself is microseconds. Throttle the redraw to ~200 frames
        # over the run (every frame while recording). Legends are separate blocks that
        # empty!(ax) does NOT remove, so axislegend must be called ONCE, not per frame -
        # otherwise they stack and smear. These flags make each legend one-shot.
        plot_every    = recording ? 1 : max(1, nt ÷ 200)
        legend2_added = false
        legend5_added = false
        legend6_added = false

        @async try
        for t = 1:nt
            if stop_requested[]
                println("Simulation stopped at timestep $t")
                if recording
                    save(movie_name, vstream)
                    println("Saved movie to $(joinpath(pwd(), movie_name))")
                end
                break
            end

            Δh = injected_thickness(ȧ, time, Params.Δt)
            n_injections = sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            ȧ_step = Δh/Params.Δt          # step-mean accretion rate for the smeared branch
            push!(flux_vec, ȧ_step*SecYear)
            ȧ_discrete = n_injections*Sillthick/Params.Δt
            boundary_step = 0.0
            injected_step = 0.0
            source_step = 0.0
            erupted_step = 0.0
            boundary_step_Q = 0.0
            injected_step_Q = 0.0
            source_step_Q = 0.0
            erupted_step_Q = 0.0
            magma_in_step = 0.0
            magma_out_step = 0.0
            melt_out_step = 0.0
            magma_in_step_Q = 0.0
            magma_out_step_Q = 0.0
            melt_out_step_Q = 0.0

            if run_discrete
                T,  converged, its = nonlinear_solution(F, T, Jac, colors, verbose=false, Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam)
                converged || error("Discrete thermal solve failed to converge at timestep $t after $its iterations")
                boundary_step += conductive_boundary_energy(T, Params.k, z, Params.Δt)
                source_step += source_energy(Params.Q, z, Params.Δt)
                compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = T .+ 273.15,))

                for _ in 1:n_injections

                    Sill_z0 = rand(rng, -Sillbot*1e3:1:-Silltop*1e3)
                    T_host = linear_interpolation(z, T)(Sill_z0)
                    injected_step += magma_heat_input(T_host, Tsill, Sillthick, MatParam)

                    T, rocks, magma_lost = insert_sill(T,rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                    magma_in_step += Sillthick
                    magma_out_step += magma_lost
                    Params.Told .= T
                    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases,
                        (T = Params.Told .+ 273.15,))

                    advect_tracers_sill!(tracers, Sill_z0, Sillthick)
                    add_sill_tracers!(tracers, Sill_z0, Sillthick, Tsill)

                    if trigger_method == "Elastic box model"
                        # a sill injected into an existing mush pressurizes the chamber
                        # against the elastic walls; with no mush the dike just freezes
                        region_p = find_eruptible_region(Params.ϕ, z; ϕ_threshold=0.5)
                        if region_p !== nothing
                            h_band = region_p[2] - region_p[1]
                            P_over += Sillthick/(max(h_band, Sillthick)*β_eff)
                        end
                    end

                    crust_added += Sillthick/1e3
                    crust_added_numerics = integrated_content(rocks, z)/1e3
                    println("Injecting sill @ z=$Sill_z0")
                end
                Params.Told .= T
                compute_meltfraction!(Params.ϕ, MatParam, Params.Phases,
                    (T = Params.Told .+ 273.15,))
            end

            if run_Qmagma
                # same physics, but with sills smeared into a steady
                # volumetric source Q_magma instead of discrete injection events
                compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ_step, Silltop=Silltop, Sillbot=Sillbot)
                # the injected-magma indicator rides the same host-rock displacement as the
                # column, then takes this step's smeared delivery spread over the zone
                rocks_Q_adv = conservative_advection(rocks_Q, Params_Q.w .* Params_Q.Δt, z)
                magma_out_step_Q += integrated_content(rocks_Q, z) - integrated_content(rocks_Q_adv, z)
                rocks_Q = rocks_Q_adv
                add_uniform_content!(rocks_Q, z, zone_lo, zone_hi, Δh)
                magma_in_step_Q += Δh
                advect_w!(Params_Q)   # semi-Lagrangian host-rock displacement, as with discrete sills
                compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ_step, Silltop=Silltop, Sillbot=Sillbot)
                advect_markers!(zone_markers, Params_Q)
                advect_tracers!(tracers_Q, Params_Q)
                for _ in 1:n_injections
                    # replenish tracers at the zone center, since host rock is
                    # continuously advected away from it under Q_magma
                    add_zone_tracers!(tracers_Q, Silltop, Sillbot, Tsill)
                end
                T_Q, converged_Q, its_Q = nonlinear_solution(F_Q, T_Q, Jac, colors, verbose=false, Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam)
                converged_Q || error("Q_magma thermal solve failed to converge at timestep $t after $its_Q iterations")
                boundary_step_Q += conductive_boundary_energy(T_Q, Params_Q.k, z, Params_Q.Δt)
                source_step_Q += source_energy(Params_Q.Q, z, Params_Q.Δt)
                Params_Q.Told .= T_Q
                compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases,
                    (T = Params_Q.Told .+ 273.15,))
            end

            time += Params.Δt
            time_kyrs = time/SecYear/1e3
            time_Myr  = time_kyrs/1e3

            if run_discrete
                update_tracers_T!(tracers, T, z, time_Myr, Params.ϕ)
            end
            run_Qmagma && update_tracers_T!(tracers_Q, T_Q, z, time_Myr, Params_Q.ϕ)

            dh_mode = false   # hoisted so it's in scope for the chamber-diagnostics push below
            if trigger_method != "None"
                # trigger decides WHEN, erupt_mode (from menu_collapse) decides HOW the vent
                # closes - any trigger pairs with any collapse
                box_mode = trigger_method == "Elastic box model"
                dh_mode  = trigger_method == "D&H 3-phase"
                # skip eruptions whose footprint reaches the domain edges: erupt_melt!'s
                # collapse needs real host rock on both sides to flow inward, and there's
                # none left to draw on once the melt zone touches the surface or the
                # domain's bottom boundary
                margin = 5*Δz

                # each model erupts independently, based on its own melt fraction
                if run_discrete
                    region = find_eruptible_region(Params.ϕ, z; ϕ_threshold=0.5)
                    if region !== nothing
                        z_lo, z_hi = region
                        thickness = z_hi - z_lo
                        near_boundary = (z_lo - margin <= z[1]) || (z_hi + margin >= z[end])
                        # only the band's melt content can leave the column (the crystal
                        # framework stays): threshold modes erupt it all at once; the box
                        # model erupts the volume held in the chamber's elastic storage
                        h_melt  = melt_thickness(Params.ϕ, z, z_lo, z_hi)
                        # D&H: evolve the lumped chamber overpressure this step from the mush
                        # (T,ϕ); erupts on ΔP≥ΔP_crit, releasing the elastic-storage volume (b)
                        zc_dh = 0.0
                        if dh_mode
                            ind_e, V_e, zc_dh = eruptible_mush(Params.ϕ, z; ϕ_erupt=erupt_params.ϕ_erupt)
                            if V_e > 0
                                update_lithostatic!(erupt_state, erupt_params, zc_dh)
                                T_mush_K = sum(T[ind_e])/length(ind_e) + 273.15
                                ϕ_mush   = sum(Params.ϕ[ind_e])/length(ind_e)
                                # item 2a: η_r computed from the (cooler, maturing) country rock
                                # just outside the mush; as the crust heats over the run η_r drops
                                # and the chamber crosses storing -> erupting on its own
                                iw_lo = max(1, minimum(ind_e)-1); iw_hi = min(length(T), maximum(ind_e)+1)
                                T_wall_K = 0.5*(T[iw_lo] + T[iw_hi]) + 273.15
                                erupt_params.η_r = wall_relaxation_viscosity(erupt_params, T_wall_K)
                                # sub-steps the overpressure ODE and drains internally on every
                                # ΔP≥ΔP_crit crossing (depth + gas-lock gated); erupt_state.h_erupt
                                # is the total drained melt this step, mass-conserving (=recharge)
                                step_overpressure!(erupt_state, erupt_params, T_mush_K, ϕ_mush, V_e, ȧ_discrete, Params.Δt; z_centroid=zc_dh)
                            end
                        end
                        h_requested = dh_mode ? erupt_state.h_erupt : 0.0
                        if h_requested > 0
                            push!(eruption_trigger_time_vec, time_kyrs)
                            push!(eruption_trigger_volume_vec, A_sill*h_requested/1e9)
                        end
                        # D&H queues sub-grid drainage until the physical withdrawal is
                        # resolvable; box uses elastic storage; threshold erupts the whole band.
                        h_erupt = box_mode ? min(β_eff*thickness*P_over, h_melt) :
                                  dh_mode  ? pending_withdrawal!(erupt_state, h_requested, h_melt, Δz; time) : h_melt
                        # Maximum depth applies to every trigger; D&H checks it internally too.
                        fires = eruption_fires(trigger_method; thickness, threshold=Eruption_thick,
                            overpressure=P_over, pressure_critical=ΔPc, h_erupt, z_lo, z_hi,
                            z_erupt_max, near_boundary, Δz)
                        if fires
                            Erupt_z0 = (z_lo + z_hi)/2
                            sills_before = integrated_content(rocks, z)
                            aggregated = dh_mode && erupt_state.h_pending > h_requested + 64eps(max(h_erupt, 1.0))
                            T, rocks, erupted, event = realize_eruption!(rng, T, rocks,
                                tracers, Params.ϕ, z, MatParam, Params.Phases;
                                realization_time=time,
                                trigger_time=dh_mode ? erupt_state.pending_since : time,
                                h_requested=h_erupt, h_booked=h_erupt, z_lo, z_hi,
                                trigger=trigger_method, closure=erupt_mode, aggregated,
                                eligible_phase=nothing)
                            append!(erupted_tracers, erupted)
                            push!(eruption_events, event)
                            erupted_step += event.erupted_enthalpy
                            magma_out_step += event.magma_removed
                            melt_out_step += max(0.0, event.melt_removed)
                            sills_after = integrated_content(rocks, z)
                            println("Intruded sills before eruption: $(round(sills_before, digits=1)) m, after: $(round(sills_after, digits=1)) m (erupted thickness: $(round(h_erupt, digits=1)) m)")
                            println("  event enthalpy residual ($(erupt_mode)): $(round(event.enthalpy_residual/1e9, digits=3)) GJ/m²")
                            Params.Told .= T
                            # ϕ was computed from the pre-eruption T during this step's
                            # nonlinear solve and isn't otherwise refreshed until next
                            # timestep - recompute it now so the melt-fraction plot and the
                            # eruptibility check both see the post-eruption state immediately
                            compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
                            # remaining tracers ride the host rock through the closure
                            collapse_tracers!(tracers, Erupt_z0, h_erupt; method=erupt_mode)
                            dh_mode && commit_pending_withdrawal!(erupt_state, event.state_withdrawn)
                            event_volume = A_sill*event.booked / 1e9   # m^3 -> km^3
                            erupted_volume += event_volume
                            push!(eruption_event_time_vec, time_kyrs)
                            push!(eruption_event_volume_vec, event_volume)
                            push!(collapse_event_time_vec, time_kyrs)
                            push!(collapse_event_thickness_vec, event.state_withdrawn)
                            surface_subsidence += collapse_surface_subsidence(erupt_mode, event.state_withdrawn)
                            box_mode && (P_over = 0.0)   # chamber relaxed to lithostatic
                            # D&H: ΔP already reset inside step_overpressure! on each drain
                            println("Eruption (discrete) @ z=$Erupt_z0, erupted thickness=$(round(event.booked, digits=2)) m, volume=$(round(event_volume, digits=4)) km^3, cumulative=$(round(erupted_volume, digits=4)) km^3")
                        end
                    end
                end

                if run_Qmagma
                    region = find_eruptible_region(Params_Q.ϕ, z; ϕ_threshold=0.5)
                    if region !== nothing
                        z_lo, z_hi = region
                        thickness = z_hi - z_lo
                        # the steady Q_magma recharge pressurizes the chamber continuously
                        if box_mode
                            P_over_Q += Δh/(max(thickness, Sillthick)*β_eff)
                        end
                        zc_dh = 0.0
                        if dh_mode
                            ind_e, V_e, zc_dh = eruptible_mush(Params_Q.ϕ, z; ϕ_erupt=erupt_params.ϕ_erupt)
                            if V_e > 0
                                update_lithostatic!(erupt_state_Q, erupt_params, zc_dh)
                                T_mush_K = sum(T_Q[ind_e])/length(ind_e) + 273.15
                                ϕ_mush   = sum(Params_Q.ϕ[ind_e])/length(ind_e)
                                # item 2a: η_r computed from the maturing country rock at the
                                # mush edge (see the Params model above)
                                iw_lo = max(1, minimum(ind_e)-1); iw_hi = min(length(T_Q), maximum(ind_e)+1)
                                T_wall_K = 0.5*(T_Q[iw_lo] + T_Q[iw_hi]) + 273.15
                                erupt_params.η_r = wall_relaxation_viscosity(erupt_params, T_wall_K)
                                step_overpressure!(erupt_state_Q, erupt_params, T_mush_K, ϕ_mush, V_e, ȧ_step, Params_Q.Δt; z_centroid=zc_dh)
                            end
                        end
                        near_boundary = (z_lo - margin <= z[1]) || (z_hi + margin >= z[end])
                        h_melt  = melt_thickness(Params_Q.ϕ, z, z_lo, z_hi)
                        h_requested = dh_mode ? erupt_state_Q.h_erupt : 0.0
                        if h_requested > 0
                            push!(eruption_trigger_time_Q_vec, time_kyrs)
                            push!(eruption_trigger_volume_Q_vec, A_sill*h_requested/1e9)
                        end
                        h_erupt = box_mode ? min(β_eff*thickness*P_over_Q, h_melt) :
                                  dh_mode  ? pending_withdrawal!(erupt_state_Q, h_requested, h_melt, Δz; time) : h_melt
                        fires = eruption_fires(trigger_method; thickness, threshold=Eruption_thick,
                            overpressure=P_over_Q, pressure_critical=ΔPc, h_erupt, z_lo, z_hi,
                            z_erupt_max, near_boundary, Δz)
                        if fires
                            Erupt_z0 = (z_lo + z_hi)/2
                            aggregated = dh_mode && erupt_state_Q.h_pending > h_requested + 64eps(max(h_erupt, 1.0))
                            T_Q, rocks_Q, erupted, event = realize_eruption!(rng, T_Q, rocks_Q,
                                tracers_Q, Params_Q.ϕ, z, MatParam, Params_Q.Phases;
                                realization_time=time,
                                trigger_time=dh_mode ? erupt_state_Q.pending_since : time,
                                h_requested=h_erupt, h_booked=h_erupt, z_lo, z_hi,
                                trigger=trigger_method, closure=erupt_mode, aggregated,
                                eligible_phase=nothing)
                            append!(erupted_tracers_Q, erupted)
                            push!(eruption_events_Q, event)
                            erupted_step_Q += event.erupted_enthalpy
                            magma_out_step_Q += event.magma_removed
                            melt_out_step_Q += max(0.0, event.melt_removed)
                            Params_Q.Told .= T_Q
                            compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
                            # the injection-zone boundary markers (dashed lines in ax2)
                            # ride on the host rock through the closure
                            collapse_markers!(zone_markers, Erupt_z0, h_erupt; method=erupt_mode)
                            # remaining tracers ride the host rock through the closure
                            collapse_tracers!(tracers_Q, Erupt_z0, h_erupt; method=erupt_mode)
                            dh_mode && commit_pending_withdrawal!(erupt_state_Q, event.state_withdrawn)
                            event_volume = A_sill*event.booked / 1e9   # m^3 -> km^3
                            erupted_volume_Q += event_volume
                            push!(eruption_event_time_Q_vec, time_kyrs)
                            push!(eruption_event_volume_Q_vec, event_volume)
                            push!(collapse_event_time_Q_vec, time_kyrs)
                            push!(collapse_event_thickness_Q_vec, event.state_withdrawn)
                            surface_subsidence_Q += collapse_surface_subsidence(erupt_mode, event.state_withdrawn)
                            box_mode && (P_over_Q = 0.0)   # chamber relaxed to lithostatic
                            # D&H: ΔP already reset inside step_overpressure! on each drain
                            println("Eruption (Q_magma) @ z=$Erupt_z0, erupted thickness=$(round(event.booked, digits=2)) m, volume=$(round(event_volume, digits=4)) km^3, cumulative=$(round(erupted_volume_Q, digits=4)) km^3")
                        end
                    end
                end
            end
            if run_discrete
                update_enthalpy_budget!(enthalpy_budget,
                    column_enthalpy(T, z, MatParam, Params.Phases);
                    boundary=boundary_step, injected=injected_step,
                    source=source_step, erupted=erupted_step)
                push!(enthalpy_budget_vec, enthalpy_budget_snapshot(enthalpy_budget))
                # ϕ is left over from the solve, which ran on the pre-injection T; the
                # budgets and the plots below need the melt fraction of the state the step
                # actually ends in
                compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
                update_mass_budget!(mass_budget, integrated_content(rocks, z),
                    melt_thickness(Params.ϕ, z, z[1], z[end]);
                    injected=magma_in_step, withdrawn=magma_out_step, erupted=melt_out_step)
                push!(mass_budget_vec, mass_budget_snapshot(mass_budget))
            end
            if run_Qmagma
                update_enthalpy_budget!(enthalpy_budget_Q,
                    column_enthalpy(T_Q, z, MatParam, Params_Q.Phases);
                    boundary=boundary_step_Q, injected=injected_step_Q,
                    source=source_step_Q, erupted=erupted_step_Q)
                push!(enthalpy_budget_Q_vec, enthalpy_budget_snapshot(enthalpy_budget_Q))
                compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
                update_mass_budget!(mass_budget_Q, integrated_content(rocks_Q, z),
                    melt_thickness(Params_Q.ϕ, z, z[1], z[end]);
                    injected=magma_in_step_Q, withdrawn=magma_out_step_Q, erupted=melt_out_step_Q)
                push!(mass_budget_Q_vec, mass_budget_snapshot(mass_budget_Q))
            end
            push!(erupted_volume_vec, erupted_volume)
            push!(erupted_volume_Q_vec, erupted_volume_Q)
            push!(surface_subsidence_vec, surface_subsidence)
            push!(surface_subsidence_Q_vec, surface_subsidence_Q)

            push!(time_vec, time_kyrs)
            if run_discrete
                push!(Tmax_vec, maximum(T))
                push!(ϕmax_vec, maximum(Params.ϕ))
            end
            if run_Qmagma
                push!(TQmax_vec, maximum(T_Q))
                push!(ϕQmax_vec, maximum(Params_Q.ϕ))
            end

            # D&H chamber H₂O speciation + wall viscosity vs time (diagnostics set inside
            # step_overpressure!; 0 before the mush forms). Only in D&H mode, so these
            # vectors stay aligned with time_vec.
            if dh_mode
                if run_discrete
                    push!(dP_vec, erupt_state.P - erupt_state.P_lith); push!(mdiss_vec, erupt_state.m_diss)
                    push!(Xg_vec, erupt_state.X_g); push!(phig_vec, erupt_state.ϕ_g)
                    push!(rhogas_vec, erupt_state.ρ_gas); push!(etar_vec, erupt_state.η_r); push!(phimush_vec, erupt_state.ϕ_mush)
                end
                if run_Qmagma
                    push!(dP_Q_vec, erupt_state_Q.P - erupt_state_Q.P_lith); push!(mdiss_Q_vec, erupt_state_Q.m_diss)
                    push!(Xg_Q_vec, erupt_state_Q.X_g); push!(phig_Q_vec, erupt_state_Q.ϕ_g)
                    push!(rhogas_Q_vec, erupt_state_Q.ρ_gas); push!(etar_Q_vec, erupt_state_Q.η_r); push!(phimush_Q_vec, erupt_state_Q.ϕ_mush)
                end
            end

            # redraw only every plot_every steps (and always the final frame): the physics
            # already ran above, this block is pure visualization
            if mod(t, plot_every)==0 || t==nt
                time_val[] = time_kyrs

                empty!(ax2)
                if run_discrete
                    Tplot[] = T
                    ϕplot[] = Params.ϕ
                    rock_low  = Point2f.(zero(rocks), z/1e3)
                    rock_high = Point2f.(clamp.(rocks, 0.0, 1.0), z/1e3)
                    band!(ax2, rock_low, rock_high, color=(:lightgrey,1.0), label="injected sills")
                    lines!(ax2, Params.ϕ, z/1e3, color=:blue, label="discrete sills ϕ")
                end
                if run_Qmagma
                    TQplot[] = T_Q
                    ϕQplot[] = Params_Q.ϕ
                    lines!(ax2, Params_Q.ϕ, z/1e3, color=:purple, linestyle=:dash, label="Q_magma ϕ")
                    hlines!(ax2, zone_markers./1e3, color=:black, linestyle=:dash, linewidth=1)
                end
                if !legend2_added
                    axislegend(ax2, position=:rb, framevisible=true, labelsize=9, patchsize=(12,12))
                    legend2_added = true
                end

                empty!(ax3)
                if run_discrete
                    lines!(ax3, time_vec, Tmax_vec, color=:red)
                    scatter!(ax3, time_vec[end], Tmax_vec[end], color=:red)
                end
                if run_Qmagma
                    lines!(ax3, time_vec, TQmax_vec, color=:orange, linestyle=:dash)
                end
                if run_discrete
                    Tmax_all = vcat(Tmax_vec, run_Qmagma ? TQmax_vec : Float64[])
                    ylims!(ax3, minimum(Tmax_all)-10, maximum(Tmax_all)+10)
                else
                    ylims!(ax3, minimum(TQmax_vec)-10, Tsill+10)
                end

                empty!(ax4)
                if run_discrete
                    lines!(ax4, time_vec, ϕmax_vec, color=:blue)
                    scatter!(ax4, time_vec[end], ϕmax_vec[end], color=:blue)
                end
                if run_Qmagma
                    lines!(ax4, time_vec, ϕQmax_vec, color=:purple, linestyle=:dash)
                end
                ylims!(ax4, 0, 1.01)

                empty!(ax5)
                if run_discrete
                    lines!(ax5, time_vec, erupted_volume_vec, color=:darkgreen, label="cumulative (sills)")
                end
                if run_Qmagma
                    lines!(ax5, time_vec, erupted_volume_Q_vec, color=:darkgreen, linestyle=:dash, label="cumulative (Q_magma)")
                end
                vol_max = max(run_discrete ? maximum(erupted_volume_vec)   : 0.0,
                              run_Qmagma   ? maximum(erupted_volume_Q_vec) : 0.0, 1e-9)
                ylims!(ax5, 0, vol_max*1.1)

                empty!(ax5b)
                if !isempty(eruption_event_time_vec)
                    stem!(ax5b, eruption_event_time_vec, eruption_event_volume_vec, color=:orange, label="per event (sills)")
                end
                if !isempty(eruption_event_time_Q_vec)
                    stem!(ax5b, eruption_event_time_Q_vec, eruption_event_volume_Q_vec, color=:purple, label="per event (Q_magma)")
                end
                event_max = max(isempty(eruption_event_volume_vec)   ? 0.0 : maximum(eruption_event_volume_vec),
                                isempty(eruption_event_volume_Q_vec) ? 0.0 : maximum(eruption_event_volume_Q_vec), 1e-9)
                ylims!(ax5b, 0, event_max*1.1)
                # merged legend for both eruption series (cumulative on ax5, per-event on ax5b)
                if !legend5_added && (run_discrete || run_Qmagma) &&
                   (!isempty(erupted_volume_vec) || !isempty(eruption_event_time_vec) || !isempty(eruption_event_time_Q_vec))
                    axislegend(ax5, position=:lt, framevisible=true, labelsize=9, patchsize=(12,12))
                    legend5_added = true
                end

                # bottom panel: D&H chamber H₂O speciation (left, teal=sills / purple=Q) overlaid
                # with the mush-mean melt fraction ϕ_mush that drives the split (right, dashed).
                empty!(ax6); empty!(ax6b)
                if run_discrete && !isempty(mdiss_vec)
                    lines!(ax6, time_vec, mdiss_vec, color=:teal, label="dissolved H₂O")
                    lines!(ax6, time_vec, Xg_vec, color=:teal, linestyle=:dot, label="exsolved H₂O (gas)")
                    lines!(ax6b, time_vec, phimush_vec, color=:teal, linestyle=:dash, label="ϕ_mush")
                end
                if run_Qmagma && !isempty(mdiss_Q_vec)
                    lines!(ax6, time_vec, mdiss_Q_vec, color=:purple, label="dissolved H₂O (Q)")
                    lines!(ax6, time_vec, Xg_Q_vec, color=:purple, linestyle=:dot, label="exsolved H₂O (Q)")
                    lines!(ax6b, time_vec, phimush_Q_vec, color=:purple, linestyle=:dash, label="ϕ_mush (Q)")
                end
                h2o_max = max(run_discrete && !isempty(mdiss_vec) ? max(maximum(mdiss_vec), maximum(Xg_vec)) : 0.0,
                              run_Qmagma  && !isempty(mdiss_Q_vec) ? max(maximum(mdiss_Q_vec), maximum(Xg_Q_vec)) : 0.0,
                              1e-9)
                ylims!(ax6, 0, h2o_max*1.1)
                ylims!(ax6b, 0, 1.01)
                if !legend6_added && ((run_discrete && !isempty(mdiss_vec)) || (run_Qmagma && !isempty(mdiss_Q_vec)))
                    axislegend(ax6, position=:lt, framevisible=true, labelsize=9, patchsize=(12,12))
                    legend6_added = true
                end

                println("Timestep $t, $time_kyrs kyrs, nz=$(length(T)) pts; crust added: $crust_added km")

                if recording
                    recordframe!(vstream)
                end
            end

        end

        if recording && !stop_requested[]
            save(movie_name, vstream)
            println("Saved movie to $(joinpath(pwd(), movie_name))")
        end

        empty!(last_run)
        last_run[:z] = z
        last_run[:T_background] = T_background
        last_run[:gaussian_sigma] = R_sill
        last_run[:time_vec] = time_vec
        last_run[:flux_m_per_yr] = flux_vec
        last_run[:flux_mode] = ȧ.mode
        last_run[:flux_base_m_per_yr] = ȧ.base*SecYear
        last_run[:flux_peak_m_per_yr] = ȧ.peak*SecYear
        last_run[:flux_start_kyr] = ȧ.t_start/SecYear/1e3
        last_run[:flux_end_kyr] = ȧ.t_end/SecYear/1e3
        last_run[:flux_table_time_kyr] = ȧ.times./SecYear./1e3
        last_run[:flux_table_m_per_yr] = ȧ.rates.*SecYear
        if run_discrete
            last_run[:T] = T
            last_run[:phi] = Params.ϕ
            last_run[:rocks] = rocks
            last_run[:Tmax_vec] = Tmax_vec
            last_run[:phimax_vec] = ϕmax_vec
        end
        if run_Qmagma
            last_run[:T_Qmagma] = T_Q
            last_run[:phi_Qmagma] = Params_Q.ϕ
            last_run[:Tmax_vec_Qmagma] = TQmax_vec
            last_run[:phimax_vec_Qmagma] = ϕQmax_vec
            last_run[:erupted_volume_vec_Qmagma] = erupted_volume_Q_vec
            last_run[:eruption_event_time_vec_Qmagma] = eruption_event_time_Q_vec
            last_run[:eruption_event_volume_vec_Qmagma] = eruption_event_volume_Q_vec
            last_run[:collapse_event_time_vec_Qmagma] = collapse_event_time_Q_vec
            last_run[:collapse_event_thickness_vec_Qmagma] = collapse_event_thickness_Q_vec
            last_run[:surface_subsidence_vec_Qmagma] = surface_subsidence_Q_vec
            last_run[:zone_markers_Qmagma] = copy(zone_markers)
            last_run[:tracers_Qmagma] = tracers_Q
            last_run[:erupted_tracers_Qmagma] = erupted_tracers_Q
            last_run[:eruption_events_Qmagma] = eruption_events_Q
            last_run[:enthalpy_budget_Qmagma] = enthalpy_budget_Q_vec
            last_run[:mass_budget_Qmagma] = mass_budget_Q_vec
            last_run[:rocks_Qmagma] = rocks_Q
        end
        # Generic keys retain their historical meaning: the discrete branch is primary
        # in comparison runs, otherwise they point to the sole Q_magma branch.
        last_run[:tracers] = run_discrete ? tracers : tracers_Q
        last_run[:erupted_tracers] = run_discrete ? erupted_tracers : erupted_tracers_Q
        last_run[:eruption_events] = run_discrete ? eruption_events : eruption_events_Q
        last_run[:enthalpy_budget] = run_discrete ? enthalpy_budget_vec : enthalpy_budget_Q_vec
        last_run[:mass_budget] = run_discrete ? mass_budget_vec : mass_budget_Q_vec
        last_run[:erupted_volume_vec] = erupted_volume_vec
        last_run[:eruption_event_time_vec] = eruption_event_time_vec
        last_run[:eruption_event_volume_vec] = eruption_event_volume_vec
        last_run[:collapse_event_time_vec] = collapse_event_time_vec
        last_run[:collapse_event_thickness_vec] = collapse_event_thickness_vec
        last_run[:surface_subsidence_vec] = surface_subsidence_vec

        # D&H chamber H₂O-evolution history vs time_vec (stepping stone for the inversion):
        # overpressure, dissolved/exsolved H₂O mass fractions, gas volume fraction & density,
        # and the computed wall viscosity η_r. Present only for a D&H-trigger run.
        if trigger_method == "D&H 3-phase" && run_discrete
            last_run[:chamber_dP_vec]     = dP_vec
            last_run[:chamber_mdiss_vec]  = mdiss_vec
            last_run[:chamber_Xg_vec]     = Xg_vec
            last_run[:chamber_phig_vec]   = phig_vec
            last_run[:chamber_rhogas_vec] = rhogas_vec
            last_run[:chamber_etar_vec]   = etar_vec
            last_run[:chamber_phimush_vec] = phimush_vec
            last_run[:eruption_trigger_time_vec] = eruption_trigger_time_vec
            last_run[:eruption_trigger_volume_vec] = eruption_trigger_volume_vec
            last_run[:pending_withdrawal] = erupt_state.h_pending
        end
        if trigger_method == "D&H 3-phase" && run_Qmagma
            last_run[:chamber_dP_vec_Qmagma]     = dP_Q_vec
            last_run[:chamber_mdiss_vec_Qmagma]  = mdiss_Q_vec
            last_run[:chamber_Xg_vec_Qmagma]     = Xg_Q_vec
            last_run[:chamber_phig_vec_Qmagma]   = phig_Q_vec
            last_run[:chamber_rhogas_vec_Qmagma] = rhogas_Q_vec
            last_run[:chamber_etar_vec_Qmagma]   = etar_Q_vec
            last_run[:chamber_phimush_vec_Qmagma] = phimush_Q_vec
            last_run[:eruption_trigger_time_vec_Qmagma] = eruption_trigger_time_Q_vec
            last_run[:eruption_trigger_volume_vec_Qmagma] = eruption_trigger_volume_Q_vec
            last_run[:pending_withdrawal_Qmagma] = erupt_state_Q.h_pending
        end

        # Preserve the existing REPL aliases: discrete is primary in comparison runs.
        global tracers_out = run_discrete ? tracers : tracers_Q
        global erupted_tracers_out = run_discrete ? erupted_tracers : erupted_tracers_Q
        global last_run_out = copy(last_run)
        println("Simulation data available in the REPL: `QMagma.tracers_out` (tracer T-t histories), `QMagma.erupted_tracers_out` (tracers removed by eruption), `QMagma.last_run_out` (profiles, time series, unified eruption_events, cumulative enthalpy_budget, and — for a D&H run — chamber H₂O diagnostics). SAVE DATA writes it all to <filename>.jld2.")
        println("SAVE JLD2 + VTK also writes Gaussian 2D/3D temperature VTKs for each active model, with σ equal to the sill radius and lateral extent ±3σ.")
        println("Compute zircon ages with: `QMagma.compute_zircon_ages(QMagma.tracers_out)`, or click COMPUTE ZIRCON AGES")
        catch err
            @error "Simulation loop failed" exception=(err, catch_backtrace())
        finally
            sim_running[] = false
        end
    end

    screen = display(fig; title="QMagma")

    # center the window on the primary monitor
    try
        monitor   = GLMakie.GLFW.GetPrimaryMonitor()
        vidmode   = GLMakie.GLFW.GetVideoMode(monitor)
        win_w, win_h = size
        x = max(0, div(vidmode.width  - win_w, 2))
        y = max(0, div(vidmode.height - win_h, 2))
        GLMakie.GLFW.SetWindowPos(screen.glscreen, x, y)
    catch e
        @warn "Could not center the window automatically" exception=e
    end

    # Force GLMakie's resize-driven relayout once: the bottom chamber panel (a nested
    # GridLayout in the figure's 2nd row, added after the 1st row) is otherwise not measured
    # until the user manually resizes, so it renders off-window on first display. Nudging the
    # window size by 1px and back fires the resize callback that recomputes the whole layout.
    try
        win_w, win_h = size
        GLMakie.GLFW.SetWindowSize(screen.glscreen, win_w, win_h - 1)
        GLMakie.GLFW.SetWindowSize(screen.glscreen, win_w, win_h)
    catch e
        @warn "Could not nudge the window size to force the initial relayout" exception=e
    end

    return nothing
end

#sill_intrusion_1D()
