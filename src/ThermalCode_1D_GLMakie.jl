# GUI for GLMakie
using GLMakie
using JLD2

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
    last_run = Dict{Symbol,Any}()

    Label(fig[0, 1:3], text = "1D Sill Injection", fontsize = 30)

    fig[1, 1] = depth_fig = GridLayout()
    ax1 =  Axis(depth_fig[1:3, 1], xlabel="Temperature [ᵒC]", ylabel="Depth [km]", title = @lift("t = $(round($time_val, digits = 2)) kyrs"))
    ax2 =  Axis(depth_fig[1:3, 2], xlabel="Melt fraction ϕ")

    fig[1, 2] = timeseries_fig = GridLayout()
    ax3 =  Axis(timeseries_fig[1, 1], xlabel="Time [kyrs]", ylabel="Maximum Temperature [ᵒC]",ytickcolor=:red,ylabelcolor=:red,yticklabelcolor=:red)
    ax4 =  Axis(timeseries_fig[1, 1], ylabel="Maximum melt fraction ϕ",ytickcolor=:blue,ylabelcolor=:blue,yticklabelcolor=:blue,  yaxisposition = :right)
    ax5 =  Axis(timeseries_fig[2, 1], xlabel="Time [kyrs]", ylabel="Cumulative erupted volume [km³]",ytickcolor=:darkgreen,ylabelcolor=:darkgreen,yticklabelcolor=:darkgreen)
    ax5b = Axis(timeseries_fig[2, 1], ylabel="Erupted volume per event [km³]",ytickcolor=:orange,ylabelcolor=:orange,yticklabelcolor=:orange,  yaxisposition = :right)

    linkxaxes!(ax3, ax4)
    linkxaxes!(ax3, ax5)
    linkxaxes!(ax3, ax5b)

    fig[1, 3] = grid = GridLayout(tellwidth = false)
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

    Box(grid[6:8, 1:4], color = :lightyellow, cornerradius = 10)
    grid[6, 1:2] = Tsill_box    = add_textbox(fig,"Sill T [ᵒC]:",1200.0)
    grid[6, 3:4] = Sill_thick_box = add_textbox(fig,"Sill thick [m]:",100.0)
    grid[7, 1:2] = Sill_interval_box = add_textbox(fig,"Sill interval [yrs]:",1000.0)
    grid[8, 1:2] = Sill_interval_top_box = add_textbox(fig,"Top inj. [km]:",10.0)
    grid[8, 3:4] = Sill_interval_bot_box = add_textbox(fig,"Bottom inj. [km]:",20.0)

    Box(grid[9:11, 1:4], color = (:red,0.3), cornerradius = 10 )
    grid[9, 1:2] = Ql_box = add_textbox(fig,"Latent heat [kJ/kg]:",255.0)
    grid[10, 1:4] = menu_conduct = Menu(fig, options = ["T-dependent conductivity", "Constant conductivity 3 W/m/K"], default = "Constant conductivity 3 W/m/K", height=18, fontsize=11)
    grid[11, 1:4] = menu_melting = Menu(fig, options = ["MeltingParam_Assimilation", "MeltingParam_Basalt", "MeltingParam_Rhyolite"], default = "MeltingParam_Basalt", height=18, fontsize=11)

    Box(grid[12:13, 1:4], color = (:orange,0.3), cornerradius = 10 )
    grid[12, 1:4] = Label(fig, "Method:")
    grid[13, 1:4] = menu_method = Menu(fig, options = ["Discrete sills", "Q_magma", "Both (compare)"], default = "Both (compare)", height=18, fontsize=11)

    Box(grid[14:18, 1:4], color = (:purple,0.2), cornerradius = 10 )
    grid[14, 1:4] = Label(fig, "Eruption method:")
    grid[15, 1:4] = menu_eruption = Menu(fig, options = ["None", "Caldera collapse", "Elastic collapse", "Hybrid collapse", "Elastic box model"], default = "None", height=18, fontsize=11)
    grid[16, 1:2] = Eruption_thick_box = add_textbox(fig,"Threshold [m]:",500.0)
    grid[16, 3:4] = Sill_radius_box    = add_textbox(fig,"Sill radius [km]:",5.0)
    grid[17, 1:2] = dPc_box            = add_textbox(fig,"ΔP crit [MPa]:",20.0)
    grid[17, 3:4] = mu_box             = add_textbox(fig,"μ shear [GPa]:",10.0)
    grid[18, 1:2] = beta_box           = add_textbox(fig,"β magma [1/GPa]:",0.1)

    Box(grid[19:22, 1:4], color = (:green,0.3), cornerradius = 10 )
    grid[19, 1:2] = filename = [Label(fig, "filename:"), Textbox(fig, stored_string = "sim1", height=18, fontsize=11, textpadding=(4,4,2,2))]
    grid[20, 1:2] = but_save =  Button(fig, label = "  SAVE SCREENSHOT  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)
    grid[20, 3:4] = but_save_data = Button(fig, label = "  SAVE DATA  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)
    grid[21, 1:2] = record_toggle = add_togglebox(fig,"Record movie:",false)
    grid[22, 1:4] = but_zircon = Button(fig, label = "  COMPUTE ZIRCON AGES  ", buttoncolor = (:lightgreen, 0.5), height=18, fontsize=11)

    for r in 1:22
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
            jld2_name = filename[2].stored_string.val * ".jld2"
            jldsave(jld2_name; last_run...)
            println("Saved data to $(joinpath(pwd(), jld2_name))")
        end
    end

    on(but_stop.clicks) do n
        stop_requested[] = true
        println("Stopping simulation...")
    end

    on(but_zircon.clicks) do n
        if sim_running[]
            println("Simulation is still finishing up - wait until it stops completely before computing zircon ages")
        elseif isempty(tracers_out)
            println("No tracer data yet - run the simulation first")
        else
            # compute_zircon_ages is CPU-heavy (and itself multi-threaded via
            # Threads.@threads), so run it off the GLMakie event thread - calling it
            # directly from this callback can starve/destabilize the render loop and
            # has been observed to kill the GUI window.
            @async try
                println("Computing zircon ages for $(length(tracers_out)) tracers on $(Threads.nthreads()) thread(s)...")
                zircon_result = compute_zircon_ages(tracers_out; nx=50)
                if !isempty(zircon_result.age_years)
                    last_run[:zircon_age_years]    = zircon_result.age_years
                    last_run[:zircon_radius_um]    = zircon_result.zircon_radius_um

                    age_ka = zircon_result.age_years ./ 1e3
                    n = length(age_ka)

                    zircon_fig = Figure(size=(1100,400))
                    zircon_ax  = Axis(zircon_fig[1,1], xlabel="Zircon age [ka]", ylabel="Density",
                                       title="Zircon age distribution (n=$n)")
                    density!(zircon_ax, age_ka)

                    age_sorted = sort(age_ka)
                    cum_prob   = (1:n) ./ n .* 100
                    cdf_ax = Axis(zircon_fig[1,2], xlabel="Zircon age [ka]", ylabel="Cumulative probability [%]",
                                  title="Zircon age spectrum (ranked order)")
                    stairs!(cdf_ax, age_sorted, cum_prob; step=:post)
                    ylims!(cdf_ax, 0, 100)

                    # display in its own window first: saving an undisplayed Figure
                    # directly can make GLMakie reuse/reconfigure the main GUI's existing
                    # screen instead of opening an independent one, replacing it on-screen.
                    display(GLMakie.Screen(), zircon_fig; title="Zircon Ages")

                    zircon_name = filename[2].stored_string.val * "_zircon_ages.png"
                    save(zircon_name, zircon_fig)
                    println("Saved zircon age density + cumulative probability plot to $(joinpath(pwd(), zircon_name))")
                else
                    println("No tracers had enough recorded history to compute zircon ages; skipped zircon age plot")
                end
            catch err
                @error "Zircon age computation failed" exception=(err, catch_backtrace())
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
        nz          = floor(Int64, H*1e3/Δz)
        nt          = get_valuebox(nt_box)
        γ           = get_valuebox(γ_box)
        Tsill       = get_valuebox(Tsill_box)
        Ttop        = get_valuebox(Ttop_box)
        Δt          = get_valuebox(Δt_yrs_box)*SecYear
        Silltop     = get_valuebox(Sill_interval_top_box)
        Sillbot     = get_valuebox(Sill_interval_bot_box)
        Sillthick   = get_valuebox(Sill_thick_box)
        Sill_int_yr = get_valuebox(Sill_interval_box)
        Ql          = get_valuebox(Ql_box)*1e3
        method      = menu_method.selection[]
        run_discrete = method=="Discrete sills" || method=="Both (compare)"
        run_Qmagma   = method=="Q_magma"        || method=="Both (compare)"

        eruption_method = menu_eruption.selection[]
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
        

        MatParam     = (SetMaterialParams(Name="RockMelt", Phase=0, 
                                        Density         = ConstantDensity(ρ=2700kg/m^3),                            # used in the parameterisation of Whittington 
                                        LatentHeat      = ConstantLatentHeat(Q_L=Ql*J/kg),
                                        RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                                        Conductivity    = conductivity,                             #  T-dependent k
                                        HeatCapacity    = heatcapacity,                             # T-dependent cp
                                        Melting         = melting                                   # Quadratic parameterization as in Tierney et al.
        ),)

    

        @info "parameters" nz, H, γ, Tsill, Ttop, nt 
        Tbot = Ttop +   H*γ

        # setup model
        Params, BC, N, Δ, T, z = init_model(nz=nz, L=H*1e3, Geotherm=γ, Ttop=Ttop, Tbot=Tbot, Δt=Δt, MatParam=MatParam)

        # second model, evolved with an equivalent steady volumetric source Q_magma
        # instead of discrete sill injection (compared side-by-side in the plots below)
        Params_Q = deepcopy(Params)
        T_Q      = deepcopy(T)
        Params_Q.Told .= T_Q
        ȧ        = Sillthick/Sill_int_yr/SecYear   # time-averaged accretion rate [m/s]

        rocks = zero(T) # will later contain locations with injected sills

        # injection-zone boundary markers, advected by Params_Q.w so the dashed lines
        # on ax2 show how far the host rock at the zone edges has moved under Q_magma
        zone_markers = [-Silltop*1e3, -Sillbot*1e3]

        # passive tracers: discrete sills take priority when both methods run, since
        # only one tracer set is tracked per run (advect_tracers_sill! vs advect_tracers!
        # use unrelated displacement mechanisms and can't be mixed for the same tracers)
        track_discrete_tracers = run_discrete
        tracers = init_tracers(Silltop, Sillbot)
        erupted_tracers = Tracer[]

        # add initial perturbation (if any)
        T_cen =  (Silltop + Sillbot)/2*1e3

        if run_discrete
            ind = findall( abs.(z .+ T_cen) .< Sillthick/2)
            if !isempty(ind)
                T[ind] .= Tsill
                rocks[ind] .= 1
            end
            Params.Told .= T
        end

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
        if run_discrete
            ax1.limits=(minimum(T)-10, maximum(T)+10,extrema(z/1e3)...)
        else
            ax1.limits=(minimum(T)-10, Tsill+10,extrema(z/1e3)...)
        end
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
        Tmax_vec  = Float64[]
        ϕmax_vec  = Float64[]
        TQmax_vec = Float64[]
        ϕQmax_vec = Float64[]

        # cumulative erupted volume [km^3], assuming each eruption event empties a cube
        # whose edge length is the erupted thickness (Erupt_thick^3)
        erupted_volume = 0.0
        erupted_volume_vec = Float64[]
        eruption_event_time_vec   = Float64[]   # time [kyrs] of each individual eruption
        eruption_event_volume_vec = Float64[]   # volume [km^3] of that single event
        # same bookkeeping for the Q_magma model, which erupts independently based on
        # its own melt fraction
        erupted_volume_Q = 0.0
        erupted_volume_Q_vec = Float64[]
        eruption_event_time_Q_vec   = Float64[]
        eruption_event_volume_Q_vec = Float64[]

        # chamber overpressure [Pa] for the elastic box model, one per model: recharge
        # into an existing mush inflates the chamber against the elastic walls
        # (storage β_eff); eruption relaxes it back to lithostatic
        P_over   = 0.0
        P_over_Q = 0.0

        Sill_z0 = -20e3;
        println("Injecting sill @ z=$Sill_z0")

        # perform timestepping
        crust_added =  Sillthick/1e3
        crust_added_numerics = sum(rocks)*Δz/1e3
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

            if run_discrete
                T,  converged, its = nonlinear_solution(F, T, Jac, colors, verbose=false, Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam)

                if mod(time/SecYear, Sill_int_yr)==0 && t>1

                    Sill_z0 = rand(-Sillbot*1e3:1:-Silltop*1e3)

                    T, rocks = insert_sill(T,rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                    Params.Told .= T

                    if track_discrete_tracers
                        advect_tracers_sill!(tracers, Sill_z0, Sillthick)
                        add_sill_tracers!(tracers, Sill_z0, Sillthick, Tsill)
                    end

                    if eruption_method == "Elastic box model"
                        # a sill injected into an existing mush pressurizes the chamber
                        # against the elastic walls; with no mush the dike just freezes
                        region_p = find_eruptible_region(Params.ϕ, z; ϕ_threshold=0.5)
                        if region_p !== nothing
                            h_band = region_p[2] - region_p[1]
                            P_over += Sillthick/(max(h_band, Sillthick)*β_eff)
                        end
                    end

                    crust_added += Sillthick/1e3
                    crust_added_numerics = sum(rocks)*Δz/1e3
                    println("Injecting sill @ z=$Sill_z0")
                end
                Params.Told .= T
            end

            if run_Qmagma
                # same physics, but with sills smeared into a steady
                # volumetric source Q_magma instead of discrete injection events
                compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
                advect_w!(Params_Q)   # semi-Lagrangian host-rock displacement, as with discrete sills
                advect_markers!(zone_markers, Params_Q)
                if !track_discrete_tracers
                    advect_tracers!(tracers, Params_Q)
                    if mod(time/SecYear, Sill_int_yr)==0 && t>1
                        # replenish tracers at the zone center, since host rock is
                        # continuously advected away from it under Q_magma
                        add_zone_tracers!(tracers, Silltop, Sillbot, Tsill)
                    end
                end
                T_Q, converged_Q, its_Q = nonlinear_solution(F_Q, T_Q, Jac, colors, verbose=false, Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam)
                Params_Q.Told .= T_Q
            end

            time += Params.Δt
            time_kyrs = time/SecYear/1e3
            time_Myr  = time_kyrs/1e3

            if track_discrete_tracers
                update_tracers_T!(tracers, T, z, time_Myr, Params.ϕ)
            else
                update_tracers_T!(tracers, T_Q, z, time_Myr, Params_Q.ϕ)
            end

            if eruption_method != "None"
                # closure kinematics: the box model uses the hybrid closure (the axial
                # trace of half-space chamber deflation); the other modes as named
                box_mode   = eruption_method == "Elastic box model"
                erupt_mode = eruption_method == "Elastic collapse" ? :elastic :
                             eruption_method == "Caldera collapse" ? :caldera : :hybrid
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
                        h_erupt = box_mode ? min(β_eff*thickness*P_over, h_melt) : h_melt
                        trigger = box_mode ? (P_over >= ΔPc) : (thickness >= Eruption_thick)
                        if trigger && !near_boundary && h_erupt > 2*Δz
                            Erupt_z0 = (z_lo + z_hi)/2
                            sills_before = sum(rocks)*Δz
                            T, rocks = erupt_melt!(T, rocks, z; Erupt_z0=Erupt_z0, Erupt_thick=h_erupt, method=erupt_mode)
                            sills_after = sum(rocks)*Δz
                            println("Intruded sills before eruption: $(round(sills_before, digits=1)) m, after: $(round(sills_after, digits=1)) m (erupted thickness: $(round(h_erupt, digits=1)) m)")
                            Params.Told .= T
                            # ϕ was computed from the pre-eruption T during this step's
                            # nonlinear solve and isn't otherwise refreshed until next
                            # timestep - recompute it now so the melt-fraction plot and the
                            # eruptibility check both see the post-eruption state immediately
                            compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
                            if track_discrete_tracers
                                erupted = extract_erupted_tracers!(tracers, Erupt_z0, h_erupt)
                                append!(erupted_tracers, erupted)
                                # remaining tracers ride the host rock through the closure
                                collapse_tracers!(tracers, Erupt_z0, h_erupt; method=erupt_mode)
                            end
                            event_volume = A_sill*h_erupt / 1e9   # m^3 -> km^3
                            erupted_volume += event_volume
                            push!(eruption_event_time_vec, time_kyrs)
                            push!(eruption_event_volume_vec, event_volume)
                            box_mode && (P_over = 0.0)   # chamber relaxed to lithostatic
                            println("Eruption (discrete) @ z=$Erupt_z0, erupted thickness=$(round(h_erupt, digits=1)) m, volume=$(round(event_volume, digits=4)) km^3, cumulative=$(round(erupted_volume, digits=4)) km^3")
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
                            P_over_Q += (ȧ*Params_Q.Δt)/(max(thickness, Sillthick)*β_eff)
                        end
                        near_boundary = (z_lo - margin <= z[1]) || (z_hi + margin >= z[end])
                        h_melt  = melt_thickness(Params_Q.ϕ, z, z_lo, z_hi)
                        h_erupt = box_mode ? min(β_eff*thickness*P_over_Q, h_melt) : h_melt
                        trigger = box_mode ? (P_over_Q >= ΔPc) : (thickness >= Eruption_thick)
                        if trigger && !near_boundary && h_erupt > 2*Δz
                            Erupt_z0 = (z_lo + z_hi)/2
                            T_Q, _ = erupt_melt!(T_Q, zero(T_Q), z; Erupt_z0=Erupt_z0, Erupt_thick=h_erupt, method=erupt_mode)
                            Params_Q.Told .= T_Q
                            compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
                            # the injection-zone boundary markers (dashed lines in ax2)
                            # ride on the host rock through the closure
                            collapse_markers!(zone_markers, Erupt_z0, h_erupt; method=erupt_mode)
                            if !track_discrete_tracers
                                erupted = extract_erupted_tracers!(tracers, Erupt_z0, h_erupt)
                                append!(erupted_tracers, erupted)
                                # remaining tracers ride the host rock through the closure
                                collapse_tracers!(tracers, Erupt_z0, h_erupt; method=erupt_mode)
                            end
                            event_volume = A_sill*h_erupt / 1e9   # m^3 -> km^3
                            erupted_volume_Q += event_volume
                            push!(eruption_event_time_Q_vec, time_kyrs)
                            push!(eruption_event_volume_Q_vec, event_volume)
                            box_mode && (P_over_Q = 0.0)   # chamber relaxed to lithostatic
                            println("Eruption (Q_magma) @ z=$Erupt_z0, erupted thickness=$(round(h_erupt, digits=1)) m, volume=$(round(event_volume, digits=4)) km^3, cumulative=$(round(erupted_volume_Q, digits=4)) km^3")
                        end
                    end
                end
            end
            push!(erupted_volume_vec, erupted_volume)
            push!(erupted_volume_Q_vec, erupted_volume_Q)

            push!(time_vec, time_kyrs)
            if run_discrete
                push!(Tmax_vec, maximum(T))
                push!(ϕmax_vec, maximum(Params.ϕ))
            end
            if run_Qmagma
                push!(TQmax_vec, maximum(T_Q))
                push!(ϕQmax_vec, maximum(Params_Q.ϕ))
            end

            # save file to disk
            if mod(t,1)==0
                time_val[] = time_kyrs

                empty!(ax2)
                if run_discrete
                    Tplot[] = T
                    ϕplot[] = Params.ϕ
                    rock_low  = Point2f.(zero(rocks), z/1e3)
                    rock_high = Point2f.(rocks, z/1e3)
                    band!(ax2, rock_low, rock_high, color=(:lightgrey,1.0))
                    lines!(ax2, Params.ϕ, z/1e3, color=:blue)
                end
                if run_Qmagma
                    TQplot[] = T_Q
                    ϕQplot[] = Params_Q.ϕ
                    lines!(ax2, Params_Q.ϕ, z/1e3, color=:purple, linestyle=:dash)
                    hlines!(ax2, zone_markers./1e3, color=:black, linestyle=:dash, linewidth=1)
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
                    lines!(ax5, time_vec, erupted_volume_vec, color=:darkgreen)
                end
                if run_Qmagma
                    lines!(ax5, time_vec, erupted_volume_Q_vec, color=:darkgreen, linestyle=:dash)
                end
                vol_max = max(run_discrete ? maximum(erupted_volume_vec)   : 0.0,
                              run_Qmagma   ? maximum(erupted_volume_Q_vec) : 0.0, 1e-9)
                ylims!(ax5, 0, vol_max*1.1)

                empty!(ax5b)
                if !isempty(eruption_event_time_vec)
                    stem!(ax5b, eruption_event_time_vec, eruption_event_volume_vec, color=:orange)
                end
                if !isempty(eruption_event_time_Q_vec)
                    stem!(ax5b, eruption_event_time_Q_vec, eruption_event_volume_Q_vec, color=:purple)
                end
                event_max = max(isempty(eruption_event_volume_vec)   ? 0.0 : maximum(eruption_event_volume_vec),
                                isempty(eruption_event_volume_Q_vec) ? 0.0 : maximum(eruption_event_volume_Q_vec), 1e-9)
                ylims!(ax5b, 0, event_max*1.1)

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
        last_run[:time_vec] = time_vec
        if run_discrete
            last_run[:T] = T
            last_run[:phi] = Params.ϕ
            last_run[:Tmax_vec] = Tmax_vec
            last_run[:phimax_vec] = ϕmax_vec
        end
        if run_Qmagma
            last_run[:T_Qmagma] = T_Q
            last_run[:phi_Qmagma] = Params_Q.ϕ
            last_run[:Tmax_vec_Qmagma] = TQmax_vec
            last_run[:phimax_vec_Qmagma] = ϕQmax_vec
            last_run[:erupted_volume_vec_Qmagma] = erupted_volume_Q_vec
        end
        last_run[:tracers] = tracers
        last_run[:erupted_tracers] = erupted_tracers
        last_run[:erupted_volume_vec] = erupted_volume_vec

        global tracers_out = tracers
        global erupted_tracers_out = erupted_tracers
        global last_run_out = copy(last_run)
        println("Simulation data available in the REPL: `QMagma.tracers_out` (tracer T-t histories), `QMagma.erupted_tracers_out` (tracers removed by eruption), `QMagma.last_run_out` (1D profiles z/T/phi vs depth, Tmax/phimax vs time, and erupted_volume_vec)")
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

    return nothing
end

#sill_intrusion_1D()