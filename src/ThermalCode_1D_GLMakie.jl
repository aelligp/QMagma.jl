# GUI for GLMakie
using GLMakie
using JLD2

export sill_intrusion_1D, compute_zircon_ages, volume_averaged_age

include("ThermalCode_1D.jl")

# populated with the latest run's data at the end of each simulation, so it's reachable
# from the REPL: tracer T-t histories (`QMagma.tracers_out`, for ZirconGrowth.jl) and 1D
# profiles / temporal evolution (`QMagma.last_run_out`)
tracers_out = Tracer[]
last_run_out = Dict{Symbol,Any}()

# Few helpers: 
add_textbox(fig, label, value) = [Label(fig, label), Textbox(fig, stored_string = string(value), validator = typeof(value))]
add_togglebox(fig, label, active) = [Label(fig, label), Toggle(fig, active=active)]
get_valuebox(box::Vector) = parse(box[2].validator.val, box[2].stored_string.val)

"""
    sill_intrusion_1D(; size=(900,900))

Interactive GLMakie App for 1D thermal intrusion model. `size` is the size of the window in pixels.
"""
function sill_intrusion_1D(; size=(1000,1000))
    GLMakie.activate!()
    GLMakie.closeall() # close any open screen

    fig = Figure(size=size)

    time_val = Observable(0.0)
    stop_requested = Observable(false)
    last_run = Dict{Symbol,Any}()

    Label(fig[0, 1:3], text = "1D Sill Injection", fontsize = 30)

    fig[1:2, 1:2] = plots_fig = GridLayout()

    ax1 =  Axis(plots_fig[1, 1], xlabel="Temperature [ᵒC]", ylabel="Depth [km]", title = @lift("t = $(round($time_val, digits = 2)) kyrs"))
    ax2 =  Axis(plots_fig[1, 2], xlabel="Melt fraction ϕ")
    ax3 =  Axis(plots_fig[2, 1:2], xlabel="Time [kyrs]", ylabel="Maximum Temperature [ᵒC]",ytickcolor=:red,ylabelcolor=:red,yticklabelcolor=:red)
    ax4 =  Axis(plots_fig[2, 1:2], ylabel="Maximum melt fraction ϕ",ytickcolor=:blue,ylabelcolor=:blue,yticklabelcolor=:blue,  yaxisposition = :right)

    linkxaxes!(ax3, ax4)

    fig[1:2, 3] = grid = GridLayout(tellwidth = false)
    rowgap!(grid, 2)

    grid[1, 1] = but            = Button(fig, label = "  RUN SIMULATION  ", buttoncolor = :lightgreen)
    grid[1, 2] = but_stop       = Button(fig, label = "  STOP  ", buttoncolor = :red)

    Box(grid[2:4, 1:2], color = :lightgrey, cornerradius = 10)
    grid[2, 1:2] = Δz_box       = add_textbox(fig,"Grid spacing Δz [m]:",20)
    grid[3, 1:2] = nt_box       = add_textbox(fig,"# timesteps nt:",3000)
    grid[4, 1:2] = Δt_yrs_box   = add_textbox(fig,"timestep Δt [yrs]:",100.0)

    Box(grid[5:7, 1:2], color = :lightblue, cornerradius = 10)
    grid[5, 1:2] = H_box        = add_textbox(fig,"Crustal thickness [km]:",40.0)
    grid[6, 1:2] = Ttop_box     = add_textbox(fig,"Ttop [ᵒC]:",0.0)
    grid[7, 1:2] = γ_box        = add_textbox(fig,"Geotherm [ᵒC/km]:",20.0)

    Box(grid[8:12, 1:2], color = :lightyellow, cornerradius = 10)
    grid[8, 1:2] = Tsill_box    = add_textbox(fig,"Sill Temperature [ᵒC]:",1200.0)
    grid[9, 1:2] = Sill_thick_box = add_textbox(fig,"Sill thickness [m]:",100.0)
    grid[10, 1:2] = Sill_interval_box = add_textbox(fig,"Sill injection interval [yrs]:",1000.0)
    grid[11, 1:2] = Sill_interval_top_box = add_textbox(fig,"Top sill injection [km]:",10.0)
    grid[12, 1:2] = Sill_interval_bot_box = add_textbox(fig,"Bottom sill injection [km]:",20.0)

    Box(grid[13:15, 1:2], color = (:red,0.3), cornerradius = 10 )
    grid[13, 1:2] = Ql_box = add_textbox(fig,"Latent heat [kJ/kg]:",255.0)
    grid[14, 1:2] = menu_conduct = Menu(fig, options = ["T-dependent conductivity", "Constant conductivity 3 W/m/K"], default = "Constant conductivity 3 W/m/K")
    grid[15, 1:2] = menu_melting = Menu(fig, options = ["MeltingParam_Assimilation", "MeltingParam_Basalt", "MeltingParam_Rhyolite"], default = "MeltingParam_Basalt")

    Box(grid[16:17, 1:2], color = (:orange,0.3), cornerradius = 10 )
    grid[16, 1:2] = Label(fig, "Method:")
    grid[17, 1:2] = menu_method = Menu(fig, options = ["Discrete sills", "Q_magma", "Both (compare)"], default = "Both (compare)")

    Box(grid[18:21, 1:2], color = (:green,0.3), cornerradius = 10 )
    grid[18, 1:2] = filename = [Label(fig, "filename (no extension):"), Textbox(fig, stored_string = "sim1")]
    grid[19, 1] = but_save =  Button(fig, label = "  SAVE SCREENSHOT  ", buttoncolor = (:lightgreen, 0.5))
    grid[19, 2] = but_save_data = Button(fig, label = "  SAVE DATA  ", buttoncolor = (:lightgreen, 0.5))
    grid[20, 1:2] = record_toggle = add_togglebox(fig,"Record movie:",false)
    grid[21, 1:2] = but_zircon = Button(fig, label = "  COMPUTE ZIRCON AGES  ", buttoncolor = (:lightgreen, 0.5))

    for r in 1:21
        rowsize!(grid, r, Fixed(28))
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
        if isempty(tracers_out)
            println("No tracer data yet - run the simulation first")
        else
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

                zircon_name = filename[2].stored_string.val * "_zircon_ages.png"
                save(zircon_name, zircon_fig)
                println("Saved zircon age density + cumulative probability plot to $(joinpath(pwd(), zircon_name))")
            else
                println("No tracers had enough recorded history to compute zircon ages; skipped zircon age plot")
            end
        end
    end


    rowsize!(plots_fig, 2, Relative(1/4))

    SecYear = 3600*24*365.25
    # Start the simulation
    on(but.clicks) do n
        stop_requested[] = false
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
        end
        last_run[:tracers] = tracers

        global tracers_out = tracers
        global last_run_out = copy(last_run)
        println("Simulation data available in the REPL: `QMagma.tracers_out` (tracer T-t histories), `QMagma.last_run_out` (1D profiles z/T/phi vs depth, and Tmax/phimax vs time)")
        println("Compute zircon ages with: `QMagma.compute_zircon_ages(QMagma.tracers_out)`, or click COMPUTE ZIRCON AGES")
        catch err
            @error "Simulation loop failed" exception=(err, catch_backtrace())
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