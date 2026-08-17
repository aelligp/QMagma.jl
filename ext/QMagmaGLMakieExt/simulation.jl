# The timestep loop driven by the RUN SIMULATION button.

"""
    wire_simulation!(ui)

Attach the RUN SIMULATION handler to the layout `ui`. Each click reads the control panel,
sets up both emplacement models, and runs the timestep loop in an `@async` task that
refreshes the plots, so the window stays responsive and STOP can interrupt it.
"""
function wire_simulation!(ui)
    (;
        fig, ax1, ax2, ax3, ax4, ax5, ax5b, ax6, ax6b,
        but, but_stop, but_save, but_save_data, but_zircon,
        Δz_box, nt_box, Δt_yrs_box, H_box, Ttop_box, γ_box,
        Tsill_box, Sill_thick_box, menu_flux,
        flux_base_box, flux_peak_box, flux_start_box, flux_end_box, flux_table_box,
        Sill_interval_top_box, Sill_interval_bot_box,
        Ql_box, menu_conduct, menu_melting, menu_method,
        menu_trigger, menu_collapse, Sill_radius_box,
        dPc_box, G, mw_box, hmelt_box,
        filename, record_toggle,
        time_val, stop_requested, sim_running, zircon_running, last_run, last_matparam,
    ) = ui

    # Start the simulation
    on(but.clicks) do n
        stop_requested[] = false
        sim_running[] = true
        # Retrieve data from GUI
        Δz = get_valuebox(Δz_box)
        H = get_valuebox(H_box)
        H > 0 || throw(ArgumentError("Crustal thickness must be positive"))
        Δz > 0 || throw(ArgumentError("Grid spacing must be positive"))
        nz = round(Int, H * 1.0e3 / Δz) + 1
        nz >= 2 || throw(ArgumentError("Grid spacing must not exceed crustal thickness"))
        nt = get_valuebox(nt_box)
        γ = get_valuebox(γ_box)
        Tsill = get_valuebox(Tsill_box)
        Ttop = get_valuebox(Ttop_box)
        Δt = get_valuebox(Δt_yrs_box) * SecYear
        Silltop = get_valuebox(Sill_interval_top_box)
        Sillbot = get_valuebox(Sill_interval_bot_box)
        Sillthick = get_valuebox(Sill_thick_box)
        Sillthick > 0 || throw(ArgumentError("Sill thickness must be positive"))
        ȧ = gui_flux_history(
            menu_flux.selection[];
            base_m_per_yr = get_valuebox(flux_base_box),
            peak_m_per_yr = get_valuebox(flux_peak_box),
            start_kyr = get_valuebox(flux_start_box),
            end_kyr = get_valuebox(flux_end_box),
            table_path = flux_table_box[2].stored_string.val
        )
        if isempty(ȧ.depths)
            0 <= Silltop < Sillbot <= H || throw(
                ArgumentError(
                    "Injection depths must satisfy 0 ≤ top < bottom ≤ crustal thickness"
                )
            )
        else
            half_sill = Sillthick / 2
            all(d -> half_sill <= d <= H * 1.0e3 - half_sill, ȧ.depths) || throw(
                ArgumentError(
                    "CSV depths must keep the full sill inside the crustal domain"
                )
            )
        end
        Ql = get_valuebox(Ql_box) * 1.0e3
        method = menu_method.selection[]
        run_discrete = method == "Discrete sills" || method == "Both (compare)"
        run_Qmagma = method == "Q_magma"        || method == "Both (compare)"

        # eruption = trigger (when) + collapse (how the vent closes); chosen independently
        trigger_method = menu_trigger.selection[]
        collapse_method = menu_collapse.selection[]
        erupt_mode = collapse_method == "Caldera" ? :caldera : :hybrid
        # lateral extent of the sill/chamber the 1D column represents: erupted volumes
        # are A_sill * bulk erupted thickness, with A_sill the plan-view area of the
        # crack-shaped body the column is the axis of
        R_sill = get_valuebox(Sill_radius_box) * 1.0e3          # [m]
        A_sill = lateral_effective_area(R_sill)               # [m^2]
        ΔPc = get_valuebox(dPc_box) * 1.0e6                  # critical overpressure [Pa]
        μ_shear = get_valuebox(G) * 1.0e9                   # host-rock shear modulus [Pa]
        # One lumped chamber state per emplacement model, both reading these parameters.
        # η_r is not among them: step_chamber_eruption! derives it per model from that
        # model's own country-rock T, so the two chambers cannot write through each other.
        m_w = get_valuebox(mw_box) / 100                    # total magmatic H₂O [wt%] -> mass fraction
        h_melt_min = get_valuebox(hmelt_box)                  # min chamber melt content [m]
        # composition fixes the melting parameterisation and the solubility law together
        composition = gui_composition(menu_melting.selection[])
        melting = composition.melting
        erupt_params = EruptionParams(
            ΔP_crit = ΔPc, ϕ_erupt = 0.5, m_w = m_w, h_melt_min = h_melt_min,
            μ_shear = μ_shear, R_sill = R_sill,
            solubility = composition.solubility,
            melt_viscosity = composition.melt_viscosity
        )


        conductivity = T_Conductivity_Whittington()
        heatcapacity = T_HeatCapacity_Whittington()
        if menu_conduct.selection[] == "Constant conductivity 3 W/m/K"
            conductivity = ConstantConductivity(k = 3.0)
            heatcapacity = ConstantHeatCapacity()
        end


        @info "parameters" nz, H, γ, Tsill, Ttop, nt
        Tbot = Ttop + H * γ

        # setup model. init_model assembles the single MatParam every entry point shares;
        # the host-rock thermal density comes from the same parameter object as the chamber's
        # crustal density, and check_density_consistency refuses a run where the two differ.
        Params, BC, N, Δ, T, z = init_model(
            nz = nz, L = H * 1.0e3, Geotherm = γ, Ttop = Ttop, Tbot = Tbot, Δt = Δt,
            R_lat = R_sill,
            ρ = crust_reference_density(erupt_params), Q_L = Ql,
            Conductivity = conductivity, HeatCapacity = heatcapacity,
            Melting = melting
        )
        MatParam = Params.MatParam
        last_matparam[] = MatParam
        check_density_consistency(MatParam, erupt_params)
        check_sill_temperature(melting, Tsill)
        Δz = Δ[1]
        T_background = copy(T)

        # second model, evolved with an equivalent steady volumetric source Q_magma
        # instead of discrete sill injection (compared side-by-side in the plots below)
        Params_Q = deepcopy(Params)
        T_Q = deepcopy(T)
        Params_Q.Told .= T_Q
        # One accretion history drives both emplacement models. Each step's exactly
        # integrated thickness becomes whole sills in the discrete branch and a step-mean
        # source rate in the Q_magma branch.
        A_inj = 0.0                             # cumulative injected thickness [m]
        depth_forcing = !isempty(ȧ.depths)
        if depth_forcing
            half_sill_km = Sillthick / 2.0e3
            tracer_top = minimum(ȧ.depths) / 1.0e3 - half_sill_km
            tracer_bot = maximum(ȧ.depths) / 1.0e3 + half_sill_km
            initial_depth = injection_depth(ȧ, 0.0) / 1.0e3
            marker_top = initial_depth - half_sill_km
            marker_bot = initial_depth + half_sill_km
        else
            tracer_top, tracer_bot = Silltop, Sillbot
            marker_top, marker_bot = Silltop, Sillbot
        end

        rocks = zero(T) # will later contain locations with injected sills
        # same indicator for the smeared branch: magma arrives spread over the injection
        # zone rather than as a sill, and is advected by the same host-rock velocity
        rocks_Q = zero(T_Q)
        # injection-zone boundary markers, advected by Params_Q.w so the dashed lines
        # on ax2 show how far the host rock at the zone edges has moved under Q_magma
        zone_markers = [-marker_top * 1.0e3, -marker_bot * 1.0e3]

        # Each model owns its tracer population. Sharing one population would make a
        # Q_magma event impossible to reconcile while the same tracers follow discrete
        # sill displacements (and vice versa).
        tracers = run_discrete ? init_tracers(tracer_top, tracer_bot) : Tracer[]
        tracers_Q = run_Qmagma ? init_tracers(tracer_top, tracer_bot) : Tracer[]
        erupted_tracers = Tracer[]
        erupted_tracers_Q = Tracer[]
        rng = Random.default_rng()

        # create initial plot
        PlotData = (; ax1, ax2, fig)
        println("Running simulation $n")

        # timestepping
        F = zero(T)
        time = 0.0
        timevec = Observable([0.0, 1.0])
        Tmaxvec = Observable([0.0, 1.0])

        Tplot = Observable(T)
        ϕplot = Observable(Params.ϕ)
        TQplot = Observable(T_Q)
        ϕQplot = Observable(Params_Q.ϕ)

        empty!(ax1)
        if run_discrete
            lines!(ax1, Tplot, z / 1.0e3, color = :red, label = "discrete sills")
        end
        if run_Qmagma
            lines!(ax1, TQplot, z / 1.0e3, color = :orange, linestyle = :dash, label = "Q_magma")
        end
        if run_discrete && run_Qmagma
            axislegend(ax1, position = :lb)
        end
        ax1.limits = (minimum(T) - 10, Tsill + 100, extrema(z / 1.0e3)...)
        empty!(ax2)
        if run_discrete
            lines!(ax2, ϕplot, z / 1.0e3, color = :blue)
        end
        if run_Qmagma
            lines!(ax2, ϕQplot, z / 1.0e3, color = :purple, linestyle = :dash)
        end
        ax2.limits = (0, 1, extrema(z / 1.0e3)...)
        xlims!(ax3, 0, nt * Δt / SecYear / 1.0e3)
        xlims!(ax4, 0, nt * Δt / SecYear / 1.0e3)

        # Get initial sparsity pattern of matrix
        nz = N[1]
        J1 = Tridiagonal(ones(nz - 1), ones(nz), ones(nz - 1))
        J1[1, 2] = 0; J1[2, 1] = 0; J1[nz - 1, nz] = 0; J1[nz, nz - 1] = 0
        Jac = sparse(Float64.(abs.(J1) .> 0))
        colors = matrix_colors(Jac)

        time_vec = Float64[]
        flux_vec = Float64[]
        Tmax_vec = Float64[]
        ϕmax_vec = Float64[]
        TQmax_vec = Float64[]
        ϕQmax_vec = Float64[]

        # Cumulative physically withdrawn volume [km³] and realized event history.
        erupted_volume = 0.0
        erupted_volume_vec = Float64[]
        eruption_event_time_vec = Float64[]   # time [kyrs] of each individual eruption
        eruption_event_volume_vec = Float64[]   # volume [km^3] of that single event
        collapse_event_time_vec = Float64[]     # time [kyrs] of each physical column closure
        collapse_event_thickness_vec = Float64[] # closure amplitude [m]
        surface_subsidence = 0.0
        surface_subsidence_vec = Float64[]
        eruption_trigger_time_vec = Float64[] # D&H drainage times [kyrs], including sub-grid drainage
        eruption_trigger_volume_vec = Float64[] # drained volume [km³] during each thermal step
        # same bookkeeping for the Q_magma model, which erupts independently based on
        # its own melt fraction
        erupted_volume_Q = 0.0
        erupted_volume_Q_vec = Float64[]
        eruption_event_time_Q_vec = Float64[]
        eruption_event_volume_Q_vec = Float64[]
        collapse_event_time_Q_vec = Float64[]
        collapse_event_thickness_Q_vec = Float64[]
        surface_subsidence_Q = 0.0
        surface_subsidence_Q_vec = Float64[]
        eruption_trigger_time_Q_vec = Float64[]
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
        dP_vec = Float64[]; mdiss_vec = Float64[]; Xg_vec = Float64[]; phig_vec = Float64[]; rhogas_vec = Float64[]; etar_vec = Float64[]; phimush_vec = Float64[]
        dP_Q_vec = Float64[]; mdiss_Q_vec = Float64[]; Xg_Q_vec = Float64[]; phig_Q_vec = Float64[]; rhogas_Q_vec = Float64[]; etar_Q_vec = Float64[]; phimush_Q_vec = Float64[]
        M_vec = Float64[]; MH2O_vec = Float64[]; Mres_vec = Float64[]
        M_Q_vec = Float64[]; MH2O_Q_vec = Float64[]; Mres_Q_vec = Float64[]

        # one lumped chamber state per emplacement model, evolved every timestep by
        # step_overpressure!; reset fresh for this run
        erupt_state = EruptionState()
        erupt_state_Q = EruptionState()

        Sill_z0 = NaN

        # perform timestepping
        crust_added = 0.0
        crust_added_numerics = integrated_content(rocks, z) / 1.0e3
        F_Q = zero(T_Q)

        recording = record_toggle[2].active[]
        movie_name = filename[2].stored_string.val * ".mp4"
        # VideoStream reconfigures the Figure's existing screen (it's the same one used for
        # the live display); without `visible=true` it hides the window after the first
        # recorded frame.
        vstream = recording ? VideoStream(fig; framerate = 24, visible = true) : nothing
        if recording
            println("Recording movie to $(joinpath(pwd(), movie_name))")
        end

        # Redrawing every axis (empty! + full replot) every timestep is the real cost of
        # the run - the physics itself is microseconds. Throttle the redraw to ~200 frames
        # over the run (every frame while recording). Legends are separate blocks that
        # empty!(ax) does NOT remove, so axislegend must be called ONCE, not per frame -
        # otherwise they stack and smear. These flags make each legend one-shot.
        plot_every = recording ? 1 : max(1, nt ÷ 200)
        legend2_added = false
        legend3_added = false
        legend5_added = false
        legend6_added = false

        @async try
            for t in 1:nt
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
                ȧ_step = Δh / Params.Δt          # step-mean accretion rate for the smeared branch
                push!(flux_vec, ȧ_step * SecYear)
                ȧ_discrete = n_injections * Sillthick / Params.Δt
                depth_m = injection_depth(ȧ, time + Params.Δt / 2)
                if depth_m === nothing
                    source_top, source_bot = Silltop, Sillbot
                else
                    half_sill_km = Sillthick / 2.0e3
                    source_top = depth_m / 1.0e3 - half_sill_km
                    source_bot = depth_m / 1.0e3 + half_sill_km
                end
                zone_lo, zone_hi = -source_bot * 1.0e3, -source_top * 1.0e3
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
                    T, converged, its = nonlinear_solution(F, T, Jac, colors, verbose = false, Δ = Δ, N = N, BC = BC, Params = Params, MatParam = MatParam)
                    converged || error("Discrete thermal solve failed to converge at timestep $t after $its iterations")
                    boundary_step += conductive_boundary_energy(T, Params.k, z, Params.Δt) +
                        lateral_loss_energy(T, Params, z, Params.Δt)
                    source_step += source_energy(Params.Q, z, Params.Δt)
                    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = T .+ 273.15,))

                    for _ in 1:n_injections

                        Sill_z0 = depth_m === nothing ?
                            rand(rng, (-Sillbot * 1.0e3):1:(-Silltop * 1.0e3)) : -depth_m
                        T_host = linear_interpolation(z, T)(Sill_z0)
                        injected_step += magma_heat_input(T_host, Tsill, Sillthick, MatParam)

                        T, rocks, magma_lost = insert_sill(T, rocks, z, Sill_thick = Sillthick, Sill_z0 = Sill_z0, Sill_T = Tsill, r = R_sill)
                        magma_in_step += Sillthick
                        magma_out_step += magma_lost
                        Params.Told .= T
                        compute_meltfraction!(
                            Params.ϕ, MatParam, Params.Phases,
                            (T = Params.Told .+ 273.15,)
                        )

                        advect_tracers_sill!(tracers, Sill_z0, Sillthick; r = R_sill)
                        add_sill_tracers!(tracers, Sill_z0, Sillthick, Tsill)

                        crust_added += Sillthick / 1.0e3
                        crust_added_numerics = integrated_content(rocks, z) / 1.0e3
                        println("Injecting sill @ z=$Sill_z0")
                    end
                    Params.Told .= T
                    compute_meltfraction!(
                        Params.ϕ, MatParam, Params.Phases,
                        (T = Params.Told .+ 273.15,)
                    )
                end

                if run_Qmagma
                    # same physics, but with sills smeared into a steady
                    # volumetric source Q_magma instead of discrete injection events
                    depth_forcing && (zone_markers .= [-source_top * 1.0e3, -source_bot * 1.0e3])
                    compute_Q_magma!(Params_Q, MatParam, z; Tsill = Tsill, ȧ = ȧ_step, Silltop = source_top, Sillbot = source_bot, r = R_sill)
                    # the injected-magma indicator rides the same host-rock displacement as the
                    # column, then takes this step's smeared delivery spread over the zone
                    rocks_Q_adv = conservative_advection(rocks_Q, Params_Q.w .* Params_Q.Δt, z)
                    magma_out_step_Q += nonnegative_debit(
                        integrated_content(rocks_Q, z),
                        integrated_content(rocks_Q_adv, z), "injected-magma content"; ncells = length(z)
                    )
                    rocks_Q = rocks_Q_adv
                    add_uniform_content!(rocks_Q, z, zone_lo, zone_hi, Δh)
                    magma_in_step_Q += Δh
                    advect_w!(Params_Q)   # semi-Lagrangian host-rock displacement, as with discrete sills
                    compute_Q_magma!(Params_Q, MatParam, z; Tsill = Tsill, ȧ = ȧ_step, Silltop = source_top, Sillbot = source_bot, r = R_sill)
                    advect_markers!(zone_markers, Params_Q)
                    advect_tracers!(tracers_Q, Params_Q)
                    for _ in 1:n_injections
                        # replenish tracers at the zone center, since host rock is
                        # continuously advected away from it under Q_magma
                        add_zone_tracers!(tracers_Q, source_top, source_bot, Tsill)
                    end
                    T_Q, converged_Q, its_Q = nonlinear_solution(F_Q, T_Q, Jac, colors, verbose = false, Δ = Δ, N = N, BC = BC, Params = Params_Q, MatParam = MatParam)
                    converged_Q || error("Q_magma thermal solve failed to converge at timestep $t after $its_Q iterations")
                    boundary_step_Q += conductive_boundary_energy(T_Q, Params_Q.k, z, Params_Q.Δt) +
                        lateral_loss_energy(T_Q, Params_Q, z, Params_Q.Δt)
                    source_step_Q += source_energy(Params_Q.Q, z, Params_Q.Δt)
                    Params_Q.Told .= T_Q
                    compute_meltfraction!(
                        Params_Q.ϕ, MatParam, Params_Q.Phases,
                        (T = Params_Q.Told .+ 273.15,)
                    )
                end

                time += Params.Δt
                time_kyrs = time / SecYear / 1.0e3
                time_Myr = time_kyrs / 1.0e3

                if run_discrete
                    update_tracers_T!(tracers, T, z, time_Myr, Params.ϕ)
                end
                run_Qmagma && update_tracers_T!(tracers_Q, T_Q, z, time_Myr, Params_Q.ϕ)

                dh_mode = trigger_method != "None"
                if dh_mode
                    # skip eruptions whose footprint reaches the domain edges: erupt_melt!'s
                    # collapse needs real host rock on both sides to flow inward, and there's
                    # none left to draw on once the melt zone touches the surface or the
                    # domain's bottom boundary
                    margin = 5 * Δz

                    # each model runs its own chamber, on its own mush
                    if run_discrete
                        T, rocks, erupted, event = step_chamber_eruption!(
                            rng, erupt_state,
                            erupt_params, T, rocks, tracers, Params.ϕ, z, MatParam, Params.Phases;
                            ȧ = ȧ_discrete, Δt = Params.Δt, time, closure = erupt_mode, margin,
                            T_background
                        )
                        if erupt_state.h_erupt > 0
                            push!(eruption_trigger_time_vec, time_kyrs)
                            push!(eruption_trigger_volume_vec, A_sill * erupt_state.h_erupt / 1.0e9)
                        end
                        if event !== nothing
                            append!(erupted_tracers, erupted)
                            push!(eruption_events, event)
                            erupted_step += event.erupted_enthalpy
                            magma_out_step += event.magma_removed
                            melt_out_step += max(0.0, event.melt_removed)
                            Params.Told .= T
                            # ϕ was computed from the pre-eruption T during this step's
                            # nonlinear solve and isn't otherwise refreshed until next
                            # timestep - recompute it now so the melt-fraction plot and the
                            # eruptibility check both see the post-eruption state immediately
                            compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
                            event_volume = A_sill * event.requested / 1.0e9   # m^3 -> km^3
                            erupted_volume += event_volume
                            push!(eruption_event_time_vec, time_kyrs)
                            push!(eruption_event_volume_vec, event_volume)
                            push!(collapse_event_time_vec, time_kyrs)
                            push!(collapse_event_thickness_vec, event.requested)
                            surface_subsidence += collapse_surface_subsidence(erupt_mode, event.requested)
                            println("Eruption (discrete) @ z=$((event.z_lo + event.z_hi) / 2), erupted thickness=$(round(event.requested, digits = 2)) m, volume=$(round(event_volume, digits = 4)) km^3, cumulative=$(round(erupted_volume, digits = 4)) km^3")
                            println("  event enthalpy residual ($(erupt_mode)): $(round(event.enthalpy_residual / 1.0e9, digits = 3)) GJ/m²")
                        end
                    end

                    if run_Qmagma
                        T_Q, rocks_Q, erupted, event = step_chamber_eruption!(
                            rng, erupt_state_Q,
                            erupt_params, T_Q, rocks_Q, tracers_Q, Params_Q.ϕ, z, MatParam,
                            Params_Q.Phases;
                            ȧ = ȧ_step, Δt = Params_Q.Δt, time, closure = erupt_mode, margin,
                            T_background
                        )
                        if erupt_state_Q.h_erupt > 0
                            push!(eruption_trigger_time_Q_vec, time_kyrs)
                            push!(eruption_trigger_volume_Q_vec, A_sill * erupt_state_Q.h_erupt / 1.0e9)
                        end
                        if event !== nothing
                            append!(erupted_tracers_Q, erupted)
                            push!(eruption_events_Q, event)
                            erupted_step_Q += event.erupted_enthalpy
                            magma_out_step_Q += event.magma_removed
                            melt_out_step_Q += max(0.0, event.melt_removed)
                            Params_Q.Told .= T_Q
                            compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
                            # the injection-zone boundary markers (dashed lines in ax2)
                            # ride on the host rock through the closure
                            collapse_markers!(
                                zone_markers, (event.z_lo + event.z_hi) / 2,
                                event.requested; method = erupt_mode
                            )
                            event_volume = A_sill * event.requested / 1.0e9   # m^3 -> km^3
                            erupted_volume_Q += event_volume
                            push!(eruption_event_time_Q_vec, time_kyrs)
                            push!(eruption_event_volume_Q_vec, event_volume)
                            push!(collapse_event_time_Q_vec, time_kyrs)
                            push!(collapse_event_thickness_Q_vec, event.requested)
                            surface_subsidence_Q += collapse_surface_subsidence(erupt_mode, event.requested)
                            println("Eruption (Q_magma) @ z=$((event.z_lo + event.z_hi) / 2), erupted thickness=$(round(event.requested, digits = 2)) m, volume=$(round(event_volume, digits = 4)) km^3, cumulative=$(round(erupted_volume_Q, digits = 4)) km^3")
                        end
                    end
                end
                if run_discrete
                    update_enthalpy_budget!(
                        enthalpy_budget,
                        column_enthalpy(T, z, MatParam, Params.Phases);
                        boundary = boundary_step, injected = injected_step,
                        source = source_step, erupted = erupted_step
                    )
                    push!(enthalpy_budget_vec, enthalpy_budget_snapshot(enthalpy_budget))
                    # ϕ is left over from the solve, which ran on the pre-injection T; the
                    # budgets and the plots below need the melt fraction of the state the step
                    # actually ends in
                    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
                    update_mass_budget!(
                        mass_budget, integrated_content(rocks, z),
                        melt_thickness(Params.ϕ, z, z[1], z[end]);
                        injected = magma_in_step, withdrawn = magma_out_step, erupted = melt_out_step
                    )
                    push!(mass_budget_vec, mass_budget_snapshot(mass_budget))
                end
                if run_Qmagma
                    update_enthalpy_budget!(
                        enthalpy_budget_Q,
                        column_enthalpy(T_Q, z, MatParam, Params_Q.Phases);
                        boundary = boundary_step_Q, injected = injected_step_Q,
                        source = source_step_Q, erupted = erupted_step_Q
                    )
                    push!(enthalpy_budget_Q_vec, enthalpy_budget_snapshot(enthalpy_budget_Q))
                    compute_meltfraction!(Params_Q.ϕ, MatParam, Params_Q.Phases, (T = Params_Q.Told .+ 273.15,))
                    update_mass_budget!(
                        mass_budget_Q, integrated_content(rocks_Q, z),
                        melt_thickness(Params_Q.ϕ, z, z[1], z[end]);
                        injected = magma_in_step_Q, withdrawn = magma_out_step_Q, erupted = melt_out_step_Q
                    )
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
                        push!(M_vec, erupt_state.M); push!(MH2O_vec, erupt_state.M_H2O); push!(Mres_vec, erupt_state.mass_residual)
                    end
                    if run_Qmagma
                        push!(dP_Q_vec, erupt_state_Q.P - erupt_state_Q.P_lith); push!(mdiss_Q_vec, erupt_state_Q.m_diss)
                        push!(Xg_Q_vec, erupt_state_Q.X_g); push!(phig_Q_vec, erupt_state_Q.ϕ_g)
                        push!(rhogas_Q_vec, erupt_state_Q.ρ_gas); push!(etar_Q_vec, erupt_state_Q.η_r); push!(phimush_Q_vec, erupt_state_Q.ϕ_mush)
                        push!(M_Q_vec, erupt_state_Q.M); push!(MH2O_Q_vec, erupt_state_Q.M_H2O); push!(Mres_Q_vec, erupt_state_Q.mass_residual)
                    end
                end

                # redraw only every plot_every steps (and always the final frame): the physics
                # already ran above, this block is pure visualization
                if mod(t, plot_every) == 0 || t == nt
                    time_val[] = time_kyrs

                    empty!(ax2)
                    if run_discrete
                        Tplot[] = T
                        ϕplot[] = Params.ϕ
                        rock_low = Point2f.(zero(rocks), z / 1.0e3)
                        rock_high = Point2f.(clamp.(rocks, 0.0, 1.0), z / 1.0e3)
                        band!(ax2, rock_low, rock_high, color = (:lightgrey, 1.0), label = "injected sills")
                        lines!(ax2, Params.ϕ, z / 1.0e3, color = :blue, label = "discrete sills ϕ")
                    end
                    if run_Qmagma
                        TQplot[] = T_Q
                        ϕQplot[] = Params_Q.ϕ
                        lines!(ax2, Params_Q.ϕ, z / 1.0e3, color = :purple, linestyle = :dash, label = "Q_magma ϕ")
                        hlines!(ax2, zone_markers ./ 1.0e3, color = :black, linestyle = :dash, linewidth = 1)
                    end
                    if !legend2_added
                        axislegend(ax2, position = :rb, framevisible = true, labelsize = 9, patchsize = (12, 12))
                        legend2_added = true
                    end

                    empty!(ax3)
                    # eruption times marked on every time-series panel, so a kink in the
                    # T/ϕ curves can be attributed to an eruption rather than to cooling
                    erupt_times = vcat(eruption_event_time_vec, eruption_event_time_Q_vec)
                    # vlines! cannot autolimit an empty vector, so only draw once an eruption
                    # has happened; the legend entry comes from a proxy that is always present
                    isempty(erupt_times) || vlines!(ax3, erupt_times, color = (:black, 0.25), linewidth = 1)
                    lines!(ax3, Float64[], Float64[], color = (:black, 0.25), label = "eruption")
                    if run_discrete
                        lines!(ax3, time_vec, Tmax_vec, color = :red, label = "T (sills)")
                        scatter!(ax3, time_vec[end], Tmax_vec[end], color = :red)
                    end
                    if run_Qmagma
                        lines!(ax3, time_vec, TQmax_vec, color = :orange, linestyle = :dash, label = "T (Q_magma)")
                    end
                    # ax4 shares the cell with ax3, so its ϕ series need their own legend entries
                    if run_discrete
                        lines!(ax3, Float64[], Float64[], color = :blue, label = "ϕ (sills)")
                    end
                    if run_Qmagma
                        lines!(ax3, Float64[], Float64[], color = :purple, linestyle = :dash, label = "ϕ (Q_magma)")
                    end
                    lines!(ax3, Float64[], Float64[], color = :gray30, label = "ϕ_mush")
                    if !legend3_added && (run_discrete || run_Qmagma)
                        axislegend(ax3, position = :lt, framevisible = true, labelsize = 9, patchsize = (12, 12))
                        legend3_added = true
                    end
                    if run_discrete
                        Tmax_all = vcat(Tmax_vec, run_Qmagma ? TQmax_vec : Float64[])
                        ylims!(ax3, minimum(Tmax_all) - 10, maximum(Tmax_all) + 10)
                    else
                        ylims!(ax3, minimum(TQmax_vec) - 10, Tsill + 10)
                    end

                    empty!(ax4)
                    if run_discrete
                        lines!(ax4, time_vec, ϕmax_vec, color = :blue)
                        scatter!(ax4, time_vec[end], ϕmax_vec[end], color = :blue)
                    end
                    if run_Qmagma
                        lines!(ax4, time_vec, ϕQmax_vec, color = :purple, linestyle = :dash)
                    end
                    # ϕ_mush shares this axis with the maximum melt fraction: both are 0–1
                    # melt fractions, and the pair shows how far the chamber mean lags the peak
                    if !isempty(phimush_vec)
                        lines!(ax4, time_vec, phimush_vec, color = :gray30)
                    end
                    if !isempty(phimush_Q_vec)
                        lines!(ax4, time_vec, phimush_Q_vec, color = :gray30, linestyle = :dash)
                    end
                    ylims!(ax4, 0, 1.01)

                    empty!(ax5)
                    if run_discrete
                        lines!(ax5, time_vec, erupted_volume_vec, color = :darkgreen, label = "cumulative (sills)")
                    end
                    if run_Qmagma
                        lines!(ax5, time_vec, erupted_volume_Q_vec, color = :darkgreen, linestyle = :dash, label = "cumulative (Q_magma)")
                    end
                    vol_max = max(
                        run_discrete ? maximum(erupted_volume_vec) : 0.0,
                        run_Qmagma ? maximum(erupted_volume_Q_vec) : 0.0, 1.0e-9
                    )
                    ylims!(ax5, 0, vol_max * 1.1)

                    empty!(ax5b)
                    if !isempty(eruption_event_time_vec)
                        stem!(ax5b, eruption_event_time_vec, eruption_event_volume_vec, color = :orange, label = "per event (sills)")
                    end
                    if !isempty(eruption_event_time_Q_vec)
                        stem!(ax5b, eruption_event_time_Q_vec, eruption_event_volume_Q_vec, color = :purple, label = "per event (Q_magma)")
                    end
                    event_max = max(
                        isempty(eruption_event_volume_vec) ? 0.0 : maximum(eruption_event_volume_vec),
                        isempty(eruption_event_volume_Q_vec) ? 0.0 : maximum(eruption_event_volume_Q_vec), 1.0e-9
                    )
                    ylims!(ax5b, 0, event_max * 1.1)
                    # merged legend for both eruption series (cumulative on ax5, per-event on ax5b)
                    if !legend5_added && (run_discrete || run_Qmagma) &&
                            (!isempty(erupted_volume_vec) || !isempty(eruption_event_time_vec) || !isempty(eruption_event_time_Q_vec))
                        axislegend(ax5, position = :lt, framevisible = true, labelsize = 9, patchsize = (12, 12))
                        legend5_added = true
                    end

                    # bottom panel: D&H chamber H₂O speciation, discrete sills on the left axis
                    # (teal) and Q_magma on the right (purple). Both axes get the same limits,
                    # so the two models stay directly comparable by eye.
                    empty!(ax6); empty!(ax6b)
                    isempty(erupt_times) || vlines!(ax6, erupt_times, color = (:black, 0.25), linewidth = 1)
                    lines!(ax6, Float64[], Float64[], color = (:black, 0.25), label = "eruption")
                    if run_discrete && !isempty(mdiss_vec)
                        lines!(ax6, time_vec, mdiss_vec, color = :teal, label = "dissolved H₂O")
                        lines!(ax6, time_vec, Xg_vec, color = :teal, linestyle = :dot, label = "exsolved H₂O (gas)")
                    end
                    if run_Qmagma && !isempty(mdiss_Q_vec)
                        lines!(ax6b, time_vec, mdiss_Q_vec, color = :purple)
                        lines!(ax6b, time_vec, Xg_Q_vec, color = :purple, linestyle = :dot)
                        # ax6b's plots are invisible to axislegend(ax6); proxy them across
                        lines!(ax6, Float64[], Float64[], color = :purple, label = "dissolved H₂O (Q)")
                        lines!(ax6, Float64[], Float64[], color = :purple, linestyle = :dot, label = "exsolved H₂O (Q)")
                    end
                    h2o_max = max(
                        run_discrete && !isempty(mdiss_vec) ? max(maximum(mdiss_vec), maximum(Xg_vec)) : 0.0,
                        run_Qmagma  && !isempty(mdiss_Q_vec) ? max(maximum(mdiss_Q_vec), maximum(Xg_Q_vec)) : 0.0,
                        1.0e-9
                    )
                    # Keep X_g = 0 visible instead of clipping it into the bottom spine.
                    # The data remain nonnegative; this is only a small plotting margin.
                    h2o_margin = 0.03 * h2o_max
                    ylims!(ax6, -h2o_margin, h2o_max * 1.1)
                    ylims!(ax6b, -h2o_margin, h2o_max * 1.1)
                    if !legend6_added && ((run_discrete && !isempty(mdiss_vec)) || (run_Qmagma && !isempty(mdiss_Q_vec)))
                        axislegend(ax6, position = :lt, framevisible = true, labelsize = 9, patchsize = (12, 12))
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
            last_run[:R_sill] = R_sill
            last_run[:time_vec] = time_vec
            last_run[:flux_m_per_yr] = flux_vec
            last_run[:flux_mode] = ȧ.mode
            last_run[:flux_base_m_per_yr] = ȧ.base * SecYear
            last_run[:flux_peak_m_per_yr] = ȧ.peak * SecYear
            last_run[:flux_start_kyr] = ȧ.t_start / SecYear / 1.0e3
            last_run[:flux_end_kyr] = ȧ.t_end / SecYear / 1.0e3
            last_run[:flux_table_time_kyr] = ȧ.times ./ SecYear ./ 1.0e3
            last_run[:flux_table_m_per_yr] = ȧ.rates .* SecYear
            last_run[:flux_table_depth_km] = ȧ.depths ./ 1.0e3
            # Fixed-aperture sills quantize the forcing: the flux delivers a continuous
            # thickness, but the discrete branch emplaces it only one whole sill at a time.
            # The undelivered remainder is the forcing the two branches do not share at the
            # end of a run, so it is exported rather than left implicit in A_inj.
            last_run[:delivered_thickness] = A_inj
            last_run[:pending_thickness] = mod(A_inj, Sillthick)
            # Which branches ran: the generic keys below alias one of them, so consumers
            # need these to tell a comparison run from a single-model run.
            last_run[:run_discrete] = run_discrete
            last_run[:run_Qmagma] = run_Qmagma
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
                last_run[:chamber_dP_vec] = dP_vec
                last_run[:chamber_mdiss_vec] = mdiss_vec
                last_run[:chamber_Xg_vec] = Xg_vec
                last_run[:chamber_phig_vec] = phig_vec
                last_run[:chamber_rhogas_vec] = rhogas_vec
                last_run[:chamber_etar_vec] = etar_vec
                last_run[:chamber_phimush_vec] = phimush_vec
                last_run[:chamber_M_vec] = M_vec
                last_run[:chamber_MH2O_vec] = MH2O_vec
                last_run[:chamber_mass_residual_vec] = Mres_vec
                last_run[:eruption_trigger_time_vec] = eruption_trigger_time_vec
                last_run[:eruption_trigger_volume_vec] = eruption_trigger_volume_vec
                last_run[:pending_withdrawal] = erupt_state.h_pending
            end
            if trigger_method == "D&H 3-phase" && run_Qmagma
                last_run[:chamber_dP_vec_Qmagma] = dP_Q_vec
                last_run[:chamber_mdiss_vec_Qmagma] = mdiss_Q_vec
                last_run[:chamber_Xg_vec_Qmagma] = Xg_Q_vec
                last_run[:chamber_phig_vec_Qmagma] = phig_Q_vec
                last_run[:chamber_rhogas_vec_Qmagma] = rhogas_Q_vec
                last_run[:chamber_etar_vec_Qmagma] = etar_Q_vec
                last_run[:chamber_phimush_vec_Qmagma] = phimush_Q_vec
                last_run[:chamber_M_vec_Qmagma] = M_Q_vec
                last_run[:chamber_MH2O_vec_Qmagma] = MH2O_Q_vec
                last_run[:chamber_mass_residual_vec_Qmagma] = Mres_Q_vec
                last_run[:eruption_trigger_time_vec_Qmagma] = eruption_trigger_time_Q_vec
                last_run[:eruption_trigger_volume_vec_Qmagma] = eruption_trigger_volume_Q_vec
                last_run[:pending_withdrawal_Qmagma] = erupt_state_Q.h_pending
            end

            # Preserve the existing REPL aliases: discrete is primary in comparison runs.
            QMagma.tracers_out = run_discrete ? tracers : tracers_Q
            QMagma.erupted_tracers_out = run_discrete ? erupted_tracers : erupted_tracers_Q
            QMagma.last_run_out = copy(last_run)
            println("Simulation data available in the REPL: `QMagma.tracers_out` (tracer T-t histories), `QMagma.erupted_tracers_out` (tracers removed by eruption), `QMagma.last_run_out` (profiles, time series, unified eruption_events, cumulative enthalpy_budget, and — for a D&H run — chamber H₂O diagnostics). SAVE DATA writes it all to <filename>.jld2.")
            println("SAVE JLD2 + VTK also writes 2D/3D temperature VTKs for each active model, tapered as a penny-shaped crack of the sill radius R over a lateral extent ±1.5R.")
            println("Compute zircon ages with: `QMagma.compute_zircon_ages(QMagma.tracers_out)`, or click COMPUTE ZIRCON AGES")
        catch err
            @error "Simulation loop failed" exception = (err, catch_backtrace())
        finally
            sim_running[] = false
        end
    end
    return nothing
end
