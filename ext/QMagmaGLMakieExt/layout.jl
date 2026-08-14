# Figure, axes and control-panel construction for the interactive app.

"""
    build_layout(size) -> ui

Build the app window of pixel dimensions `size` and return a named tuple holding every
figure, axis and widget the callbacks operate on, together with the run state they share:
the elapsed-time label `time_val`, the `stop_requested`/`sim_running`/`zircon_running`
flags, the `last_run` result dictionary, and `last_matparam`, which holds the material
parameters of the most recent run for the export callback.
"""
function build_layout(size)
    fig = Figure(; size)

    time_val = Observable(0.0)
    stop_requested = Observable(false)
    sim_running = Observable(false)
    zircon_running = Observable(false)
    last_run = Dict{Symbol, Any}()
    last_matparam = Ref{Any}(nothing)

    Label(fig[0, 1:2], text = "1D Sill Injection", fontsize = 30)

    fig[1, 1] = depth_fig = GridLayout()
    ax1 = Axis(depth_fig[1:3, 1], xlabel = "Temperature [ᵒC]", ylabel = "Depth [km]", title = @lift("t = $(round($time_val, digits = 2)) kyrs"))
    ax2 = Axis(depth_fig[1:3, 2], xlabel = "Melt fraction ϕ")

    fig[1, 2] = timeseries_fig = GridLayout()
    ax3 = Axis(timeseries_fig[1, 1], xlabel = "Time [kyrs]", ylabel = "Maximum Temperature [ᵒC]", ytickcolor = :red, ylabelcolor = :red, yticklabelcolor = :red)
    ax4 = Axis(timeseries_fig[1, 1], ylabel = "Maximum melt fraction ϕ", ytickcolor = :blue, ylabelcolor = :blue, yticklabelcolor = :blue, yaxisposition = :right)
    ax5 = Axis(timeseries_fig[2, 1], xlabel = "Time [kyrs]", ylabel = "Cumulative erupted volume [km³]", ytickcolor = :darkgreen, ylabelcolor = :darkgreen, yticklabelcolor = :darkgreen)
    ax5b = Axis(timeseries_fig[2, 1], ylabel = "Erupted volume per event [km³]", ytickcolor = :orange, ylabelcolor = :orange, yticklabelcolor = :orange, yaxisposition = :right)

    # bottom row, full width under temp / melt-frac / eruptions: the D&H chamber's H₂O
    # speciation (dissolved vs exsolved), one model per y-axis so the two can be read
    # apart. Only populated when the D&H 3-phase trigger runs.
    fig[2, 1:2] = chamber_fig = GridLayout()
    ax6 = Axis(
        chamber_fig[1, 1], xlabel = "Time [kyrs]", ylabel = "H₂O [-] (discrete sills)",
        title = "D&H chamber: H₂O speciation",
        ytickcolor = :teal, ylabelcolor = :teal, yticklabelcolor = :teal
    )
    ax6b = Axis(
        chamber_fig[1, 1], ylabel = "H₂O [-] (Q_magma)",
        ytickcolor = :purple, ylabelcolor = :purple, yticklabelcolor = :purple, yaxisposition = :right
    )

    linkxaxes!(ax3, ax4)
    linkxaxes!(ax3, ax5)
    linkxaxes!(ax3, ax5b)
    linkxaxes!(ax3, ax6)
    linkxaxes!(ax3, ax6b)

    # keep the full-width H₂O panel from collapsing under the taller row-1 panels
    rowsize!(fig.layout, 2, Relative(0.28))

    fig[1:2, 3] = grid = GridLayout(tellwidth = false, valign = :top)
    colgap!(fig.layout, 2, 60)

    grid[1, 1] = but = Button(fig, label = "  RUN SIMULATION  ", buttoncolor = :lightgreen, height = 18, fontsize = 11)
    grid[1, 2] = but_stop = Button(fig, label = "  STOP  ", buttoncolor = :red, height = 18, fontsize = 11)

    Box(grid[2:3, 1:4], color = :lightgrey, cornerradius = 10)
    grid[2, 1:2] = Δz_box = add_textbox(fig, "Δz [m]:", 20)
    grid[2, 3:4] = nt_box = add_textbox(fig, "# steps nt:", 3000)
    grid[3, 1:2] = Δt_yrs_box = add_textbox(fig, "Δt [yrs]:", 100.0)

    Box(grid[4:5, 1:4], color = :lightblue, cornerradius = 10)
    grid[4, 1:2] = H_box = add_textbox(fig, "Crust [km]:", 40.0)
    grid[4, 3:4] = Ttop_box = add_textbox(fig, "Ttop [ᵒC]:", 0.0)
    grid[5, 1:2] = γ_box = add_textbox(fig, "Geotherm [ᵒC/km]:", 20.0)
    # η_r (shell relaxation viscosity) is computed each step from the mush temperature and
    # the far-field geotherm (crustal_relaxation_viscosity), so no GUI knob for it —
    # grid[5,3:4] left free.

    Box(grid[6:11, 1:4], color = :lightyellow, cornerradius = 10)
    grid[6, 1:2] = Tsill_box = add_textbox(fig, "Sill T [ᵒC]:", 1200.0)
    grid[6, 3:4] = Sill_thick_box = add_textbox(fig, "Sill thick [m]:", 100.0)
    grid[7, 1:4] = menu_flux = Menu(
        fig,
        options = ["Constant", "Linear ramp", "Pulse", "CSV table"],
        default = "Constant", height = 18, fontsize = 11
    )
    grid[8, 1:2] = flux_base_box = add_textbox(fig, "Base flux [m/yr]:", 0.1)
    grid[8, 3:4] = flux_peak_box = add_textbox(fig, "Peak/end flux [m/yr]:", 0.2)
    grid[9, 1:2] = flux_start_box = add_textbox(fig, "Flux start [kyr]:", 50.0)
    grid[9, 3:4] = flux_end_box = add_textbox(fig, "Flux end [kyr]:", 100.0)
    grid[10, 1:2] = flux_table_box = [
        Label(fig, "Flux CSV path:"),
        Textbox(
            fig, stored_string = "examples/flux_history.csv", height = 18, fontsize = 11,
            textpadding = (4, 4, 2, 2)
        ),
    ]
    grid[11, 1:2] = Sill_interval_top_box = add_textbox(fig, "Top inj. [km]:", 10.0)
    grid[11, 3:4] = Sill_interval_bot_box = add_textbox(fig, "Bottom inj. [km]:", 20.0)

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
        flux_base_enabled[] = mode in ("Constant", "Linear ramp")   # a pulse has no base flux
        flux_peak_enabled[] = variable
        flux_times_enabled[] = variable
        flux_table_enabled[] = table
        return nothing
    end
    on(update_flux_controls!, menu_flux.selection)
    update_flux_controls!(menu_flux.selection[])

    Box(grid[12:14, 1:4], color = (:red, 0.3), cornerradius = 10)
    grid[12, 1:2] = Ql_box = add_textbox(fig, "Latent heat [kJ/kg]:", 255.0)
    grid[13, 1:4] = menu_conduct = Menu(fig, options = ["T-dependent conductivity", "Constant conductivity 3 W/m/K"], default = "Constant conductivity 3 W/m/K", height = 18, fontsize = 11)
    grid[14, 1:4] = menu_melting = Menu(fig, options = ["MeltingParam_Assimilation", "MeltingParam_Basalt", "MeltingParam_Rhyolite"], default = "MeltingParam_Basalt", height = 18, fontsize = 11)

    Box(grid[15:16, 1:4], color = (:orange, 0.3), cornerradius = 10)
    grid[15, 1:4] = Label(fig, "Method:")
    grid[16, 1:4] = menu_method = Menu(fig, options = ["Discrete sills", "Q_magma", "Both (compare)"], default = "Both (compare)", height = 18, fontsize = 11)

    Box(grid[17:21, 1:4], color = (:purple, 0.2), cornerradius = 10)
    # eruption is two independent choices: the TRIGGER (when the column erupts) and the
    # COLLAPSE kinematics (how the column closes the vent afterwards).
    grid[17, 1:4] = Label(fig, "Eruption  trigger  |  collapse:")
    grid[18, 1:2] = menu_trigger = Menu(fig, options = ["None", "D&H 3-phase"], default = "None", height = 18, fontsize = 11)
    grid[18, 3:4] = menu_collapse = Menu(fig, options = ["Hybrid", "Caldera"], default = "Hybrid", height = 18, fontsize = 11)
    grid[19, 1:2] = Sill_radius_box = add_textbox(fig, "Sill radius [km]:", 5.0)
    grid[19, 3:4] = dPc_box = add_textbox(fig, "ΔP crit [MPa]:", 20.0)
    grid[20, 1:2] = mu_box = add_textbox(fig, "μ shear [GPa]:", 10.0)
    grid[20, 3:4] = hmelt_box = add_textbox(fig, "Min. chamber melt [m]:", 500.0)
    grid[21, 1:2] = mw_box = add_textbox(fig, "H₂O [wt%]:", 0.0)

    # Grey inactive eruption inputs and prevent focus/opening, so the selected trigger
    # is the only source of applicable settings.
    radius_enabled = Observable(false)
    pressure_enabled = Observable(false)
    shear_enabled = Observable(false)
    water_enabled = Observable(false)
    collapse_enabled = Observable(false)
    bind_textbox_enabled!(Sill_radius_box, radius_enabled)
    bind_textbox_enabled!(dPc_box, pressure_enabled)
    bind_textbox_enabled!(mu_box, shear_enabled)
    bind_textbox_enabled!(mw_box, water_enabled)
    bind_textbox_enabled!(hmelt_box, water_enabled)
    bind_menu_enabled!(menu_collapse, collapse_enabled)

    function update_eruption_controls!(trigger)
        controls = eruption_control_state(trigger)
        radius_enabled[] = controls.radius
        pressure_enabled[] = controls.pressure
        shear_enabled[] = controls.shear_modulus
        water_enabled[] = controls.water
        collapse_enabled[] = controls.collapse
        return nothing
    end
    on(update_eruption_controls!, menu_trigger.selection)
    update_eruption_controls!(menu_trigger.selection[])

    Box(grid[23:26, 1:4], color = (:green, 0.3), cornerradius = 10)
    grid[23, 1:2] = filename = [Label(fig, "filename:"), Textbox(fig, stored_string = "sim1", height = 18, fontsize = 11, textpadding = (4, 4, 2, 2))]
    grid[24, 1:2] = but_save = Button(fig, label = "  SAVE SCREENSHOT  ", buttoncolor = (:lightgreen, 0.5), height = 18, fontsize = 11)
    grid[24, 3:4] = but_save_data = Button(fig, label = "  SAVE JLD2 + VTK  ", buttoncolor = (:lightgreen, 0.5), height = 18, fontsize = 11)
    grid[25, 1:2] = record_toggle = add_togglebox(fig, "Record movie:", false)
    grid[26, 1:4] = but_zircon = Button(fig, label = "  COMPUTE ZIRCON AGES  ", buttoncolor = (:lightgreen, 0.5), height = 18, fontsize = 11)

    for r in 1:26
        rowsize!(grid, r, Fixed(18))
    end
    # rowgap! only reaches rows that already exist, so it must follow every widget:
    # 25 default gaps make the panel taller than the window.
    rowgap!(grid, 6)
    rowsize!(timeseries_fig, 2, Relative(1 / 2))

    return (;
        fig, depth_fig, timeseries_fig, chamber_fig, grid,
        ax1, ax2, ax3, ax4, ax5, ax5b, ax6, ax6b,
        but, but_stop, but_save, but_save_data, but_zircon,
        Δz_box, nt_box, Δt_yrs_box, H_box, Ttop_box, γ_box,
        Tsill_box, Sill_thick_box, menu_flux,
        flux_base_box, flux_peak_box, flux_start_box, flux_end_box, flux_table_box,
        Sill_interval_top_box, Sill_interval_bot_box,
        Ql_box, menu_conduct, menu_melting, menu_method,
        menu_trigger, menu_collapse, Sill_radius_box,
        dPc_box, mu_box, mw_box, hmelt_box,
        filename, record_toggle,
        time_val, stop_requested, sim_running, zircon_running, last_run, last_matparam,
    )
end
