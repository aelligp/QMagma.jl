# Export, stop and zircon-age buttons of the interactive app.

# Zircon ages are ≥0 by construction, so the density has a hard boundary at 0. A plain
# Gaussian KDE places mass below it, which is then silently dropped (`kde(x; boundary)`
# sets only the evaluation grid, it does not renormalize), biasing the youngest ages low.
#
# Jones' linear boundary kernel corrects this: within a bandwidth of 0 the Gaussian is
# reweighted by the truncated moments a0, a1, a2 of the mass that remains above 0, which
# cancels the leading bias for *any* boundary density. Reflecting the sample about 0
# would instead impose f'(0) = 0, a shape the crystallization history does not justify;
# merely rescaling each kernel by its surviving mass conserves the total but leaves the
# bias at 0 uncorrected. The correction can undershoot slightly, hence the clamp.
#
# Bandwidth is Silverman's rule of thumb, floored at one plot bin: a sample whose ages are
# all equal has zero spread by both measures, and Silverman's h would then be zero and
# divide through the kernel. Such a sample is physically ordinary — one sill, one narrow
# crystallization window — so it plots as a spike one bin wide rather than raising an error.
function _zircon_density!(ax, ages_ka, ub, color, alpha, label)
    n = length(ages_ka)
    σ = min(std(ages_ka), (quantile(ages_ka, 0.75) - quantile(ages_ka, 0.25)) / 1.34)
    h = max(0.9 * σ * n^(-1 / 5), ub / 512)
    φ(z) = exp(-z^2 / 2) / sqrt(2π)
    Φ(z) = (1 + erf(z / sqrt(2))) / 2
    x = range(0, ub, length = 512)
    y = map(x) do xi
        p = xi / h
        a0, a1, a2 = Φ(p), -φ(p), Φ(p) - p * φ(p)
        s = sum(ages_ka) do a
            t = (xi - a) / h
            return φ(t) * (a2 - a1 * t)
        end
        return max(s / (n * h * (a0 * a2 - a1^2)), 0.0)
    end
    band!(ax, x, zero(y), y; color = (color, alpha))
    return lines!(ax, x, y; color, linewidth = 2, label)
end

"""
    zircon_populations(last_run) -> Vector{Tuple{String, Vector{Tracer}, Vector{Tracer}}}

`(label, reservoir tracers, erupted cargo)` for each injection model of a finished run.
The two models carry separate tracer populations and so have separate zircon spectra;
the discrete branch comes first, matching the primary-branch convention of `last_run`.
Empty before the first run.
"""
function zircon_populations(last_run)
    pops = Tuple{String, Vector{Tracer}, Vector{Tracer}}[]
    get(last_run, :run_discrete, false) &&
        push!(pops, ("sill", last_run[:tracers], last_run[:erupted_tracers]))
    get(last_run, :run_Qmagma, false) &&
        push!(pops, ("Q_magma", last_run[:tracers_Qmagma], last_run[:erupted_tracers_Qmagma]))
    return pops
end

"""
    wire_buttons!(ui)

Attach the screenshot, data-export, stop and zircon-age handlers to the buttons of the
layout `ui`. Screenshot and data export write to the filename in the panel; the zircon
handler computes ages on a worker thread and opens its own plot window.
"""
function wire_buttons!(ui)
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
            matparam = last_matparam[]
            isnothing(matparam) && error("No material parameters available for 2D/3D melt-fraction export")
            R_sill = last_run[:R_sill]
            # the body ends at R_sill; the grid runs past it so the edge and some host rock
            # are in the exported field
            x2 = range(-1.5R_sill, 1.5R_sill; length = 41)
            x3 = range(-1.5R_sill, 1.5R_sill; length = 21)
            for (label, temperature_key) in (("discrete", :T), ("Qmagma", :T_Qmagma))
                haskey(last_run, temperature_key) || continue
                temperature = last_run[temperature_key]
                if temperature_key === :T
                    append!(
                        vtk_names, export_thermal_structure(
                            base_name * "_" * label, last_run[:z];
                            fields = (temperature, melt_fraction = last_run[:phi], rocks = last_run[:rocks]),
                            formats = (:vtk,)
                        )
                    )
                else
                    append!(
                        vtk_names, export_thermal_structure(
                            base_name * "_" * label, last_run[:z];
                            fields = (temperature, melt_fraction = last_run[:phi_Qmagma]), formats = (:vtk,)
                        )
                    )
                end
                T2 = lateral_thermal_structure(temperature, last_run[:T_background], x2; R = R_sill)
                T3 = lateral_thermal_structure(temperature, last_run[:T_background], x3; y = x3, R = R_sill)
                ϕ2 = melt_fraction_from_temperature(T2, matparam)
                ϕ3 = melt_fraction_from_temperature(T3, matparam)
                fields2 = (temperature = T2, melt_fraction = ϕ2)
                fields3 = (temperature = T3, melt_fraction = ϕ3)
                if temperature_key === :T
                    rocks = last_run[:rocks]
                    # injected magma thins away from the axis with the same profile as the
                    # anomaly, so its volume is the one lateral_effective_area converts
                    rocks2 = lateral_profile.(x2, R_sill) .* reshape(rocks, 1, :)
                    rocks3 = lateral_profile.(sqrt.(x3 .^ 2 .+ (x3') .^ 2), R_sill) .*
                        reshape(rocks, 1, 1, :)
                    fields2 = (; fields2..., rocks = rocks2)
                    fields3 = (; fields3..., rocks = rocks3)
                end
                append!(
                    vtk_names, export_thermal_structure(
                        base_name * "_" * label * "_2d", last_run[:z];
                        x = x2, fields = fields2, formats = (:vtk,)
                    )
                )
                append!(
                    vtk_names, export_thermal_structure(
                        base_name * "_" * label * "_3d", last_run[:z];
                        x = x3, y = x3, fields = fields3, formats = (:vtk,)
                    )
                )
            end
            println("Saved data to $(joinpath(pwd(), jld2_name)) and $(join(vtk_names, ", "))")
        end
    end

    on(but_stop.clicks) do n
        stop_requested[] = true
        println("Stopping simulation...")
    end

    on(but_zircon.clicks) do n
        # Snapshot: the worker runs while the GUI can start the next simulation.
        pops = [(label, copy(reservoir), copy(cargo)) for (label, reservoir, cargo) in zircon_populations(last_run)]

        if sim_running[]
            println("Simulation is still finishing up - wait until it stops completely before computing zircon ages")
        elseif zircon_running[]
            println("Zircon ages are already being computed")
        elseif all(isempty(reservoir) for (_, reservoir, _) in pops)
            println("No tracer data yet - run the simulation first")
        else
            zircon_running[] = true
            Threads.nthreads() == 1 && println("Zircon calculation uses the only Julia thread; restart with `julia -t auto` to keep the GUI responsive")
            println(
                "Computing zircon ages on $(Threads.nthreads()) thread(s) for " *
                    join(("$label: $(length(reservoir)) reservoir + $(length(cargo)) erupted-cargo tracers" for (label, reservoir, cargo) in pops), ", ")
            )
            worker = Threads.@spawn begin
                # One clock for every population, so the two models' spectra and their
                # erupted cargoes are all directly comparable.
                t_ref = maximum(
                    (
                        tr.time_vec[end] for (_, reservoir, _) in pops
                            for tr in reservoir if length(tr.time_vec) >= 2
                    ); init = 0.0
                )
                map(pops) do (label, reservoir, cargo)
                    res = compute_zircon_ages(reservoir; nx = 50, t_ref_Myr = t_ref)
                    car = isempty(cargo) ? Float64[] :
                        compute_zircon_ages(cargo; nx = 50, t_ref_Myr = t_ref).age_years
                    return (; label, res.age_years, res.zircon_radius_um, cargo_years = car, n_cargo = length(cargo))
                end
            end

            # Keep all GLMakie calls on the GUI task. `@async` alone is cooperative and
            # cannot make CPU-bound work responsive; it only waits for the worker here.
            @async try
                results = fetch(worker)
                # Model colors; within a model, the resident reservoir and the erupted
                # cargo get the cool/warm member of the pair.
                palette = ((:steelblue, :firebrick), (:seagreen, :darkorange))
                groups = Dict{String, Vector{NamedTuple}}()
                for (i, r) in pairs(results)
                    # Generic keys carry the first (discrete-primary) branch, matching the
                    # rest of `last_run`; the second branch is always Q_magma.
                    sfx = i == 1 ? "" : "_Qmagma"
                    isempty(r.age_years) && continue
                    last_run[Symbol(:zircon_age_years, sfx)] = r.age_years
                    last_run[Symbol(:zircon_radius_um, sfx)] = r.zircon_radius_um
                    isempty(r.cargo_years) ||
                        (last_run[Symbol(:zircon_age_years_erupted, sfx)] = r.cargo_years)

                    cool, warm = palette[mod1(i, length(palette))]
                    group = NamedTuple[]
                    push!(group, (; label = "$(r.label) reservoir", ka = r.age_years ./ 1.0e3, color = cool, alpha = 0.4))
                    isempty(r.cargo_years) ||
                        push!(group, (; label = "$(r.label) erupted", ka = r.cargo_years ./ 1.0e3, color = warm, alpha = 0.3))
                    r.n_cargo > 0 &&
                        println("$(r.label) erupted zircon cargo: $(length(r.cargo_years)) datable ages (of $(r.n_cargo) extracted tracers)")
                    groups[r.label] = group
                end

                if !isempty(groups)
                    ub = maximum(maximum(entry.ka) for group in values(groups) for entry in group) * 1.05
                    # "sill" (discrete) always occupies row 1, "Q_magma" row 2, so the two
                    # models never share an axis and stay visually comparable.
                    row_labels = filter(l -> haskey(groups, l), ("sill", "Q_magma"))

                    zircon_fig = Figure(size = (1100, 400 * length(row_labels)))
                    density_axes = Axis[]
                    for (row, label) in enumerate(row_labels)
                        title = label == "sill" ? "Discrete sills" : "Q_magma"
                        density_ax = Axis(
                            zircon_fig[row, 1], xlabel = "Zircon age [ka]", ylabel = "Density",
                            title = "$title zircon age distribution"
                        )
                        cdf_ax = Axis(
                            zircon_fig[row, 2], xlabel = "Zircon age [ka]", ylabel = "Cumulative probability [%]",
                            title = "$title zircon age spectrum (ranked order)"
                        )
                        for s in groups[label]
                            ns = length(s.ka)
                            plabel = "$(s.label) (n=$ns)"
                            ns > 1 && _zircon_density!(density_ax, s.ka, ub, s.color, s.alpha, plabel)
                            stairs!(cdf_ax, sort(s.ka), (1:ns) ./ ns .* 100; step = :post, color = s.color, label = plabel)
                        end
                        xlims!(density_ax, 0, ub)
                        ylims!(cdf_ax, 0, 100)
                        any(length(s.ka) > 1 for s in groups[label]) && axislegend(density_ax)
                        axislegend(cdf_ax, position = :rb)
                        push!(density_axes, density_ax)
                    end
                    # Same y-limits (density scale) across rows so peak heights are
                    # directly comparable, not just the shared x-axis (age, via `ub`).
                    length(density_axes) > 1 && linkyaxes!(density_axes...)

                    # display in its own window first: saving an undisplayed Figure
                    # directly can make GLMakie reuse/reconfigure the main GUI's existing
                    # screen instead of opening an independent one, replacing it on-screen.
                    display(GLMakie.Screen(), zircon_fig; title = "Zircon Ages")

                    zircon_name = filename[2].stored_string.val * "_zircon_ages.png"
                    save(zircon_name, zircon_fig)
                    println("Saved zircon age density + cumulative probability plot to $(joinpath(pwd(), zircon_name))")
                else
                    println("No tracers had enough recorded history to compute zircon ages; skipped zircon age plot")
                end
            catch err
                @error "Zircon age computation failed" exception = (err, catch_backtrace())
            finally
                zircon_running[] = false
            end
        end
    end
    return nothing
end
