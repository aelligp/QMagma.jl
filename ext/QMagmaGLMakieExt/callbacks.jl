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
        dPc_box, mu_box, mw_box, hmelt_box,
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
            sigma = last_run[:gaussian_sigma]
            x2 = range(-3sigma, 3sigma; length = 41)
            x3 = range(-3sigma, 3sigma; length = 21)
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
                T2 = gaussian_thermal_structure(temperature, last_run[:T_background], x2; sigma)
                T3 = gaussian_thermal_structure(temperature, last_run[:T_background], x3; y = x3, sigma)
                ϕ2 = melt_fraction_from_temperature(T2, matparam)
                ϕ3 = melt_fraction_from_temperature(T3, matparam)
                fields2 = (temperature = T2, melt_fraction = ϕ2)
                fields3 = (temperature = T3, melt_fraction = ϕ3)
                if temperature_key === :T
                    rocks = last_run[:rocks]
                    rocks2 = (abs.(x2) .<= sigma) .* reshape(rocks, 1, :)
                    rocks3 = ((x3 .^ 2 .+ (x3') .^ 2) .<= sigma^2) .* reshape(rocks, 1, 1, :)
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
        if sim_running[]
            println("Simulation is still finishing up - wait until it stops completely before computing zircon ages")
        elseif zircon_running[]
            println("Zircon ages are already being computed")
        elseif isempty(QMagma.tracers_out)
            println("No tracer data yet - run the simulation first")
        else
            zircon_running[] = true
            reservoir = copy(QMagma.tracers_out)
            cargo = copy(QMagma.erupted_tracers_out)
            Threads.nthreads() == 1 && println("Zircon calculation uses the only Julia thread; restart with `julia -t auto` to keep the GUI responsive")
            println("Computing zircon ages for $(length(reservoir)) reservoir tracers + $(length(cargo)) erupted-cargo tracers on $(Threads.nthreads()) thread(s)...")
            worker = Threads.@spawn begin
                t_ref = maximum((tr.time_vec[end] for tr in reservoir if length(tr.time_vec) >= 2); init = 0.0)
                zircon_result = compute_zircon_ages(reservoir; nx = 50, t_ref_Myr = t_ref)
                cargo_result = isempty(cargo) ? nothing : compute_zircon_ages(cargo; nx = 50, t_ref_Myr = t_ref)
                return zircon_result, cargo_result
            end

            # Keep all GLMakie calls on the GUI task. `@async` alone is cooperative and
            # cannot make CPU-bound work responsive; it only waits for the worker here.
            @async try
                zircon_result, cargo_result = fetch(worker)
                if !isempty(zircon_result.age_years)
                    last_run[:zircon_age_years] = zircon_result.age_years
                    last_run[:zircon_radius_um] = zircon_result.zircon_radius_um
                    cargo_result !== nothing && (last_run[:zircon_age_years_erupted] = cargo_result.age_years)

                    age_ka = zircon_result.age_years ./ 1.0e3
                    n = length(age_ka)
                    cargo_ka = (cargo_result === nothing || isempty(cargo_result.age_years)) ? Float64[] : cargo_result.age_years ./ 1.0e3
                    ne = length(cargo_ka)

                    zircon_fig = Figure(size = (1100, 400))
                    zircon_ax = Axis(
                        zircon_fig[1, 1], xlabel = "Zircon age [ka]", ylabel = "Density",
                        title = "Zircon age distribution (reservoir n=$n, erupted n=$ne)"
                    )
                    ub = maximum(vcat(age_ka, cargo_ka)) * 1.05
                    n > 1 && _zircon_density!(zircon_ax, age_ka, ub, :steelblue, 0.4, "reservoir (n=$n)")
                    if ne > 1
                        _zircon_density!(zircon_ax, cargo_ka, ub, :firebrick, 0.3, "erupted (n=$ne)")
                    end
                    xlims!(zircon_ax, 0, ub)
                    axislegend(zircon_ax)

                    age_sorted = sort(age_ka)
                    cum_prob = (1:n) ./ n .* 100
                    cdf_ax = Axis(
                        zircon_fig[1, 2], xlabel = "Zircon age [ka]", ylabel = "Cumulative probability [%]",
                        title = "Zircon age spectrum (ranked order)"
                    )
                    stairs!(cdf_ax, age_sorted, cum_prob; step = :post, color = :steelblue, label = "reservoir (n=$n)")
                    if ne > 1
                        cargo_sorted = sort(cargo_ka)
                        stairs!(cdf_ax, cargo_sorted, (1:ne) ./ ne .* 100; step = :post, color = :firebrick, label = "erupted (n=$ne)")
                    end
                    ylims!(cdf_ax, 0, 100)
                    ne > 1 && axislegend(cdf_ax, position = :rb)

                    # display in its own window first: saving an undisplayed Figure
                    # directly can make GLMakie reuse/reconfigure the main GUI's existing
                    # screen instead of opening an independent one, replacing it on-screen.
                    display(GLMakie.Screen(), zircon_fig; title = "Zircon Ages")

                    zircon_name = filename[2].stored_string.val * "_zircon_ages.png"
                    save(zircon_name, zircon_fig)
                    println("Saved zircon age density + cumulative probability plot to $(joinpath(pwd(), zircon_name))")
                    ne > 0 && println("Erupted zircon cargo: $ne datable ages (of $(length(cargo)) extracted tracers)")
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
