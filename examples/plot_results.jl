# Figure of the Q_magma forward model and the flux-inversion twin test.
#
#   julia --project=examples -t auto examples/plot_results.jl [--results=DIR] [--out=FILE]
#
# Top row: one forward run (ramp flux, no eruptions): flux history, final temperature profile
# and zircon age distribution. Bottom row: twin test from `invert_flux.jl` - known flux versus
# the recovered one, the age distributions the two produce, and which flux nodes the fit
# actually pins down.
#
# `--results=DIR` is the output directory of a twin-test run of `invert_flux.jl` without
# eruptions (default: examples/inversion_out; it is run first if it does not exist).

include(joinpath(@__DIR__, "invert_flux.jl"))
using CairoMakie, DelimitedFiles

# ─── colours: categorical slots 1-2 of the reference palette; ink and surface tokens ──────
const C_BLUE, C_ORANGE = colorant"#2a78d6", colorant"#eb6834"
const INK, INK2, GRID, SURFACE = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1", colorant"#fcfcfb"
const C_ENS = RGBAf(0.32, 0.32, 0.31, 0.28)

function parse_plot_args(args)
    results = joinpath(@__DIR__, "inversion_out")
    out = joinpath(@__DIR__, "zircon_flux_results.png")
    for a in args
        startswith(a, "--results=") ? (results = a[11:end]) :
            startswith(a, "--out=") ? (out = a[7:end]) : error("unknown option $a")
    end
    return results, out
end

function styled_axis(pos; kw...)
    return Axis(
        pos; backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
        xgridwidth = 1, ygridwidth = 1, topspinevisible = false, rightspinevisible = false,
        leftspinecolor = INK2, bottomspinecolor = INK2, xtickcolor = INK2, ytickcolor = INK2,
        xticklabelcolor = INK2, yticklabelcolor = INK2, xlabelcolor = INK, ylabelcolor = INK,
        titlecolor = INK, titlealign = :left, titlesize = 15, titlegap = 8, kw...
    )
end

function main(args)
    results, out = parse_plot_args(args)
    isfile(joinpath(results, "ensemble.csv")) ||
        run_inversion(; eruptions = false, nlhs = 48, out = results)

    # ── forward run: the example ramp, at the GUI's default resolution ─────────────────────
    ȧ = FluxHistory(
        :ramp; base = 0.05 / SecYear, peak = 0.15 / SecYear,
        t_start = 50.0e3SecYear, t_end = 150.0e3SecYear
    )
    fwd = run_Q_forward(ȧ; nt = 3000, Δt_yr = 100.0)
    t_kyr = range(0, 300, length = 601)
    flux = [ȧ(t * 1.0e3SecYear) * SecYear for t in t_kyr]
    T0 = [-20.0 * zz / 1.0e3 for zz in fwd.z]                 # initial geotherm, Ttop = 0
    ages_fwd = fwd.ages.age_years ./ 1.0e3

    # ── twin test: truth, recovered flux, and the models in between ────────────────────────
    model = merge(BASE_MODEL, (; seed = 1))
    truth = [-1.3, -1.0, -0.6, -0.3, -0.8, -1.5, -2.0]
    ens = readdlm(joinpath(results, "ensemble.csv"), ',', skipstart = 1)
    misf, logr = ens[:, 1], ens[:, 2:end]                        # sorted by misfit
    best = vec(logr[1, :])
    observed = synthetic_observations(truth, model; population = :reservoir)
    ages_truth = model_ages(truth, model)
    ages_best = model_ages(best, model)
    tight = findall(misf .<= 5.0)                                # models that fit to ~noise

    set_theme!(fontsize = 12, fonts = (; regular = "DejaVu Sans", bold = "DejaVu Sans Bold"))
    fig = Figure(size = (1500, 900), backgroundcolor = SURFACE, figure_padding = (24, 24, 20, 20))

    Label(
        fig[0, 1:3], "Q_magma forward model (top) and flux inversion from zircon ages (bottom)";
        fontsize = 20, font = :bold, color = INK, halign = :left
    )

    # (a) flux history of the forward run
    axa = styled_axis(
        fig[1, 1]; title = "a   Forward run: magma accretion rate", xlabel = "Time [kyr]",
        ylabel = "Accretion rate [m/yr]"
    )
    lines!(axa, t_kyr, flux; color = C_BLUE, linewidth = 2)
    ylims!(axa, 0, 0.18)

    # (b) final temperature profile
    axb = styled_axis(
        fig[1, 2]; title = "b   Temperature after 300 kyr", xlabel = "Temperature [°C]",
        ylabel = "Depth [km]", yreversed = true
    )
    hspan!(axb, 10, 20; color = (C_ORANGE, 0.10))
    lines!(axb, T0, -fwd.z ./ 1.0e3; color = INK2, linewidth = 1.5, linestyle = :dash, label = "initial geotherm")
    lines!(axb, fwd.T, -fwd.z ./ 1.0e3; color = C_BLUE, linewidth = 2, label = "after 300 kyr")
    text!(axb, 20, 10.6; text = "injection window", color = INK2, fontsize = 11, align = (:left, :top))
    axislegend(axb; position = :lb, framevisible = false, labelcolor = INK2, patchsize = (24, 10))
    ylims!(axb, 40, 0)

    # (c) zircon age distribution of the forward run
    axc = styled_axis(
        fig[1, 3]; title = "c   Zircon ages of the reservoir (n = $(length(ages_fwd)))",
        xlabel = "Age [ka before end of run]", ylabel = "Density"
    )
    hist!(
        axc, ages_fwd; bins = 40, normalization = :pdf, color = (C_BLUE, 0.75),
        strokewidth = 1, strokecolor = SURFACE
    )

    # (d) recovered flux
    axd = styled_axis(
        fig[2, 1]; title = "d   Twin test: recovered accretion rate", xlabel = "Time [kyr]",
        ylabel = "Accretion rate [m/yr]", yscale = log10
    )
    for i in tight[1:min(end, 40)]
        lines!(axd, NODE_TIMES_KYR, 10.0 .^ vec(logr[i, :]); color = C_ENS, linewidth = 1)
    end
    ln_e = lines!(axd, [NaN], [NaN]; color = C_ENS, linewidth = 1)
    ln_t = lines!(axd, NODE_TIMES_KYR, 10.0 .^ truth; color = C_BLUE, linewidth = 2.5)
    scatter!(axd, NODE_TIMES_KYR, 10.0 .^ truth; color = C_BLUE, markersize = 9, strokecolor = SURFACE, strokewidth = 2)
    ln_b = lines!(axd, NODE_TIMES_KYR, 10.0 .^ best; color = C_ORANGE, linewidth = 2.5)
    scatter!(axd, NODE_TIMES_KYR, 10.0 .^ best; color = C_ORANGE, markersize = 9, strokecolor = SURFACE, strokewidth = 2)
    axislegend(
        axd, [ln_t, ln_b, ln_e], ["true flux", "best fit", "other models within 5 ka"];
        position = :lb, framevisible = false, labelcolor = INK2, patchsize = (24, 10)
    )
    ylims!(axd, 10.0^-2.6, 10.0^0.1)

    # (e) age distributions: cumulative
    axe = styled_axis(
        fig[2, 2]; title = "e   Age distributions (cumulative)", xlabel = "Age [ka before end of run]",
        ylabel = "Fraction of zircons"
    )
    ecdf(a) = (sort(a), range(1 / length(a), 1, length = length(a)))
    xs, ys = ecdf(observed)
    ln_o = stairs!(axe, xs, ys; color = C_BLUE, linewidth = 2.5, step = :post)
    xs, ys = ecdf(ages_best)
    ln_m = stairs!(axe, xs, ys; color = C_ORANGE, linewidth = 2.5, step = :post)
    xs, ys = ecdf(ages_truth)
    ln_c = stairs!(axe, xs, ys; color = INK2, linewidth = 1.5, linestyle = :dash, step = :post)
    axislegend(
        axe, [ln_o, ln_m, ln_c], ["observed (100 synthetic ages)", "best-fit model", "true-flux model, all tracers"];
        position = :rb, framevisible = false, labelcolor = INK2, patchsize = (24, 10)
    )

    # (f) which nodes are constrained
    axf = styled_axis(
        fig[2, 3]; title = "f   Which flux nodes are recovered", xlabel = "Time [kyr]",
        ylabel = "log₁₀(model rate / true rate)"
    )
    hlines!(axf, 0; color = INK2, linewidth = 1)
    lo = [minimum(logr[tight, j] .- truth[j]) for j in 1:length(truth)]
    hi = [maximum(logr[tight, j] .- truth[j]) for j in 1:length(truth)]
    rangebars!(axf, NODE_TIMES_KYR, lo, hi; color = (INK2, 0.5), linewidth = 2, whiskerwidth = 8)
    scatter!(
        axf, NODE_TIMES_KYR, best .- truth; color = C_ORANGE, markersize = 11,
        strokecolor = SURFACE, strokewidth = 2
    )
    ylims!(axf, -1.9, 1.0)
    text!(
        axf, 0.97, 0.97; text = "dots: best fit\nbars: range over models within 5 ka",
        space = :relative, align = (:right, :top), color = INK2, fontsize = 11
    )

    colgap!(fig.layout, 28)
    rowgap!(fig.layout, 28)
    save(out, fig; px_per_unit = 2)
    println("wrote ", out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
