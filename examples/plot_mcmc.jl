# Figure of an MCMC inversion made with `invert_flux_mcmc.jl`.
#
#   julia --project=examples -t auto examples/plot_mcmc.jl --dir=examples/mcmc_out [options]
#
# Options: --out=FILE (default DIR/mcmc_results.png), --eruptions, --ndraw=30 (posterior draws re-run for panel c), --burn=0.4
#
# Panels: (a) flux history, posterior bands against the truth if `truth.csv` exists;
# (b) log-likelihood of the walkers against step, to judge burn-in; (c) the age distributions
# of random posterior draws against the data - every draw fits, whatever its flux;
# (d) correlation between the node rates in the posterior.

include(joinpath(@__DIR__, "invert_flux_mcmc.jl"))
using CairoMakie, DelimitedFiles

const BLUE, ORANGE = colorant"#2a78d6", colorant"#eb6834"
const INK, INK2, GRIDC, SURFACE = colorant"#0b0b0b", colorant"#52514e", colorant"#e6e5e1", colorant"#fcfcfb"

function axis_kw()
    return (;
        backgroundcolor = SURFACE, xgridcolor = GRIDC, ygridcolor = GRIDC, topspinevisible = false,
        rightspinevisible = false, leftspinecolor = INK2, bottomspinecolor = INK2, xtickcolor = INK2,
        ytickcolor = INK2, xticklabelcolor = INK2, yticklabelcolor = INK2, xlabelcolor = INK,
        ylabelcolor = INK, titlecolor = INK, titlealign = :left, titlesize = 15, titlegap = 8,
    )
end

function plot_mcmc(
        dir; out = joinpath(dir, "mcmc_results.png"), overrides = (;), eruptions = false,
        ndraw = 30, burn = 0.4, seed = 21
    )
    raw = readdlm(joinpath(dir, "chain.csv"), ',', skipstart = 1)
    step, logp, P = Int.(raw[:, 1]), Float64.(raw[:, 3]), Float64.(raw[:, 4:end])
    nsteps = maximum(step) + 1
    keep = step .>= round(Int, burn * nsteps)
    Pk = P[keep, :]
    d = size(P, 2)
    truth = isfile(joinpath(dir, "truth.csv")) ? vec(readdlm(joinpath(dir, "truth.csv"), ',')) : nothing
    obs = readdlm(joinpath(dir, "observed_ages_ka.csv"), ',')
    observed = vec(Float64.(obs[:, 1]))
    nodes = NODE_TIMES_KYR
    nodes = nodes[1:d]

    # posterior-predictive: re-run random draws with the model the sampler used
    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    coarse = merge(BASE_MODEL, extra, (; seed = 1, nx_zircon = 20, zircon_tracers = 3))
    rng = MersenneTwister(seed)
    idx = rand(rng, 1:size(Pk, 1), ndraw)
    draw_ages = Vector{Any}(undef, ndraw)
    println("re-running $ndraw posterior draws ...")
    Threads.@threads for i in 1:ndraw
        draw_ages[i] = model_ages(Pk[idx[i], :], coarse)
    end
    truth_ages = truth === nothing ? nothing :
        model_ages(truth, merge(BASE_MODEL, extra, HIRES_MODEL, (; seed = 1)))

    fig = Figure(size = (1500, 1000), backgroundcolor = SURFACE, figure_padding = (24, 24, 20, 20))
    nw = count(==(0), step)
    Label(
        fig[0, 1:2], "MCMC inversion of the accretion history from zircon ages ($(nw) walkers × $nsteps steps)";
        fontsize = 20, font = :bold, color = INK, halign = :left
    )

    # (a) flux history
    ax = Axis(
        fig[1, 1]; title = "a   Posterior accretion history", xlabel = "Time [kyr]",
        ylabel = "Accretion rate [m/yr]", yscale = log10, axis_kw()...
    )
    q(p) = [quantile(Pk[:, j], p) for j in 1:d]
    for i in rand(rng, 1:size(Pk, 1), 40)
        lines!(ax, nodes, 10.0 .^ Pk[i, :]; color = RGBAf(0.32, 0.32, 0.31, 0.16), linewidth = 1)
    end
    band!(ax, nodes, 10.0 .^ q(0.05), 10.0 .^ q(0.95); color = (ORANGE, 0.16))
    band!(ax, nodes, 10.0 .^ q(0.25), 10.0 .^ q(0.75); color = (ORANGE, 0.28))
    l_med = lines!(ax, nodes, 10.0 .^ q(0.5); color = ORANGE, linewidth = 2.8)
    hlines!(ax, [10.0^-2.5, 1.0]; color = INK2, linewidth = 1, linestyle = :dot)
    entries, labels = Any[l_med, PolyElement(color = (ORANGE, 0.28)), PolyElement(color = (ORANGE, 0.16))],
        ["posterior median", "25–75 %", "5–95 %"]
    if truth !== nothing
        l_t = lines!(ax, nodes, 10.0 .^ truth; color = BLUE, linewidth = 2.8)
        scatter!(ax, nodes, 10.0 .^ truth; color = BLUE, markersize = 10, strokecolor = SURFACE, strokewidth = 2)
        pushfirst!(entries, l_t); pushfirst!(labels, "true flux")
    end
    ylims!(ax, 10.0^-2.7, 10.0^0.2)
    axislegend(ax, entries, labels; position = :lb, framevisible = false, labelcolor = INK2, patchsize = (24, 10))
    text!(ax, 0.98, 0.97; text = "dotted: prior bounds", space = :relative, align = (:right, :top), color = INK2, fontsize = 11)

    # (b) log-likelihood trace
    ax = Axis(
        fig[1, 2]; title = "b   Walker log-likelihood vs step", xlabel = "Step",
        ylabel = "Log-likelihood", axis_kw()...
    )
    steps = 0:(nsteps - 1)
    byste = [logp[step .== s] for s in steps]
    band!(ax, collect(steps), [quantile(v, 0.1) for v in byste], [quantile(v, 0.9) for v in byste]; color = (BLUE, 0.2))
    lines!(ax, collect(steps), [median(v) for v in byste]; color = BLUE, linewidth = 2.4)
    vlines!(ax, round(Int, burn * nsteps); color = INK2, linewidth = 1.2, linestyle = :dash)
    text!(ax, round(Int, burn * nsteps), 0.04; text = " burn-in discarded to the left", space = :data, color = INK2, fontsize = 11, align = (:left, :bottom), offset = (4, 0))
    lo = quantile(vcat(byste[end ÷ 2:end]...), 0.02)
    ylims!(ax, lo - 0.25 * abs(lo), maximum(maximum.(byste)) + 0.05 * abs(lo))
    text!(ax, 0.98, 0.04; text = "line: median walker, band: 10–90 %", space = :relative, align = (:right, :bottom), color = INK2, fontsize = 11)

    # (c) posterior-predictive ages
    ax = Axis(
        fig[2, 1]; title = "c   Posterior draws reproduce the age distribution", xlabel = "Zircon age [ka]",
        ylabel = "Fraction of zircons", axis_kw()...
    )
    ecdf(a) = (sort(a), range(1 / length(a), 1, length = length(a)))
    for a in draw_ages
        a === nothing && continue
        xs, ys = ecdf(a)
        stairs!(ax, xs, ys; color = (ORANGE, 0.35), linewidth = 1.2, step = :post)
    end
    l_d = lines!(ax, [NaN], [NaN]; color = (ORANGE, 0.6), linewidth = 2)
    xs, ys = ecdf(observed)
    l_o = stairs!(ax, xs, ys; color = BLUE, linewidth = 2.8, step = :post)
    entries, labels = Any[l_o, l_d], ["observed ($(length(observed)) ages)", "$ndraw posterior draws"]
    if truth_ages !== nothing
        xs, ys = ecdf(truth_ages)
        l_c = stairs!(ax, xs, ys; color = INK2, linewidth = 1.6, linestyle = :dash, step = :post)
        push!(entries, l_c); push!(labels, "true-flux model (full resolution)")
    end
    axislegend(ax, entries, labels; position = :rb, framevisible = false, labelcolor = INK2, patchsize = (24, 10))

    # (d) node correlations
    ax = Axis(
        fig[2, 2]; title = "d   Correlation between node rates (posterior)", xlabel = "Node time [kyr]",
        ylabel = "Node time [kyr]", xticks = (1:d, string.(Int.(nodes))), yticks = (1:d, string.(Int.(nodes))),
        aspect = DataAspect(), axis_kw()...
    )
    ax.xgridvisible = false; ax.ygridvisible = false
    C = cor(Pk)
    hm = heatmap!(ax, 1:d, 1:d, C; colormap = [BLUE, colorant"#f0efec", ORANGE], colorrange = (-1, 1))
    for i in 1:d, j in 1:d
        text!(ax, i, j; text = string(round(C[i, j], digits = 2)), align = (:center, :center), fontsize = 11, color = abs(C[i, j]) > 0.6 ? colorant"white" : INK)
    end
    Colorbar(fig[2, 3], hm; label = "correlation", labelcolor = INK2, ticklabelcolor = INK2, width = 14)

    colgap!(fig.layout, 28); rowgap!(fig.layout, 28)
    save(out, fig; px_per_unit = 2)
    println("wrote ", out)
    return out
end

function parse_plot_args(args)
    opts = Dict{Symbol, Any}(:dir => joinpath(@__DIR__, "mcmc_out"))
    for a in args
        a == "--eruptions" ? (opts[:eruptions] = true) :
            startswith(a, "--dir=") ? (opts[:dir] = a[7:end]) :
            startswith(a, "--out=") ? (opts[:out] = a[7:end]) :
            startswith(a, "--ndraw=") ? (opts[:ndraw] = parse(Int, a[9:end])) :
            startswith(a, "--burn=") ? (opts[:burn] = parse(Float64, a[8:end])) :
            error("unknown option $a")
    end
    return opts
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = parse_plot_args(ARGS)
    overrides = (;)
    kw = Base.structdiff((; o...), (; dir = ""))
    plot_mcmc(o[:dir]; overrides, kw...)
end
