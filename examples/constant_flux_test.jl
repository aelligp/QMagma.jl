# Twin test with a CONSTANT magma accretion rate.
#
#   julia --project=examples -t auto examples/constant_flux_test.jl [options]
#
# One parameter (the log10 rate), so the posterior is computed exactly on a grid instead of
# sampled. Synthetic zircon ages are generated from the full-resolution model at a known
# rate; the posterior comes from the coarse model, as in `invert_flux_mcmc.jl`. Repeated for
# several synthetic data sets to show how much the recovered rate scatters.
#
# The likelihood smooths the model ages with a Silverman bandwidth (there are only a handful
# of them), which keeps the posterior from collapsing onto single grid points.
#
# Options: --rate=0.12 (true rate [m/yr]), --nsets=3, --sigma=2.5, --eruptions,
#          --out=DIR (default examples/constant_out)

include(joinpath(@__DIR__, "invert_flux_mcmc.jl"))
using CairoMakie

# A constant flux leaves few datable zircons (tracers cool together only after the run's
# thermal peak), so seed more tracers per sill to resolve the age density, and keep every
# tracer in the coarse model.
const CONST_COARSE = (; tracers_per_sill = 4, nx_zircon = 30, zircon_tracers = 1)
const CONST_FINE = (; tracers_per_sill = 4)          # on top of HIRES_MODEL

const GRID = LOG_RATE_BOUNDS[1]:0.02:LOG_RATE_BOUNDS[2]

constant_ages(lograte, model; population = :reservoir) = begin
    res = run_Q_forward(FluxHistory(:constant; base = 10.0^lograte / SecYear); model...)
    parts = Vector{Float64}[]
    population in (:reservoir, :both) && push!(parts, res.ages.age_years ./ 1.0e3)
    population in (:erupted, :both) && res.ages_erupted !== nothing && push!(parts, res.ages_erupted.age_years ./ 1.0e3)
    ages = isempty(parts) ? Float64[] : reduce(vcat, parts)
    isempty(ages) ? nothing : ages
end

function run_constant_test(;
        rate = 0.12, nsets = 3, sigma = 2.5, eruptions = false, population = :reservoir,
        out = joinpath(@__DIR__, "constant_out")
    )
    mkpath(out)
    extra = eruptions ? eruption_setup() : (;)
    coarse = merge(BASE_MODEL, extra, CONST_COARSE, (; seed = 1))
    fine = merge(BASE_MODEL, extra, HIRES_MODEL, CONST_FINE, (; seed = 1))
    truth = log10(rate)

    println("true rate $rate m/yr; running the full-resolution truth model ...")
    truth_run = run_Q_forward(FluxHistory(:constant; base = rate / SecYear); fine...)
    true_ages = constant_ages(truth, fine; population)
    true_ages === nothing && error("a constant rate of $rate m/yr gives no datable zircon; try a larger rate")
    println("  $(length(true_ages)) datable zircons, median $(round(median(true_ages), digits = 1)) ka")

    # likelihood profile of the coarse model: shared by all data sets
    ages_grid = Vector{Any}(undef, length(GRID))
    Threads.@threads for i in eachindex(GRID)
        ages_grid[i] = try
            constant_ages(GRID[i], coarse; population)
        catch err
            err isa ErrorException || rethrow()
            nothing
        end
    end

    results = []
    open(joinpath(out, "summary.txt"), "w") do io
        println(io, "constant-rate twin test: true rate $rate m/yr, eruptions $eruptions, sigma $sigma ka")
        println(io, "full-resolution truth model: $(length(true_ages)) datable zircons, median $(round(median(true_ages), digits = 1)) ka")
        println(io, rpad("data set", 10), rpad("MAP [m/yr]", 12), rpad("median [m/yr]", 15), rpad("90 % interval [m/yr]", 24), "true in 90 %?")
        for k in 1:nsets
            rng = MersenneTwister(6 + k)
            picked = true_ages[rand(rng, 1:length(true_ages), 100)]
            observed = max.(picked .+ 2.0 .* randn(rng, 100), 0.0)
            ll = [loglikelihood(a, observed, sigma; smooth = :silverman) for a in ages_grid]
            w = exp.(ll .- maximum(ll)); w ./= sum(w)             # uniform prior in log10 rate
            cdf = cumsum(w)
            q(p) = GRID[findfirst(>=(p), cdf)]
            lo, med, hi = q(0.05), q(0.5), q(0.95)
            map_ = GRID[argmax(w)]
            inside = lo - step(GRID) / 2 <= truth <= hi + step(GRID) / 2
            println(
                io, rpad(k, 10), rpad(round(10.0^map_, sigdigits = 3), 12), rpad(round(10.0^med, sigdigits = 3), 15),
                rpad("$(round(10.0^lo, sigdigits = 3))–$(round(10.0^hi, sigdigits = 3))", 24), inside
            )
            writedlm(joinpath(out, "profile_set$(k).csv"), [GRID ll w], ',')
            push!(results, (; k, observed, ll, w, lo, med, hi, map_))
        end
    end
    println(read(joinpath(out, "summary.txt"), String))
    plot_constant(out, results, GRID, truth, true_ages, ages_grid, truth_run, rate)
    return results
end

function plot_constant(out, results, grid, truth, true_ages, ages_grid, truth_run, rate)
    ink, ink2, surface, gridc = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb", colorant"#e6e5e1"
    blue, orange, aqua = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"
    set_cols = [blue, orange, aqua]
    ax_kw = (;
        backgroundcolor = surface, xgridcolor = gridc, ygridcolor = gridc, topspinevisible = false,
        rightspinevisible = false, leftspinecolor = ink2, bottomspinecolor = ink2, xtickcolor = ink2,
        ytickcolor = ink2, xticklabelcolor = ink2, yticklabelcolor = ink2, xlabelcolor = ink,
        ylabelcolor = ink, titlecolor = ink, titlealign = :left, titlesize = 15, titlegap = 8,
    )
    dx = step(grid)
    rates = collect(10.0 .^ grid)
    fig = Figure(size = (1500, 900), backgroundcolor = surface, figure_padding = (24, 24, 20, 20))
    Label(
        fig[0, 1:3], "Constant magma flux: Q_magma forward model (top) and grid-search inversion (bottom)";
        fontsize = 20, font = :bold, color = ink, halign = :left
    )

    # ── forward model ──────────────────────────────────────────────────────────────────────
    ax = Axis(fig[1, 1]; title = "a   Forward model: constant accretion rate", xlabel = "Time [kyr]", ylabel = "Accretion rate [m/yr]", ax_kw...)
    tmax = truth_run.time / SecYear / 1.0e3
    lines!(ax, [0, tmax], [rate, rate]; color = blue, linewidth = 2.5)
    ylims!(ax, 0, 2 * rate)
    text!(ax, 0.03, 0.45; text = "$(rate) m/yr into 10–20 km depth\nfor $(round(Int, tmax)) kyr", space = :relative, color = ink2, fontsize = 12, align = (:left, :top))

    ax = Axis(fig[1, 2]; title = "b   Temperature at the end of the run", xlabel = "Temperature [°C]", ylabel = "Depth [km]", yreversed = true, ax_kw...)
    zk = -truth_run.z ./ 1.0e3
    hspan!(ax, 10, 20; color = (orange, 0.10))
    lines!(ax, 20.0 .* zk, zk; color = ink2, linewidth = 1.5, linestyle = :dash, label = "initial geotherm")
    lines!(ax, truth_run.T, zk; color = blue, linewidth = 2, label = "end of run")
    text!(ax, 30, 19.4; text = "injection window", color = ink2, fontsize = 11, align = (:left, :bottom))
    axislegend(ax; position = :lb, framevisible = false, labelcolor = ink2, patchsize = (24, 10))
    ylims!(ax, 40, 0)

    ax = Axis(fig[1, 3]; title = "c   Zircon ages of the reservoir (n = $(length(true_ages)))", xlabel = "Age [ka before end of run]", ylabel = "Density", ax_kw...)
    hist!(ax, true_ages; bins = 25, normalization = :pdf, color = (blue, 0.75), strokewidth = 1, strokecolor = surface)

    # ── grid search ────────────────────────────────────────────────────────────────────────
    ok = [i for i in eachindex(ages_grid) if ages_grid[i] !== nothing]
    q = [quantile(ages_grid[i], [0.05, 0.5, 0.95]) for i in ok]
    ax = Axis(fig[2, 1]; title = "d   Forward map: ages depend on the rate", xlabel = "Accretion rate [m/yr]", ylabel = "Zircon age [ka]", xscale = log10, xticks = [0.05, 0.1, 0.2, 0.5, 1.0], ax_kw...)
    band!(ax, rates[ok], [v[1] for v in q], [v[3] for v in q]; color = (blue, 0.2))
    lines!(ax, rates[ok], [v[2] for v in q]; color = blue, linewidth = 2.2)
    vlines!(ax, rate; color = ink, linewidth = 1.5, linestyle = :dash)
    xlims!(ax, 0.03, 1.0)
    text!(ax, 0.04, 0.95; text = "line: median, band: 5–95 %\nno datable zircon below $(round(rates[first(ok)], digits = 3)) m/yr", space = :relative, color = ink2, fontsize = 11, align = (:left, :top))

    ax = Axis(fig[2, 2]; title = "e   Posterior of the rate (3 synthetic data sets)", xlabel = "Accretion rate [m/yr]", ylabel = "Posterior density per dex", xscale = log10, xticks = [0.05, 0.1, 0.2, 0.5, 1.0], ax_kw...)
    for r in results
        band!(ax, rates, zeros(length(rates)), r.w ./ dx; color = (set_cols[r.k], 0.18))
        lines!(ax, rates, r.w ./ dx; color = set_cols[r.k], linewidth = 2.2, label = "data set $(r.k): $(round(10.0^r.med, sigdigits = 2)) ($(round(10.0^r.lo, sigdigits = 2))–$(round(10.0^r.hi, sigdigits = 2)))")
    end
    vlines!(ax, rate; color = ink, linewidth = 1.5, linestyle = :dash)
    text!(ax, rate, 1.0; text = "true rate", space = :data, color = ink2, fontsize = 11, align = (:left, :top), offset = (6, 0))
    ylims!(ax, 0, nothing)
    xlims!(ax, 0.03, 1.0)
    axislegend(ax, "median (90 % interval) [m/yr]"; position = :rt, framevisible = false, labelcolor = ink2, titlecolor = ink2, titlesize = 11, patchsize = (24, 10))

    ax = Axis(fig[2, 3]; title = "f   Data vs best-fit model (data set 1, cumulative)", xlabel = "Age [ka before end of run]", ylabel = "Fraction of zircons", ax_kw...)
    ecdf(a) = (sort(a), range(1 / length(a), 1, length = length(a)))
    r1 = results[1]
    xs, ys = ecdf(r1.observed)
    l1 = stairs!(ax, xs, ys; color = blue, linewidth = 2.5, step = :post)
    ibest = argmin(abs.(grid .- r1.map_))
    xs, ys = ecdf(ages_grid[ibest])
    l2 = stairs!(ax, xs, ys; color = orange, linewidth = 2.5, step = :post)
    xs, ys = ecdf(true_ages)
    l3 = stairs!(ax, xs, ys; color = ink2, linewidth = 1.5, linestyle = :dash, step = :post)
    axislegend(ax, [l1, l2, l3], ["observed (100 synthetic ages)", "best-fit model", "full-resolution truth"]; position = :rb, framevisible = false, labelcolor = ink2, patchsize = (24, 10))

    colgap!(fig.layout, 28)
    rowgap!(fig.layout, 28)
    save(joinpath(out, "constant_flux_test.png"), fig; px_per_unit = 2)
    println("wrote ", joinpath(out, "constant_flux_test.png"))
end

function parse_const_args(args)
    opts = Dict{Symbol, Any}()
    for a in args
        a == "--eruptions" ? (opts[:eruptions] = true) :
            startswith(a, "--rate=") ? (opts[:rate] = parse(Float64, a[8:end])) :
            startswith(a, "--nsets=") ? (opts[:nsets] = parse(Int, a[9:end])) :
            startswith(a, "--sigma=") ? (opts[:sigma] = parse(Float64, a[9:end])) :
            startswith(a, "--out=") ? (opts[:out] = a[7:end]) : error("unknown option $a")
    end
    return (; opts...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_constant_test(; parse_const_args(ARGS)...)
end
