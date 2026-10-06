# One-parameter fit of a CONSTANT accretion rate to real zircon ages.
#
#   julia --project=examples -t 16 examples/constant_flux_fit.jl --data=FILE [options]
#
# Options: --preset=unzen, --eruptions, --population=NAME, --out=DIR (default
# examples/constant_fit_out). `--data` is `age_ka[, sigma_ka]` per line.
#
# The posterior of the log10 rate is computed exactly on a grid (flat prior in
# LOG_RATE_BOUNDS) with the likelihood of `invert_flux_mcmc.jl`, using a Silverman-smoothed
# model age density because a constant flux leaves few datable zircons (see
# `constant_flux_test.jl`). Writes profile.csv, summary.txt and constant_fit.png.

include(joinpath(@__DIR__, "constant_flux_test.jl"))

function run_constant_fit(
        data; overrides = (;), eruptions = false, population = :reservoir,
        out = joinpath(@__DIR__, "constant_fit_out")
    )
    mkpath(out)
    observed, sigma = load_ages_sigma(data)
    sigma === nothing && (sigma = 2.5)
    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    coarse = merge(BASE_MODEL, extra, CONST_COARSE, (; seed = 1))
    println("$(length(observed)) observed ages, median $(round(median(observed), digits = 1)) ka")

    ages_grid = Vector{Any}(undef, length(GRID))
    Threads.@threads for i in eachindex(GRID)
        ages_grid[i] = try
            constant_ages(GRID[i], coarse; population)
        catch err
            err isa ErrorException || rethrow()
            nothing
        end
    end
    ll = [loglikelihood(a, observed, sigma; smooth = :silverman) for a in ages_grid]
    w = exp.(ll .- maximum(ll)); w ./= sum(w)
    cdf = cumsum(w)
    q(p) = GRID[findfirst(>=(p), cdf)]
    lo, med, hi, map_ = q(0.05), q(0.5), q(0.95), GRID[argmax(w)]
    n_ages = [a === nothing ? 0 : length(a) for a in ages_grid]
    writedlm(joinpath(out, "profile.csv"), [collect(GRID) 10.0 .^ GRID ll w n_ages], ',')
    best_ages = ages_grid[argmax(w)]
    open(joinpath(out, "summary.txt"), "w") do io
        println(io, "constant-rate fit, eruptions $eruptions, population $population")
        println(io, "MAP rate $(round(10.0^map_, sigdigits = 3)) m/yr; median $(round(10.0^med, sigdigits = 3)); 90 % interval $(round(10.0^lo, sigdigits = 3))–$(round(10.0^hi, sigdigits = 3)) m/yr")
        println(io, "max log-likelihood $(round(maximum(ll), digits = 1)); a flat-rate model with no datable zircon scores $(round(loglikelihood(nothing, observed, sigma), digits = 1))")
        println(io, "model at the MAP: $(best_ages === nothing ? "no" : length(best_ages)) datable zircons; ", best_ages === nothing ? "" : spread_line(best_ages))
        println(io, "observed:         ", spread_line(observed))
        println(io, "The interval is conditional on this model and likelihood and is probably optimistic.")
    end
    println(read(joinpath(out, "summary.txt"), String))

    # figure: model ages vs rate with the data, and the posterior
    ink, ink2, surface, gridc = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb", colorant"#e6e5e1"
    blue, orange = colorant"#2a78d6", colorant"#eb6834"
    kw = (;
        backgroundcolor = surface, xgridcolor = gridc, ygridcolor = gridc, topspinevisible = false,
        rightspinevisible = false, leftspinecolor = ink2, bottomspinecolor = ink2, xtickcolor = ink2,
        ytickcolor = ink2, xticklabelcolor = ink2, yticklabelcolor = ink2, xlabelcolor = ink,
        ylabelcolor = ink, titlecolor = ink, titlealign = :left, titlesize = 15, titlegap = 8,
    )
    fig = Figure(size = (1400, 480), backgroundcolor = surface, figure_padding = (24, 24, 16, 16))
    rates = collect(10.0 .^ GRID)
    ok = [i for i in eachindex(ages_grid) if ages_grid[i] !== nothing]
    qq = [quantile(ages_grid[i], [0.05, 0.5, 0.95]) for i in ok]
    ax = Axis(fig[1, 1]; title = "a   Model ages vs constant rate", xlabel = "Accretion rate [m/yr]", ylabel = "Zircon age [ka]", xscale = log10, xticks = [0.05, 0.1, 0.2, 0.5, 1.0], kw...)
    band!(ax, rates[ok], [v[1] for v in qq], [v[3] for v in qq]; color = (blue, 0.2))
    lines!(ax, rates[ok], [v[2] for v in qq]; color = blue, linewidth = 2.4)
    hspan!(ax, quantile(observed, 0.25), quantile(observed, 0.75); color = (orange, 0.18))
    hlines!(ax, median(observed); color = orange, linewidth = 2.2)
    text!(ax, 0.03, 0.88; text = "orange: observed median and\ninterquartile range", space = :relative, align = (:left, :top), color = ink2, fontsize = 11)
    text!(ax, 0.03, 0.97; text = "blue: model median, 5–95 %", space = :relative, align = (:left, :top), color = ink2, fontsize = 11)
    xlims!(ax, 0.03, 1.0)
    ax = Axis(fig[1, 2]; title = "b   Posterior of the rate", xlabel = "Accretion rate [m/yr]", ylabel = "Posterior density per dex", xscale = log10, xticks = [0.05, 0.1, 0.2, 0.5, 1.0], kw...)
    band!(ax, rates, zeros(length(rates)), w ./ step(GRID); color = (blue, 0.2))
    lines!(ax, rates, w ./ step(GRID); color = blue, linewidth = 2.4)
    text!(ax, 0.97, 0.97; text = "$(round(10.0^med, sigdigits = 2)) m/yr\n90 %: $(round(10.0^lo, sigdigits = 2))–$(round(10.0^hi, sigdigits = 2))", space = :relative, align = (:right, :top), color = ink2, fontsize = 12)
    ylims!(ax, 0, nothing); xlims!(ax, 0.03, 1.0)
    ax = Axis(fig[1, 3]; title = "c   Data vs best-fit model (cumulative)", xlabel = "Zircon age [ka]", ylabel = "Fraction of zircons", kw...)
    ecdf(a) = (sort(a), range(1 / length(a), 1, length = length(a)))
    xs, ys = ecdf(observed)
    l1 = stairs!(ax, xs, ys; color = orange, linewidth = 2.6, step = :post)
    ents, labs = Any[l1], ["observed ($(length(observed)))"]
    if best_ages !== nothing
        xs, ys = ecdf(best_ages)
        push!(ents, stairs!(ax, xs, ys; color = blue, linewidth = 2.6, step = :post)); push!(labs, "model at the MAP ($(length(best_ages)))")
    end
    axislegend(ax, ents, labs; position = :rb, framevisible = false, labelcolor = ink2, patchsize = (24, 10))
    save(joinpath(out, "constant_fit.png"), fig; px_per_unit = 2)
    println("wrote ", joinpath(out, "constant_fit.png"))
    return (; GRID, ll, w)
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = Dict{Symbol, Any}()
    for a in ARGS
        a == "--eruptions" ? (o[:eruptions] = true) :
            startswith(a, "--data=") ? (o[:data] = a[8:end]) :
            startswith(a, "--population=") ? (o[:population] = Symbol(a[14:end])) :
            startswith(a, "--preset=") ? (o[:preset] = a[10:end]) :
            startswith(a, "--out=") ? (o[:out] = a[7:end]) : error("unknown option $a")
    end
    haskey(o, :data) || error("--data=FILE is required")
    overrides = get(o, :preset, "") == "unzen" ? unzen_overrides() : (;)
    run_constant_fit(o[:data]; overrides, Base.structdiff((; delete!(copy(o), :data)...), (; preset = ""))...)
end
