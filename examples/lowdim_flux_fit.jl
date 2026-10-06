# Low-dimensional flux models fitted to real zircon ages, with exact grid posteriors.
#
#   julia --project=examples -t 24 examples/lowdim_flux_fit.jl --data=FILE [options]
#
# Options: --preset=unzen, --eruptions, --population=NAME, --out=DIR (default
# examples/lowdim_out), --mcmc=DIR (a 7-node chain whose best sample is re-scored for comparison).
#
# Three flux models (log10 rates in m/yr, times before the end of the run, which is the
# eruption), all with flat priors on the grid:
#   constant  rate                   (1 parameter, always on)
#   onset     rate, duration D       (2) a constant rate that starts D kyr before the end
#   episode   rate, t_start, t_end   (3) a constant rate between two times, off outside
# The likelihood is that of `invert_flux_mcmc.jl` with per-age errors and a Silverman-smoothed
# model age density (these models leave few datable zircons), the same for every model so
# that maximum log-likelihoods and BIC can be compared.

include(joinpath(@__DIR__, "constant_flux_test.jl"))

# One model with the constant-flux settings (4 tracers per sill, every tracer, nx = 30) takes
# 1-5 minutes at high flux, where thousands of tracers each need a zircon calculation; a grid of
# thousands of models needs a cheaper set-up.
const LOWDIM_COARSE = (; tracers_per_sill = 2, zircon_tracers = 2, nx_zircon = 20)

const RUN_KYR = BASE_MODEL.nt * BASE_MODEL.Δt_yr / 1.0e3         # 300 kyr

history(::Val{:constant}, p) = FluxHistory(:constant; base = 10.0^p[1] / SecYear)
history(::Val{:onset}, p) = FluxHistory(
    :pulse; peak = 10.0^p[1] / SecYear, t_start = (RUN_KYR - p[2]) * 1.0e3SecYear,
    t_end = (RUN_KYR + 1) * 1.0e3SecYear
)
history(::Val{:episode}, p) = FluxHistory(
    :pulse; peak = 10.0^p[1] / SecYear, t_start = (RUN_KYR - p[3]) * 1.0e3SecYear,
    t_end = (RUN_KYR - p[2]) * 1.0e3SecYear
)                                                              # p = (rate, kyr-before-end of end, of start)

function ages_for(model, p, coarse; population = :reservoir)
    res = run_Q_forward(history(Val(model), p); coarse...)
    parts = Vector{Float64}[]
    population in (:reservoir, :both) && push!(parts, res.ages.age_years ./ 1.0e3)
    population in (:erupted, :both) && res.ages_erupted !== nothing && push!(parts, res.ages_erupted.age_years ./ 1.0e3)
    a = isempty(parts) ? Float64[] : reduce(vcat, parts)
    return isempty(a) ? nothing : a
end

# grids of (log10 rate, ...) for each model. For `episode`, p[2] and p[3] are the kyr before
# the end at which the episode stops and starts (p[2] < p[3]); 0 means "until the eruption".
const RATES = -1.5:0.1:0.0
const DURATIONS = 20.0:20.0:300.0
const GRIDS = Dict(
    :constant => [(r,) for r in -1.5:0.04:0.0],
    :onset => [(r, D) for r in RATES for D in DURATIONS],
    :episode => [(r, e, s) for r in -1.5:0.25:0.0 for s in 20.0:20.0:300.0 for e in 0.0:20.0:280.0 if e < s],
)

function fit_model(model, observed, sigma, coarse, population)
    grid = GRIDS[model]
    ll = zeros(length(grid))
    Threads.@threads for i in eachindex(grid)
        a = try
            ages_for(model, grid[i], coarse; population)
        catch err
            err isa ErrorException || rethrow()
            nothing
        end
        ll[i] = loglikelihood(a, observed, sigma; smooth = :silverman)
    end
    w = exp.(ll .- maximum(ll)); w ./= sum(w)
    return (; model, grid, ll, w)
end

function marginal(fit, k)
    vals = sort(unique(p[k] for p in fit.grid))
    m = [sum(fit.w[i] for (i, p) in enumerate(fit.grid) if p[k] == v) for v in vals]
    return vals, m
end

function interval(vals, m)
    c = cumsum(m)
    return vals[findfirst(>=(0.05), c)], vals[findfirst(>=(0.5), c)], vals[findfirst(>=(0.95), c)]
end

function run_lowdim(
        data; overrides = (;), eruptions = false, population = :reservoir,
        out = joinpath(@__DIR__, "lowdim_out"), mcmc = nothing
    )
    mkpath(out)
    observed, sigma = load_ages_sigma(data)
    sigma === nothing && (sigma = 2.5)
    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    coarse = merge(BASE_MODEL, extra, LOWDIM_COARSE, (; seed = 1))
    fits = Dict{Symbol, Any}()
    for model in (:constant, :onset, :episode)
        t0 = time()
        fits[model] = fit_model(model, observed, sigma, coarse, population)
        println("$model: $(length(GRIDS[model])) models in $(round(time() - t0, digits = 0)) s, max logL $(round(maximum(fits[model].ll), digits = 1))")
    end

    # reference: best sample of a 7-node chain, scored with the same pipeline
    ref = nothing
    if mcmc !== nothing && isfile(joinpath(mcmc, "chain.csv"))
        raw = readdlm(joinpath(mcmc, "chain.csv"), ',', skipstart = 1)
        best = raw[argmax(Float64.(raw[:, 3])), 4:end]
        a = model_ages(Float64.(best), coarse; population)
        ref = (; ll = loglikelihood(a, observed, sigma; smooth = :silverman), nodes = best, ages = a)
    end

    nobs = length(observed)
    open(joinpath(out, "summary.txt"), "w") do io
        println(io, "low-dimensional flux fits, eruptions $eruptions, population $population, $nobs observed ages")
        println(io, rpad("model", 10), rpad("k", 3), rpad("max logL", 10), rpad("BIC", 9), "MAP")
        for (model, k) in ((:constant, 1), (:onset, 2), (:episode, 3))
            f = fits[model]; i = argmax(f.ll); p = f.grid[i]
            desc = model == :constant ? "rate $(round(10.0^p[1], sigdigits = 3)) m/yr" :
                model == :onset ? "rate $(round(10.0^p[1], sigdigits = 3)) m/yr from $(Int(p[2])) kyr before the end" :
                "rate $(round(10.0^p[1], sigdigits = 3)) m/yr from $(Int(p[3])) to $(Int(p[2])) kyr before the end"
            println(io, rpad(model, 10), rpad(k, 3), rpad(round(f.ll[i], digits = 1), 10), rpad(round(k * log(nobs) - 2 * f.ll[i], digits = 1), 9), desc)
        end
        ref === nothing || println(io, rpad("7-node", 10), rpad(7, 3), rpad(round(ref.ll, digits = 1), 10), rpad(round(7 * log(nobs) - 2 * ref.ll, digits = 1), 9), "best MCMC sample, same likelihood")
        println(io, "\nmarginal posteriors (5 / 50 / 95 %):")
        for (model, names) in ((:constant, ["rate [m/yr]"]), (:onset, ["rate [m/yr]", "duration [kyr]"]), (:episode, ["rate [m/yr]", "end [kyr before end]", "start [kyr before end]"]))
            for (k, nm) in enumerate(names)
                vals, m = marginal(fits[model], k)
                lo, med, hi = interval(vals, m)
                f = k == 1 ? (x -> round(10.0^x, sigdigits = 2)) : (x -> Int(x))
                println(io, rpad(model, 10), rpad(nm, 26), "$(f(lo)) / $(f(med)) / $(f(hi))")
            end
        end
        println(io, "\nBIC: lower is better; differences above ~6 are conventionally strong evidence. These are conditional on the model and the noisy likelihood.")
    end
    println(read(joinpath(out, "summary.txt"), String))

    plot_lowdim(out, fits, observed, sigma, coarse, population, ref)
    return fits
end

function plot_lowdim(out, fits, observed, sigma, coarse, population, ref)
    ink, ink2, surface, gridc = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb", colorant"#e6e5e1"
    blue, orange, aqua = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"
    kw = (;
        backgroundcolor = surface, xgridcolor = gridc, ygridcolor = gridc, topspinevisible = false,
        rightspinevisible = false, leftspinecolor = ink2, bottomspinecolor = ink2, xtickcolor = ink2,
        ytickcolor = ink2, xticklabelcolor = ink2, yticklabelcolor = ink2, xlabelcolor = ink,
        ylabelcolor = ink, titlecolor = ink, titlealign = :left, titlesize = 15, titlegap = 8,
    )
    seq = [colorant"#f6f4ef", colorant"#9ec3ee", blue, colorant"#173f73"]
    fig = Figure(size = (1500, 1000), backgroundcolor = surface, figure_padding = (24, 24, 20, 20))
    Label(fig[0, 1:2], "Low-dimensional flux models fitted to the Heisei zircon ages"; fontsize = 20, font = :bold, color = ink, halign = :left)

    # (a) onset model posterior
    f = fits[:onset]
    rv = sort(unique(p[1] for p in f.grid)); dv = sort(unique(p[2] for p in f.grid))
    Z = [sum(f.w[i] for (i, p) in enumerate(f.grid) if p[1] == r && p[2] == D) for D in dv, r in rv]
    ax = Axis(fig[1, 1]; title = "a   Constant flux that starts D before the eruption: posterior", xlabel = "Accretion rate [m/yr]", ylabel = "Duration of injection D [kyr]", xscale = log10, xticks = [0.05, 0.1, 0.2, 0.5, 1.0], kw...)
    hm = heatmap!(ax, 10.0 .^ rv, dv, permutedims(Z); colormap = seq)
    i = argmax(f.ll); scatter!(ax, [10.0^f.grid[i][1]], [f.grid[i][2]]; color = orange, markersize = 14, strokecolor = surface, strokewidth = 2)
    text!(ax, 10.0^f.grid[i][1], f.grid[i][2]; text = "  best fit", color = ink, fontsize = 12, align = (:left, :center), offset = (8, 0))
    Colorbar(fig[1, 2], hm; label = "posterior probability per cell", labelcolor = ink2, ticklabelcolor = ink2, width = 14)

    # (b) episode model: start vs end
    f = fits[:episode]
    ev = sort(unique(p[2] for p in f.grid)); sv = sort(unique(p[3] for p in f.grid))
    Z2 = [sum(f.w[i] for (i, p) in enumerate(f.grid) if p[2] == e && p[3] == s; init = 0.0) for s in sv, e in ev]
    ax = Axis(fig[1, 3]; title = "b   Episode of constant flux: posterior of start and end", xlabel = "Episode end [kyr before eruption]", ylabel = "Episode start [kyr before eruption]", kw...)
    hm2 = heatmap!(ax, ev, sv, permutedims(Z2); colormap = seq)
    lines!(ax, [0, 300], [0, 300]; color = ink2, linewidth = 1, linestyle = :dash)
    i = argmax(f.ll); scatter!(ax, [f.grid[i][2]], [f.grid[i][3]]; color = orange, markersize = 14, strokecolor = surface, strokewidth = 2)
    Colorbar(fig[1, 4], hm2; label = "posterior probability per cell", labelcolor = ink2, ticklabelcolor = ink2, width = 14)

    # (c) MAP flux histories vs time before the eruption
    ax = Axis(fig[2, 1]; title = "c   Best-fit accretion histories", xlabel = "Time before the eruption [kyr]", ylabel = "Accretion rate [m/yr]", yscale = log10, xreversed = true, kw...)
    tb = range(0, RUN_KYR, length = 601)
    cols = Dict(:constant => ink2, :onset => orange, :episode => aqua)
    names = Dict(:constant => "constant", :onset => "onset + constant", :episode => "episode")
    maps = Dict{Symbol, Any}()
    for model in (:constant, :onset, :episode)
        p = fits[model].grid[argmax(fits[model].ll)]; maps[model] = p
        h = history(Val(model), p)
        r = [max(h((RUN_KYR - t) * 1.0e3SecYear) * SecYear, 1.0e-4) for t in tb]
        lines!(ax, tb, r; color = cols[model], linewidth = model == :constant ? 2.2 : 3, linestyle = model == :constant ? :dash : :solid, label = names[model])
    end
    ref === nothing || lines!(ax, [0, 50, 100, 150, 200, 250, 300], 10.0 .^ Float64.(collect(ref.nodes)); color = (blue, 0.55), linewidth = 2, label = "best 7-node sample")
    ylims!(ax, 1.0e-2, 1.5)
    axislegend(ax; position = :lb, framevisible = false, labelcolor = ink2, patchsize = (24, 10))

    # (d) age distributions of the best fits
    ax = Axis(fig[2, 3]; title = "d   Observed vs best-fit age distributions (cumulative)", xlabel = "Zircon age [ka]", ylabel = "Fraction of zircons", kw...)
    ecdf(a) = (sort(a), range(1 / length(a), 1, length = length(a)))
    ents, labs = Any[], String[]
    for model in (:constant, :onset, :episode)
        a = ages_for(model, maps[model], coarse; population)
        a === nothing && continue
        xs, ys = ecdf(a)
        push!(ents, stairs!(ax, xs, ys; color = cols[model], linewidth = 2.2, step = :post, linestyle = model == :constant ? :dash : :solid)); push!(labs, "$(names[model]) ($(length(a)) ages)")
    end
    xs, ys = ecdf(observed)
    pushfirst!(ents, stairs!(ax, xs, ys; color = blue, linewidth = 3, step = :post)); pushfirst!(labs, "observed ($(length(observed)))")
    axislegend(ax, ents, labs; position = :rb, framevisible = false, labelcolor = ink2, patchsize = (24, 10))

    colgap!(fig.layout, 20); rowgap!(fig.layout, 28)
    save(joinpath(out, "lowdim_fits.png"), fig; px_per_unit = 2)
    println("wrote ", joinpath(out, "lowdim_fits.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = Dict{Symbol, Any}()
    for a in ARGS
        a == "--eruptions" ? (o[:eruptions] = true) :
            startswith(a, "--data=") ? (o[:data] = a[8:end]) :
            startswith(a, "--population=") ? (o[:population] = Symbol(a[14:end])) :
            startswith(a, "--preset=") ? (o[:preset] = a[10:end]) :
            startswith(a, "--mcmc=") ? (o[:mcmc] = a[8:end]) :
            startswith(a, "--out=") ? (o[:out] = a[7:end]) : error("unknown option $a")
    end
    haskey(o, :data) || error("--data=FILE is required")
    overrides = get(o, :preset, "") == "unzen" ? unzen_overrides() : (;)
    run_lowdim(o[:data]; overrides, Base.structdiff((; delete!(copy(o), :data)...), (; preset = ""))...)
end
