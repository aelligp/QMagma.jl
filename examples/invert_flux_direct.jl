# Direct (optimisation) inversion: the single best-fit flux history, no sampling.
#
#   julia --project=examples -t 24 examples/invert_flux_direct.jl [options]
#
# Maximises the same likelihood as `invert_flux_mcmc.jl` (per-age errors, flat prior in the
# bounds) over the 7 flux nodes: a Latin-hypercube global search, then Nelder–Mead from the
# best few starting points, run in parallel. Reports the best-fit history and re-evaluates it
# with the full-resolution model.
#
# Options: --preset=unzen, --data=FILE, --eruptions, --population=NAME, --sigma=S, --nlhs=N
#          (default 96), --nstarts=K (Nelder–Mead starts, default 4), --out=DIR
#          (default examples/direct_out). Without --data a synthetic twin test is run.
#
# A best fit is a point estimate. The MCMC posterior shows how well it is determined;
# with zircon ages alone the 7-node history is poorly constrained, so expect many
# different histories to fit about as well.

include(joinpath(@__DIR__, "invert_flux_mcmc.jl"))

function run_direct(;
        eruptions = false, population = :reservoir, data = nothing, nlhs = 96, nstarts = 4,
        sigma = 2.5, out = joinpath(@__DIR__, "direct_out"), seed = 4, overrides = (;),
        truth = [-1.3, -1.0, -0.6, -0.3, -0.8, -1.5, -2.0], polish_iterations = 200
    )
    mkpath(out)
    d = length(NODE_TIMES_KYR)
    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    fine = merge(BASE_MODEL, extra, HIRES_MODEL, (; seed = 1))
    cheap = merge(BASE_MODEL, extra, (; seed = 1, nx_zircon = 20, zircon_tracers = 3))

    if data === nothing
        observed = synthetic_observations(truth, fine; population)
    else
        observed, file_sigma = load_ages_sigma(data)
        file_sigma === nothing || (sigma = file_sigma)
        truth = nothing
    end
    println("$(length(observed)) observed ages, median $(round(median(observed), digits = 1)) ka; $(Threads.nthreads()) thread(s)")
    nll = p -> -logposterior(p, observed, cheap, sigma, population)    # +Inf outside the bounds

    rng = MersenneTwister(seed)
    P = latin_hypercube(rng, nlhs, d, LOG_RATE_BOUNDS...)
    J = zeros(nlhs)
    t0 = time()
    Threads.@threads for i in 1:nlhs
        J[i] = nll(P[i, :])
    end
    println("global search: $nlhs models in $(round(time() - t0, digits = 1)) s, best -logL $(round(minimum(J), digits = 1))")

    starts = sortperm(J)[1:min(nstarts, nlhs)]
    results = Vector{Any}(undef, length(starts))
    Threads.@threads for k in eachindex(starts)
        r = Optim.optimize(nll, P[starts[k], :], NelderMead(), Optim.Options(iterations = polish_iterations))
        results[k] = (Optim.minimizer(r), Optim.minimum(r))
    end
    for (k, r) in enumerate(results)
        println("  Nelder–Mead start $k: -logL $(round(J[starts[k]], digits = 1)) → $(round(r[2], digits = 1))")
    end
    best_p, best_J = results[argmin([r[2] for r in results])]
    best_p = clamp.(best_p, LOG_RATE_BOUNDS...)

    ll_hires = loglikelihood(model_ages(best_p, fine; population), observed, sigma)
    ages_best = model_ages(best_p, cheap; population)

    open(joinpath(out, "best_flux_history.csv"), "w") do io
        println(io, "time_kyr,flux_m_per_yr")
        foreach(((t, lr),) -> println(io, t, ",", 10.0^lr), zip(NODE_TIMES_KYR, best_p))
    end
    open(joinpath(out, "starts.csv"), "w") do io
        println(io, "start,neg_logL,", join(("log10_rate_$(Int(t))kyr" for t in NODE_TIMES_KYR), ","))
        for (k, r) in enumerate(results)
            println(io, k, ",", r[2], ",", join(r[1], ","))
        end
    end
    open(joinpath(out, "summary.txt"), "w") do io
        println(io, "direct inversion: eruptions $eruptions, population $population")
        println(io, "best logL (coarse model) $(-best_J);  full-resolution re-evaluation $ll_hires")
        println(io, "best-model ages: ", spread_line(ages_best))
        println(io, "observed ages:   ", spread_line(observed))
        println(io, "spread of the $(length(results)) polished starts' best-fit nodes [m/yr] (a measure of non-uniqueness):")
        println(io, rpad("t [kyr]", 9), rpad("best", 10), rpad("min–max over starts", 22), truth === nothing ? "" : "truth")
        for j in 1:d
            v = [r[1][j] for r in results]
            println(
                io, rpad(Int(NODE_TIMES_KYR[j]), 9), rpad(round(10.0^best_p[j], sigdigits = 3), 10),
                rpad("$(round(10.0^minimum(v), sigdigits = 2))–$(round(10.0^maximum(v), sigdigits = 2))", 22),
                truth === nothing ? "" : round(10.0^truth[j], sigdigits = 3)
            )
        end
    end
    println(read(joinpath(out, "summary.txt"), String))
    return (; best = best_p, J = best_J, results)
end

function parse_direct_args(args)
    opts = Dict{Symbol, Any}()
    for a in args
        a == "--eruptions" ? (opts[:eruptions] = true) :
            startswith(a, "--population=") ? (opts[:population] = Symbol(a[14:end])) :
            startswith(a, "--data=") ? (opts[:data] = a[8:end]) :
            startswith(a, "--nlhs=") ? (opts[:nlhs] = parse(Int, a[8:end])) :
            startswith(a, "--nstarts=") ? (opts[:nstarts] = parse(Int, a[11:end])) :
            startswith(a, "--sigma=") ? (opts[:sigma] = parse(Float64, a[9:end])) :
            startswith(a, "--preset=") ? (opts[:preset] = a[10:end]) :
            startswith(a, "--out=") ? (opts[:out] = a[7:end]) : error("unknown option $a")
    end
    return opts
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = parse_direct_args(ARGS)
    overrides = get(o, :preset, "") == "unzen" ? unzen_overrides() : (;)
    run_direct(; Base.structdiff((; o...), (; preset = ""))..., overrides)
end
