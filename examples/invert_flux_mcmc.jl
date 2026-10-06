# Bayesian (MCMC) inversion of the magma accretion history from zircon ages.
#
#   julia --project=examples -t 24 examples/invert_flux_mcmc.jl [options]
#
# Where `invert_flux.jl` returns one best-fit history, this samples the posterior of the
# flux nodes, so the range of histories the data allow is explicit.
#
# Options (all optional):
#   --eruptions            run the forward models with the D&H eruption trigger (a model
#                          choice you make, never a fitted parameter; default: off)
#   --population=NAME      `reservoir` (default), `erupted` or `both` (see invert_flux.jl)
#   --data=FILE            observed ages [ka before end of run], one per line. Without it a
#                          synthetic truth is generated and recovered (twin test).
#   --nwalkers=N           ensemble size, even and >= 2*7 (default 48)
#   --nsteps=N             generations to run (default 300)
#   --sigma=S              age error [ka] in the likelihood (default 2.5); overridden by a
#                          per-age error column in --data (second column, e.g. sigma_ka)
#   --preset=unzen         Unzen/Heisei set-up: 1055 °C recharge into 8-12 km, silicic melting
#                          law (see `unzen_overrides`)
#   --out=DIR              output directory (default examples/mcmc_out)
#   --hires=N              after sampling, re-run N random posterior draws with the
#                          full-resolution (GUI default) model and compare (default 24; 0 = skip)
#   --resume               continue the chain in DIR/chain.csv
#   --summarize            only summarise the chain in DIR/chain.csv (and run the --hires
#                          check; repeat --eruptions/--population/--sigma as used for the chain)
#
# Model. Prior: log10 accretion rate at each of the 7 time nodes uniform in
# LOG_RATE_BOUNDS. Likelihood: every observed age is a draw from the model's age density
# (the forward model's ages, smoothed with a Gaussian of width `sigma` for the analytical
# error plus model error) mixed with 1 % uniform outliers. Sampler: the affine-invariant
# ensemble sampler of Goodman & Weare (2010), in the parallel half-ensemble form, so each
# half of the walkers is evaluated across threads.
#
# Low-resolution sampling, high-resolution check. The sampler needs thousands of models, so
# it runs the coarse set-up of `invert_flux.jl` (Δz = 50 m, Δt = 250 yr) with a reduced
# zircon calculation (nx = 20, every 3rd tracer): ~5 s on one thread, and its age peak sits
# ~0.5 ka off the full model (absorbed by `sigma`). 48 walkers x 300 steps is 14 400 models.
# Afterwards `--hires` re-runs posterior draws at the GUI resolution (Δz = 20 m, Δt = 100 yr,
# 2 tracers per sill, nx = 50, all tracers) to check that the coarse model did not bias them.
# The synthetic data of the twin test come from that full-resolution model as well.

include(joinpath(@__DIR__, "invert_flux.jl"))
using DelimitedFiles

# GUI-default resolution, used for the synthetic data and for checking posterior draws
const HIRES_MODEL = (; Δz = 20.0, Δt_yr = 100.0, nt = 3000, tracers_per_sill = 2, nx_zircon = 50)

# Unzen (Heisei 1990-95) set-up, from the compiled literature: recharge 1055 ± 75 °C (Holtz et
# al. 2005), injection into the 8-12 km low-Vs zone beneath Fugendake (Miyano et al. 2021;
# pressure sources 7-13 km, Nakada et al. 1999; de Silva: storage centred at 8 km). The silicic
# melting law is the only one whose liquidus (1059 °C) lies above the recharge temperature.
unzen_overrides() = (;
    Tsill = 1055.0, Silltop = 8.0, Sillbot = 12.0,
    melting = QMagma.gui_composition("MeltingParam_Rhyolite").melting,
)

"""
    load_ages_sigma(path) -> (ages, sigmas)

Read `age_ka[, sigma_ka, ...]` rows (a header line is skipped). `sigmas` is `nothing` when the
file has a single column.
"""
function load_ages_sigma(path)
    ages, sig = Float64[], Float64[]
    for line in eachline(path)
        f = split(strip(line), [',', ' ', '\t'], keepempty = false)
        (isempty(f) || startswith(f[1], '#')) && continue
        a = tryparse(Float64, f[1])
        a === nothing && (isempty(ages) ? continue : error("non-numeric age in $path: $line"))
        push!(ages, a)
        length(f) >= 2 && (v = tryparse(Float64, f[2]); v !== nothing && push!(sig, v))
    end
    isempty(ages) && error("no ages found in $path")
    return ages, length(sig) == length(ages) ? sig : nothing
end

const AGE_RANGE_KA = 300.0         # support of the outlier component
const OUTLIER_FRACTION = 0.01
const KDE_EXTRA_KA = 1.0           # smoothing for the finite number of model tracers

"""
    silverman_bandwidth(ages)

Silverman's rule for the density estimate of the model ages, `0.9 min(std, IQR/1.34) n^(-1/5)`,
never below `KDE_EXTRA_KA`. It widens the smoothing when the model has only a handful of datable
zircons, whose ages would otherwise make the likelihood jump between neighbouring parameters.
"""
function silverman_bandwidth(ages)
    n = length(ages)
    n < 2 && return AGE_RANGE_KA / 4
    spread = min(std(ages), (quantile(ages, 0.75) - quantile(ages, 0.25)) / 1.34)
    spread = max(spread, std(ages) / 4)           # a lone cluster must not collapse the bandwidth
    return max(0.9 * spread * n^(-0.2), KDE_EXTRA_KA)
end

"""
    loglikelihood(ages, observed, sigma; smooth = :fixed)

`sigma` is one age error [ka] for all observed ages, or a vector with one error per age.

Sum over observed ages of log[(1-ε) p_model(a) + ε/range], with `p_model` the model ages
smoothed by a Gaussian of width `√(σ² + h²)`, with `h` fixed (`smooth = :fixed`, the default,
`KDE_EXTRA_KA`) or set by `silverman_bandwidth` (`smooth = :silverman`). `ages === nothing` (no datable zircon) leaves
only the outlier component, a large but finite penalty.
"""
function loglikelihood(ages, observed, sigma; smooth = :fixed)
    base = OUTLIER_FRACTION / AGE_RANGE_KA
    ages === nothing && return length(observed) * log(base)
    h = smooth === :silverman ? silverman_bandwidth(ages) : KDE_EXTRA_KA
    ll = 0.0
    for (i, a) in enumerate(observed)
        s = sqrt((sigma isa Number ? sigma : sigma[i])^2 + h^2)      # per-age error if a vector
        dens = 0.0
        for m in ages
            dens += exp(-0.5 * ((a - m) / s)^2)
        end
        ll += log((1 - OUTLIER_FRACTION) * dens / (length(ages) * s * sqrt(2π)) + base)
    end
    return ll
end

function logposterior(p, observed, model, sigma, population)
    all(x -> LOG_RATE_BOUNDS[1] <= x <= LOG_RATE_BOUNDS[2], p) || return -Inf
    ages = try
        model_ages(p, model; population)
    catch err
        err isa ErrorException || rethrow()      # failed thermal solve: reject the proposal
        return -Inf
    end
    return loglikelihood(ages, observed, sigma)
end

# ─── Sampler ───────────────────────────────────────────────────────────────────────────────
"""
One half-ensemble update of the stretch move (scale `a`): every walker in `active` moves
along the line to a random walker of `other`. Evaluations run across threads. Returns the
number of accepted moves.
"""
function stretch_update!(X, lp, active, other, logpost, rng; a = 2.0)
    d = size(X, 2)
    prop = zeros(length(active), d)
    zs = zeros(length(active))
    for (k, i) in enumerate(active)
        z = ((a - 1) * rand(rng) + 1)^2 / a
        j = rand(rng, other)
        prop[k, :] = X[j, :] .+ z .* (X[i, :] .- X[j, :])
        zs[k] = z
    end
    lp_prop = zeros(length(active))
    Threads.@threads for k in eachindex(active)
        lp_prop[k] = logpost(prop[k, :])
    end
    accepted = 0
    for (k, i) in enumerate(active)
        if log(rand(rng)) < (d - 1) * log(zs[k]) + lp_prop[k] - lp[i]
            X[i, :] = prop[k, :]
            lp[i] = lp_prop[k]
            accepted += 1
        end
    end
    return accepted
end

function summarize(out; burn = 0.4)
    data = readdlm(joinpath(out, "chain.csv"), ',', skipstart = 1)
    step, P = Int.(data[:, 1]), data[:, 4:end]
    nsteps = maximum(step) + 1
    keep = step .>= round(Int, burn * nsteps)
    truth_file = joinpath(out, "truth.csv")
    truth = isfile(truth_file) ? vec(readdlm(truth_file, ',')) : nothing
    Pk = P[keep, :]
    mid = (maximum(step[keep]) + minimum(step[keep])) ÷ 2
    first_half = Pk[step[keep] .<= mid, :]
    second_half = Pk[step[keep] .> mid, :]

    io = IOBuffer()
    println(io, "steps: $nsteps, burn-in discarded: $(round(Int, burn * 100)) %, samples kept: $(size(Pk, 1))")
    println(io, "log10 rate bounds: $(LOG_RATE_BOUNDS); prior is uniform in them")
    println(io, rpad("t [kyr]", 9), rpad("median [m/yr]", 15), rpad("5–95 % [m/yr]", 22), rpad("1st/2nd half median", 22), truth === nothing ? "" : "truth [m/yr]")
    for j in axes(Pk, 2)
        q = quantile(Pk[:, j], [0.05, 0.5, 0.95])
        h1 = isempty(first_half) ? NaN : median(first_half[:, j])
        h2 = isempty(second_half) ? NaN : median(second_half[:, j])
        println(
            io, rpad(Int(NODE_TIMES_KYR[j]), 9), rpad(round(10.0^q[2], sigdigits = 3), 15),
            rpad("$(round(10.0^q[1], sigdigits = 2))–$(round(10.0^q[3], sigdigits = 2))", 22),
            rpad("$(round(10.0^h1, sigdigits = 2)) / $(round(10.0^h2, sigdigits = 2))", 22),
            truth === nothing ? "" : round(10.0^truth[j], sigdigits = 3)
        )
    end
    widths = [quantile(Pk[:, j], 0.95) - quantile(Pk[:, j], 0.05) for j in axes(Pk, 2)]
    prior_width = 0.9 * (LOG_RATE_BOUNDS[2] - LOG_RATE_BOUNDS[1])
    println(io, "\n90 % interval width in dex (prior: $(round(prior_width, digits = 2))): ", join(round.(widths, digits = 2), "  "))
    println(io, "Nodes whose interval fills the prior width are not constrained by the data.")
    println(io, "If the two half medians disagree, the chain has not converged: run longer (--resume).")
    txt = String(take!(io))
    write(joinpath(out, "summary.txt"), txt)
    println(txt)
    return nothing
end

"""
    hires_check(out; n=24, burn=0.4, ...)

Re-run `n` random post-burn-in draws of `out/chain.csv` with the full-resolution model and
compare their log-likelihood with the coarse model's. Writes `hires_check.csv`
(`logL_coarse`, `logL_hires`, age quantiles of the full-resolution model, and the nodes).
Draws whose likelihood drops a lot at full resolution mark where the coarse model misleads.
"""
function hires_check(
        out; n = 24, burn = 0.4, eruptions = false, population = :reservoir, sigma = 2.5, seed = 5,
        overrides = (;)
    )
    data = readdlm(joinpath(out, "chain.csv"), ',', skipstart = 1)
    step, P = Int.(data[:, 1]), data[:, 4:end]
    ok = step .>= round(Int, burn * (maximum(step) + 1))
    idx = findall(ok)
    pick = idx[randperm(MersenneTwister(seed), length(idx))[1:min(n, length(idx))]]
    obs_file = readdlm(joinpath(out, "observed_ages_ka.csv"), ',')
    observed = vec(Float64.(obs_file[:, 1]))
    size(obs_file, 2) >= 2 && (sigma = vec(Float64.(obs_file[:, 2])))     # per-age errors
    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    coarse = merge(BASE_MODEL, extra, (; seed = 1, nx_zircon = 20, zircon_tracers = 3))
    fine = merge(BASE_MODEL, extra, HIRES_MODEL, (; seed = 1))

    ll_c, ll_h = zeros(length(pick)), zeros(length(pick))
    q = zeros(length(pick), 5)
    println("re-running $(length(pick)) posterior draws at full resolution ...")
    Threads.@threads for k in eachindex(pick)
        p = P[pick[k], :]
        ll_c[k] = logposterior(p, observed, coarse, sigma, population)
        ages = model_ages(p, fine; population)
        ll_h[k] = loglikelihood(ages, observed, sigma)
        q[k, :] = ages === nothing ? fill(NaN, 5) : quantile(ages, [0.05, 0.25, 0.5, 0.75, 0.95])
    end
    open(joinpath(out, "hires_check.csv"), "w") do io
        println(io, "logL_coarse,logL_hires,age_q05,age_q25,age_q50,age_q75,age_q95,", join(("log10_rate_$(Int(t))kyr" for t in NODE_TIMES_KYR), ","))
        for k in eachindex(pick)
            println(io, ll_c[k], ",", ll_h[k], ",", join(q[k, :], ","), ",", join(P[pick[k], :], ","))
        end
    end
    Δ = ll_h .- ll_c
    msg = "full-resolution check of $(length(pick)) posterior draws: logL(hires) − logL(coarse) " *
        "median $(round(median(Δ), digits = 1)), range $(round(minimum(Δ), digits = 1)) … $(round(maximum(Δ), digits = 1)); " *
        "logL(hires) median $(round(median(ll_h), digits = 1))"
    println(msg)
    open(joinpath(out, "summary.txt"), "a") do io
        println(io, "\n", msg)
    end
    return nothing
end

function run_mcmc(;
        eruptions = false, population = :reservoir, data = nothing, nwalkers = 48,
        nsteps = 300, sigma = 2.5, out = joinpath(@__DIR__, "mcmc_out"), seed = 3,
        resume = false, hires = 24, truth = [-1.3, -1.0, -0.6, -0.3, -0.8, -1.5, -2.0],
        overrides = (;)
    )
    population in (:reservoir, :erupted, :both) || error("population must be :reservoir, :erupted or :both")
    eruptions || population === :reservoir || error("population = $population needs eruptions")
    d = length(NODE_TIMES_KYR)
    iseven(nwalkers) && nwalkers >= 2d || error("nwalkers must be even and >= $(2d)")
    mkpath(out)

    extra = merge(eruptions ? eruption_setup() : (;), overrides)
    fine = merge(BASE_MODEL, extra, HIRES_MODEL, (; seed = 1))        # data + final check
    cheap = merge(BASE_MODEL, extra, (; seed = 1, nx_zircon = 20, zircon_tracers = 3))  # sampler

    if data === nothing
        println("twin experiment: truth log10 rates = $truth")
        observed = synthetic_observations(truth, fine; population)   # full-resolution model
        writedlm(joinpath(out, "truth.csv"), truth', ',')
    else
        observed, file_sigma = load_ages_sigma(data)
        file_sigma === nothing || (sigma = file_sigma)            # per-grain analytical errors
    end
    writedlm(joinpath(out, "observed_ages_ka.csv"), sigma isa Number ? observed : [observed sigma], ',')
    println("$(length(observed)) observed ages, median $(round(median(observed), digits = 1)) ka; $(Threads.nthreads()) thread(s)")

    logpost = p -> logposterior(p, observed, cheap, sigma, population)
    chain_file = joinpath(out, "chain.csv")
    rng = MersenneTwister(seed)

    start_step = 0
    if resume && isfile(chain_file)
        data_prev = readdlm(chain_file, ',', skipstart = 1)
        last = maximum(data_prev[:, 1])
        rows = data_prev[data_prev[:, 1] .== last, :]
        rows = rows[sortperm(rows[:, 2]), :]
        X, lp = rows[:, 4:end], rows[:, 3]
        start_step = Int(last) + 1
        println("resuming after step $last")
    else
        X = latin_hypercube(rng, nwalkers, d, LOG_RATE_BOUNDS...)
        lp = zeros(nwalkers)
        Threads.@threads for i in 1:nwalkers
            lp[i] = logpost(X[i, :])
        end
        open(chain_file, "w") do io
            println(io, "step,walker,logp,", join(("log10_rate_$(Int(t))kyr" for t in NODE_TIMES_KYR), ","))
        end
    end

    half = nwalkers ÷ 2
    A, B = 1:half, (half + 1):nwalkers
    for step in start_step:(start_step + nsteps - 1)
        t0 = time()
        acc = stretch_update!(X, lp, A, B, logpost, rng) + stretch_update!(X, lp, B, A, logpost, rng)
        open(chain_file, "a") do io
            for i in 1:nwalkers
                println(io, step, ",", i, ",", lp[i], ",", join(X[i, :], ","))
            end
        end
        println(
            "step $step: acceptance $(round(acc / nwalkers, digits = 2)), ",
            "best logL $(round(maximum(lp), digits = 1)), median logL $(round(median(lp), digits = 1)), ",
            "$(round(time() - t0, digits = 1)) s"
        )
    end
    summarize(out)
    hires > 0 && hires_check(out; n = hires, eruptions, population, sigma, overrides)
    return nothing
end

function parse_mcmc_args(args)
    opts = Dict{Symbol, Any}()
    for a in args
        a == "--eruptions" ? (opts[:eruptions] = true) :
            a == "--resume" ? (opts[:resume] = true) :
            a == "--summarize" ? (opts[:summarize] = true) :
            startswith(a, "--population=") ? (opts[:population] = Symbol(a[14:end])) :
            startswith(a, "--data=") ? (opts[:data] = a[8:end]) :
            startswith(a, "--nwalkers=") ? (opts[:nwalkers] = parse(Int, a[12:end])) :
            startswith(a, "--nsteps=") ? (opts[:nsteps] = parse(Int, a[10:end])) :
            startswith(a, "--preset=") ? (opts[:preset] = a[10:end]) :
            startswith(a, "--hires=") ? (opts[:hires] = parse(Int, a[9:end])) :
            startswith(a, "--sigma=") ? (opts[:sigma] = parse(Float64, a[9:end])) :
            startswith(a, "--out=") ? (opts[:out] = a[7:end]) :
            error("unknown option $a")
    end
    return (; opts...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_mcmc_args(ARGS)
    if get(opts, :summarize, false)
        out = get(opts, :out, joinpath(@__DIR__, "mcmc_out"))
        summarize(out)
        get(opts, :hires, 24) > 0 && hires_check(
            out; n = get(opts, :hires, 24), eruptions = get(opts, :eruptions, false),
            population = get(opts, :population, :reservoir), sigma = get(opts, :sigma, 2.5),
            overrides = get(opts, :preset, "") == "unzen" ? unzen_overrides() : (;)
        )
    else
        preset = get(opts, :preset, "")
        preset in ("", "unzen") || error("unknown preset $preset")
        overrides = preset == "unzen" ? unzen_overrides() : (;)
        run_mcmc(; Base.structdiff(opts, (; summarize = true, preset = ""))..., overrides)
    end
end
