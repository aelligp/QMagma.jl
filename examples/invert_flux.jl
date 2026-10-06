# Inversion of the magma accretion history from a zircon age distribution.
#
#   julia --project=examples -t auto examples/invert_flux.jl [options]
#
# Options (all optional):
#   --eruptions              run the forward models with the D&H eruption trigger
#                            (default: no eruptions). This is a model choice you make, it is
#                            never a fitted parameter.
#   --population=NAME        which zircons are compared with the data: `reservoir` (default),
#                            `erupted` (erupted cargo, needs --eruptions) or `both`
#   --data=FILE              observed zircon ages [ka before the end of the run], one per
#                            line (a single non-numeric header line is skipped). Without
#                            it a synthetic "truth" is generated and recovered (twin test).
#   --nlhs=N                 Latin-hypercube samples for the global search (default 48)
#   --out=DIR                output directory (default examples/inversion_out)
#
# Method: the flux history is a piecewise-linear table of log10(accretion rate) at fixed
# nodes in time. A Latin-hypercube ensemble of Q_magma forward models maps the misfit
# landscape; Nelder–Mead then polishes the best members. The misfit is the 1-D Wasserstein
# distance [ka] between the modelled and observed age distributions.
#
# Identifiability: with this forward model the zircon ages mostly record WHEN the injection
# zone cooled (tracers stay hot while the flux is high and crystallise together once it
# wanes), so the age distribution can be very narrow and constrains the flux history only
# weakly. summary.txt warns when the observed spread is narrow. The twin test (no --data)
# shows how well a known flux is recovered; run it before trusting a fit to real data.
# More constraining set-ups: fewer parameters (a ramp or pulse instead of 7 nodes), a
# distribution that spans the heating period, and extra observables such as erupted cargo
# ages (--eruptions --population=both) or erupted volumes.
#
# Outputs (in --out): best_flux_history.csv (readable by QMagma.load_flux_history),
# ensemble.csv (every forward model tried, ensemble and polish) and summary.txt.

include(joinpath(@__DIR__, "forward_Qmagma.jl"))
using Optim, Random, Statistics

# ─── Fixed model set-up (not fitted) ──────────────────────────────────────────────────────
# Coarser than the GUI defaults (Δz = 20 m, Δt = 100 yr, 2 tracers per sill) so that one
# model takes seconds. Re-run the best model at GUI resolution with `run_Q_forward` to check
# that the age distribution does not depend on this.
const BASE_MODEL = (;
    Δz = 50.0, Δt_yr = 250.0, nt = 1200,          # 300 kyr
    tracers_per_sill = 1, nx_zircon = 30,
    H = 40.0, γ = 20.0, Tsill = 1200.0, Silltop = 10.0, Sillbot = 20.0,
)

# Extra settings used only when eruptions are switched on. A chamber has to build up
# overpressure to erupt, which needs a shallow, wet, silicic source; with the dry basaltic
# default injected at 10-20 km the D&H trigger never fires.
function eruption_setup()
    comp = QMagma.gui_composition("MeltingParam_Rhyolite")
    ep = EruptionParams(;
        ΔP_crit = 20.0e6, ϕ_erupt = 0.5, m_w = 0.05, h_melt_min = 500.0,
        μ_shear = 10.0e9, R_sill = 5.0e3,
        solubility = comp.solubility, melt_viscosity = comp.melt_viscosity
    )
    return (; eruption = ep, melting = comp.melting, Tsill = 1000.0, Silltop = 3.0, Sillbot = 8.0)
end

# ─── Parameterisation ───────────────────────────────────────────────────────────────────
const NODE_TIMES_KYR = collect(range(0.0, 300.0, length = 7))
const LOG_RATE_BOUNDS = (-2.5, 0.0)           # log10 of the accretion rate [m/yr]

function flux_history(logrates)
    rates = 10.0 .^ clamp.(logrates, LOG_RATE_BOUNDS...) ./ SecYear
    return FluxHistory(:table; times = NODE_TIMES_KYR .* 1.0e3SecYear, rates)
end

# Quadratic penalty for leaving the box, so Nelder-Mead stays inside it
bound_penalty(p) = sum(x -> max(0.0, LOG_RATE_BOUNDS[1] - x)^2 + max(0.0, x - LOG_RATE_BOUNDS[2])^2, p)

# ─── Forward map and misfit ───────────────────────────────────────────────────────────────
"""
Model age sample [ka] of the chosen population, or `nothing` when it has no datable zircon.
"""
function model_ages(logrates, model; population = :reservoir)
    res = run_Q_forward(flux_history(logrates); model...)
    parts = Vector{Float64}[]
    population in (:reservoir, :both) && push!(parts, res.ages.age_years ./ 1.0e3)
    if population in (:erupted, :both) && res.ages_erupted !== nothing
        push!(parts, res.ages_erupted.age_years ./ 1.0e3)
    end
    isempty(parts) && return nothing
    ages = reduce(vcat, parts)
    return isempty(ages) ? nothing : ages
end

"""
    wasserstein1(a, b; nq=200)

1-D Wasserstein-1 distance between two samples: the mean absolute difference of their
quantile functions on `nq` levels. Unlike a histogram misfit it needs no binning and does
not depend on how many tracers the model carries.
"""
function wasserstein1(a, b; nq = 200)
    qs = range(0.5 / nq, 1 - 0.5 / nq, length = nq)
    return mean(abs.(quantile(a, qs) .- quantile(b, qs)))
end

const NO_ZIRCON_MISFIT = 1.0e3   # [ka] flat penalty for a model that grows no datable zircon

function misfit(logrates, observed, model; population = :reservoir)
    pen = bound_penalty(logrates)
    ages = try
        model_ages(logrates, model; population)
    catch err
        err isa ErrorException || rethrow()   # a failed thermal solve is a bad model, not a bug
        nothing
    end
    ages === nothing && return NO_ZIRCON_MISFIT + 1.0e3 * pen
    return wasserstein1(ages, observed) + 1.0e3 * pen
end

# ─── Observed data ───────────────────────────────────────────────────────────────────────
function load_ages(path)
    ages = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        v = tryparse(Float64, first(split(replace(s, ',' => ' '))))
        v === nothing ? (isempty(ages) || error("non-numeric line in $path: $s")) : push!(ages, v)
    end
    isempty(ages) && error("no ages found in $path")
    return ages
end

"""
Synthetic data: ages of a forward model with known flux, subsampled to `n_obs` crystals and
perturbed by Gaussian analytical error `σ_ka`.
"""
function synthetic_observations(truth, model; population, n_obs = 100, σ_ka = 2.0, seed = 7)
    ages = model_ages(truth, model; population)
    ages === nothing && error("the synthetic truth produces no datable zircon in population $population")
    rng = MersenneTwister(seed)
    picked = ages[rand(rng, 1:length(ages), n_obs)]
    return max.(picked .+ σ_ka .* randn(rng, n_obs), 0.0)
end

# ─── Search ──────────────────────────────────────────────────────────────────────────────
function latin_hypercube(rng, n, d, lo, hi)
    u = reduce(hcat, [(randperm(rng, n) .- rand(rng, n)) ./ n for _ in 1:d])   # n × d
    return lo .+ (hi - lo) .* u
end

function run_inversion(;
        eruptions = false, population = :reservoir, data = nothing, nlhs = 48,
        out = joinpath(@__DIR__, "inversion_out"), seed = 1,
        truth = [-1.3, -1.0, -0.6, -0.3, -0.8, -1.5, -2.0],
        n_polish = 1, polish_iterations = 80, tol_ka = 3.0
    )
    population in (:reservoir, :erupted, :both) || error("population must be :reservoir, :erupted or :both")
    eruptions || population === :reservoir ||
        error("population = $population needs eruptions (pass --eruptions)")
    mkpath(out)

    model = merge(BASE_MODEL, eruptions ? eruption_setup() : (;))
    model = merge(model, (; seed = 1))             # fixed cargo-sampling seed: deterministic misfit
    d = length(NODE_TIMES_KYR)
    println("eruptions = $eruptions, population = $population, $(Threads.nthreads()) thread(s)")

    if data === nothing
        println("no --data given: twin experiment, truth log10 rates = $truth")
        observed = synthetic_observations(truth, model; population)
    else
        observed = load_ages(data)
        truth = nothing
    end
    println("$(length(observed)) observed ages, median $(round(median(observed), digits = 1)) ka")

    # every evaluation is logged: the uncertainty envelope is built from all of them
    log_lock = ReentrantLock()
    log_P, log_J = Vector{Float64}[], Float64[]
    function f(p)
        val = misfit(p, observed, model; population)
        lock(log_lock) do
            push!(log_P, collect(p)); push!(log_J, val)
        end
        return val
    end

    # global search: independent models, so run them across threads
    rng = MersenneTwister(seed)
    P = latin_hypercube(rng, nlhs, d, LOG_RATE_BOUNDS...)
    J = zeros(nlhs)
    t0 = time()
    Threads.@threads for i in 1:nlhs
        J[i] = f(P[i, :])
    end
    println("ensemble of $nlhs models done in $(round(time() - t0, digits = 1)) s; best misfit $(round(minimum(J), digits = 2)) ka")

    # local polish of the best members (serial: each step depends on the last)
    best_p, best_J = P[argmin(J), :], minimum(J)
    for i in sortperm(J)[1:min(n_polish, nlhs)]
        r = Optim.optimize(
            f, P[i, :], NelderMead(), Optim.Options(iterations = polish_iterations)
        )
        println("  Nelder–Mead from member $i: $(round(J[i], digits = 2)) → $(round(Optim.minimum(r), digits = 2)) ka")
        if Optim.minimum(r) < best_J
            best_J, best_p = Optim.minimum(r), clamp.(Optim.minimizer(r), LOG_RATE_BOUNDS...)
        end
    end

    # Crude range of what the data cannot exclude: the global (Latin-hypercube) members
    # within `tol_ka` of the best misfit. Nelder-Mead points are not used - they cluster
    # around the optimum and would make every parameter look better constrained than it is.
    # With few members this is empty or sparse; raise --nlhs to resolve it.
    acceptable = J .<= best_J + tol_ka
    lo, hi = if any(acceptable)
        vec(minimum(P[acceptable, :], dims = 1)), vec(maximum(P[acceptable, :], dims = 1))
    else
        fill(NaN, d), fill(NaN, d)
    end

    best_ages = model_ages(best_p, model; population)
    allP = reduce(hcat, clamp.(q, LOG_RATE_BOUNDS...) for q in log_P)'
    write_outputs(
        out, best_p, best_J, allP, log_J, lo, hi, truth, eruptions, population,
        sum(acceptable), tol_ka, observed, best_ages
    )
    return (; best = best_p, misfit = best_J, observed, P = allP, J = log_J, truth)
end

function spread_line(ages)
    q = quantile(ages, [0.05, 0.5, 0.95])
    return "5/50/95 % = $(join(round.(q, digits = 1), " / ")) ka (n = $(length(ages)))"
end

function write_outputs(
        out, best, best_J, P, J, lo, hi, truth, eruptions, population, nacc, tol_ka,
        observed, best_ages
    )
    open(joinpath(out, "best_flux_history.csv"), "w") do io
        println(io, "time_kyr,flux_m_per_yr")
        for (t, lr) in zip(NODE_TIMES_KYR, best)
            println(io, t, ",", 10.0^lr)
        end
    end
    open(joinpath(out, "ensemble.csv"), "w") do io
        println(io, "misfit_ka,", join(("log10_rate_$(Int(t))kyr" for t in NODE_TIMES_KYR), ","))
        for i in sortperm(J)
            println(io, J[i], ",", join(P[i, :], ","))
        end
    end
    open(joinpath(out, "summary.txt"), "w") do io
        println(io, "eruptions: $eruptions, population: $population")
        println(io, "best misfit [ka]: $best_J")
        println(io, "observed ages: ", spread_line(observed))
        println(io, "best-model ages: ", spread_line(best_ages))
        if quantile(observed, 0.75) - quantile(observed, 0.25) < 5.0
            println(
                io, "WARNING: half of the observed ages fall within 5 ka. Zircons then record " *
                    "mainly when the system cooled, not the whole injection history, so the " *
                    "flux nodes are weakly constrained and the fit need not match the true " *
                    "history. Fit fewer parameters and/or add observables (see header)."
            )
        end
        println(io, "$nacc global-ensemble members within $tol_ka ka of the best misfit")
        println(io, rpad("t [kyr]", 10), rpad("best [m/yr]", 14), rpad("ensemble range [m/yr]", 24), truth === nothing ? "" : "truth [m/yr]")
        for i in eachindex(best)
            rng_txt = isnan(lo[i]) ? "n/a (raise --nlhs)" :
                "$(round(10.0^lo[i], sigdigits = 2))–$(round(10.0^hi[i], sigdigits = 2))"
            println(
                io, rpad(Int(NODE_TIMES_KYR[i]), 10), rpad(round(10.0^best[i], sigdigits = 3), 14),
                rpad(rng_txt, 24), truth === nothing ? "" : round(10.0^truth[i], sigdigits = 3)
            )
        end
    end
    println(read(joinpath(out, "summary.txt"), String))
    return nothing
end

# ─── Command line ────────────────────────────────────────────────────────────────────────
function parse_args(args)
    opts = Dict{Symbol, Any}(:eruptions => false, :population => :reservoir)
    for a in args
        a == "--eruptions" ? (opts[:eruptions] = true) :
            startswith(a, "--population=") ? (opts[:population] = Symbol(a[14:end])) :
            startswith(a, "--data=") ? (opts[:data] = a[8:end]) :
            startswith(a, "--nlhs=") ? (opts[:nlhs] = parse(Int, a[8:end])) :
            startswith(a, "--out=") ? (opts[:out] = a[7:end]) :
            error("unknown option $a")
    end
    return (; opts...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_inversion(; parse_args(ARGS)...)
end
