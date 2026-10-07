# Headless Q_magma forward model: the Q_magma branch of the GUI loop
# (ext/QMagmaMakieExt/simulation.jl) without plotting. All state is local, so many models can be
# run from one session or in parallel (for example as the forward map of an inversion).

const INIT_LOCK = ReentrantLock()

"""
    run_Q_forward(ȧ; kwargs...) -> NamedTuple

Run the Q_magma emplacement model for the accretion history `ȧ` (a [`FluxHistory`](@ref) or
any callable `ȧ(t)` [m/s]) and return
`(; tracers, erupted, ages, T, ϕ, z, events, time)`.

`ages` is `compute_zircon_ages` of the reservoir tracers (and, with eruptions on,
`ages_erupted` of the erupted cargo) on one clock, `t_ref_Myr = end of run`; ages are
years before the end of the run. Pass `zircon=false` to skip it, or `zircon_tracers = k` to
use only every k-th tracer (cheaper, and a thinned sample for an optimizer).

# Keywords
- `H=40.0` [km], `γ=20.0` [°C/km], `Ttop=0.0` [°C]: crust and initial geotherm
- `Δz=20.0` [m], `Δt_yr=100.0`, `nt=3000`: grid and time stepping
- `Tsill=1200.0`, `Sillthick=100.0`, `Silltop=10.0`, `Sillbot=20.0`: injection (the
  thickness only sets the tracer-seeding cadence, as in the GUI; depth window in km)
- `R_sill=5.0e3` [m]: lateral extent entering the Q_magma velocity and erupted volumes
- `eruption=nothing`: an [`QMagma.EruptionParams`](@ref) turns on the D&H 3-phase chamber
  trigger, with `closure=:hybrid` (or `:caldera`) and `seed` for the cargo sampling
- `spinup_kyr=0`, `spinup_rate=0.1` [m/yr], `spinup_Δt_yr=1000`: spin up a mature system before the
  run (coarse steps; eruptions as in the run unless `spinup_eruption=false`; a few tracers are carried
  because the eruption bookkeeping needs them, but their histories are discarded); or pass
  the temperature profile `T_init` [°C] on the same grid directly
- `tracer_window=(top, bot)` [km]: depth range the host tracers are seeded over (default: the
  injection window); a spun-up chamber extends beyond it
- `zones=nothing`: several emplacement zones, a vector of `(top_km, bot_km, fraction)`; each takes
  its fraction of the accretion rate with its own source over `[top, bot]`. Injected tracers carry the
  zone number in `phase` (host rock stays 0). Overrides `Silltop`/`Sillbot`
- `host_tracers=20`: host-rock (phase 0) tracers seeded across the injection window at the start;
  they sample the resident, partly molten crust that mixes with the recharge, so an erupted-cargo
  study needs many more than the GUI's 20
- `tracers_per_sill=2`: tracers seeded per injection event (the GUI value). Run time scales
  with the tracer count, so an inversion can lower it
- `nx_zircon=50`, `verbose=false`
"""
function run_Q_forward(
        ȧ;
        H = 40.0, γ = 20.0, Ttop = 0.0, Δz = 20.0, Δt_yr = 100.0, nt = 3000,
        Tsill = 1200.0, Sillthick = 100.0, Silltop = 10.0, Sillbot = 20.0, R_sill = 5.0e3,
        Q_L = 255.0e3, melting = MeltingParam_Smooth3rdOrder(),
        eruption::Union{Nothing, EruptionParams} = nothing,
        closure::Symbol = :hybrid, seed::Integer = 1,
        tracers_per_sill::Integer = 2, host_tracers::Integer = 20,
        zircon::Bool = true, zircon_tracers::Integer = 1, nx_zircon::Integer = 50,
        T_init::Union{Nothing, AbstractVector} = nothing, tracer_window = nothing,
        zones = nothing,
        spinup_kyr::Real = 0.0, spinup_rate::Real = 0.1, spinup_Δt_yr::Real = 1000.0,
        spinup_eruption::Bool = true, verbose::Bool = false
    )
    # A mature system: run a coarse-step spin-up at a constant background rate first and start
    # from its thermal state, so the analysis does not have to grow its chamber from a cold
    # linear geotherm. Only the temperature is handed over; the analysis seeds its own tracers.
    if spinup_kyr > 0
        spin = run_Q_forward(
            FluxHistory(:constant; base = spinup_rate / SecYear);
            H, γ, Ttop, Δz, Δt_yr = spinup_Δt_yr, nt = round(Int, spinup_kyr * 1.0e3 / spinup_Δt_yr),
            Tsill, Sillthick, Silltop, Sillbot, R_sill, Q_L, melting,
            eruption = spinup_eruption ? eruption : nothing, closure, seed,
            tracers_per_sill = 2, host_tracers = 100, tracer_window = (4.0, 20.0), zircon = false, zones
        )
        T_init = spin.T
    end
    Δt = Δt_yr * SecYear
    nz = round(Int, H * 1.0e3 / Δz) + 1
    Tbot = Ttop + H * γ
    # GeoParams interns the material name in a global, non-thread-safe table, so concurrent
    # model set-up (an ensemble over threads) must be serialised; the time loop is thread-safe.
    Params, BC, N, Δ, T, z = lock(INIT_LOCK) do
        init_model(;
            nz, L = H * 1.0e3, Geotherm = γ, Ttop, Tbot, Δt, R_lat = R_sill,
            ρ = 2700.0, Q_L,
            Conductivity = ConstantConductivity(k = 3.0),
            HeatCapacity = ConstantHeatCapacity(),
            Melting = melting
        )
    end
    MatParam = Params.MatParam
    Params.Told .= T
    T_background = copy(T)            # the undisturbed geotherm, also with a spun-up start
    if T_init !== nothing
        length(T_init) == length(T) || throw(DimensionMismatch("T_init must have $(length(T)) points"))
        T .= T_init
        Params.Told .= T
    end
    zv = collect(z)

    # sparsity pattern of the 1D residual, Dirichlet rows decoupled
    J1 = Tridiagonal(ones(N[1] - 1), ones(N[1]), ones(N[1] - 1))
    J1[1, 2] = 0; J1[2, 1] = 0; J1[N[1] - 1, N[1]] = 0; J1[N[1], N[1] - 1] = 0
    Jac = sparse(Float64.(abs.(J1) .> 0))
    colors = matrix_colors(Jac)

    # depth-resolved histories (table with depths) move the injection window and the
    # tracer seeding zone, as in the GUI
    depth_forcing = ȧ isa FluxHistory && !isempty(ȧ.depths)
    if depth_forcing
        half_km = Sillthick / 2.0e3
        tracer_top = minimum(ȧ.depths) / 1.0e3 - half_km
        tracer_bot = maximum(ȧ.depths) / 1.0e3 + half_km
    else
        tracer_top, tracer_bot = Silltop, Sillbot
    end

    # Several emplacement zones: each takes its fraction of the accretion rate, with its own smeared
    # source over [top, bot] km. Sources and host-rock velocities add (both linear in the rate).
    zone_list = nothing
    if zones !== nothing
        tot = sum(Float64(zz[3]) for zz in zones)
        tot > 0 || throw(ArgumentError("zone fractions must sum to a positive number"))
        zone_list = [(Float64(zz[1]), Float64(zz[2]), Float64(zz[3]) / tot) for zz in zones]
        tracer_top, tracer_bot = minimum(zz[1] for zz in zone_list), maximum(zz[2] for zz in zone_list)
    end
    tracer_window === nothing || ((tracer_top, tracer_bot) = tracer_window)     # [km]
    tracers = init_tracers(tracer_top, tracer_bot; n = host_tracers)
    erupted = Tracer[]
    rocks = zero(T)
    F = zero(T)
    rng = MersenneTwister(seed)
    erupt_state = EruptionState()
    events = Any[]
    margin = 5 * Δ[1]

    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))

    Qacc, wacc = zero(T), zero(T)
    carry = zone_list === nothing ? Float64[] : zeros(length(zone_list))
    function set_sources!(ȧ_step, top, bot)
        if zone_list === nothing
            compute_Q_magma!(Params, MatParam, zv; Tsill, ȧ = ȧ_step, Silltop = top, Sillbot = bot, r = R_sill)
        else
            fill!(Qacc, 0.0); fill!(wacc, 0.0)
            for (zt, zb, f) in zone_list
                compute_Q_magma!(Params, MatParam, zv; Tsill, ȧ = ȧ_step * f, Silltop = zt, Sillbot = zb, r = R_sill)
                Qacc .+= Params.Q; wacc .+= Params.w
            end
            Params.Q .= Qacc; Params.w .= wacc
        end
        return nothing
    end

    time = 0.0
    A_inj = 0.0
    for step in 1:nt
        Δh = injected_thickness(ȧ, time, Δt)
        n_injections = sills_due(A_inj, Δh, Sillthick)
        A_inj += Δh
        ȧ_step = Δh / Δt
        depth_m = depth_forcing ? injection_depth(ȧ, time + Δt / 2) : nothing
        if depth_m === nothing
            source_top, source_bot = Silltop, Sillbot
        else
            source_top = depth_m / 1.0e3 - Sillthick / 2.0e3
            source_bot = depth_m / 1.0e3 + Sillthick / 2.0e3
        end
        zone_lo, zone_hi = -source_bot * 1.0e3, -source_top * 1.0e3

        set_sources!(ȧ_step, source_top, source_bot)
        rocks_adv = conservative_advection(rocks, Params.w .* Δt, zv)
        rocks = rocks_adv
        if zone_list === nothing
            add_uniform_content!(rocks, zv, zone_lo, zone_hi, Δh)
        else
            for (zt, zb, f) in zone_list
                add_uniform_content!(rocks, zv, -zb * 1.0e3, -zt * 1.0e3, Δh * f)
            end
        end
        advect_w!(Params)
        set_sources!(ȧ_step, source_top, source_bot)
        advect_tracers!(tracers, Params)
        if zone_list === nothing
            for _ in 1:n_injections
                add_zone_tracers!(tracers, source_top, source_bot, Tsill; n = tracers_per_sill)
            end
        else
            # tracers are shared out between the zones in proportion to their fraction, and tagged
            # with the zone (phase = zone number; 0 stays the host rock)
            for (k, (zt, zb, f)) in enumerate(zone_list)
                expected = n_injections * tracers_per_sill * f + carry[k]
                nk = floor(Int, expected); carry[k] = expected - nk
                n0 = length(tracers)
                nk > 0 && add_zone_tracers!(tracers, zt, zb, Tsill; n = nk)
                for i in (n0 + 1):length(tracers)
                    tracers[i].phase = k
                end
            end
        end
        T, converged, its = nonlinear_solution(
            F, T, Jac, colors; verbose = false, Δ, N, BC, Params, MatParam
        )
        converged || error("Q_magma thermal solve failed at step $step after $its iterations")
        Params.Told .= T
        compute_meltfraction!(
            Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,)
        )

        time += Δt
        time_Myr = time / SecYear / 1.0e6
        update_tracers_T!(tracers, T, zv, time_Myr, Params.ϕ)

        if eruption !== nothing
            T, rocks, cargo, event = step_chamber_eruption!(
                rng, erupt_state, eruption, T, rocks, tracers, Params.ϕ, zv, MatParam,
                Params.Phases; ȧ = ȧ_step, Δt, time, closure, margin, T_background
            )
            if event !== nothing
                append!(erupted, cargo)
                push!(events, (; time_kyr = time / SecYear / 1.0e3, event))
                Params.Told .= T
                compute_meltfraction!(
                    Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,)
                )
                verbose && println("Eruption @ $(round(time / SecYear / 1.0e3, digits = 1)) kyr")
            end
        end
    end

    out = (; tracers, erupted, T, ϕ = copy(Params.ϕ), z = zv, events, time, chamber = erupt_state)
    zircon || return out

    t_ref = time / SecYear / 1.0e6
    pick = zircon_tracers > 1 ? tracers[1:zircon_tracers:end] : tracers
    ages = compute_zircon_ages(pick; nx = nx_zircon, t_ref_Myr = t_ref)
    ages_erupted = isempty(erupted) ? nothing :
        compute_zircon_ages(erupted; nx = nx_zircon, t_ref_Myr = t_ref)
    return (; out..., ages, ages_erupted)
end
