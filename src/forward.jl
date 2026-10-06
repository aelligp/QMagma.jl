# Headless Q_magma forward model: the Q_magma branch of the GUI loop
# (ext/QMagmaMakieExt/simulation.jl) without plotting, with all state local so many models
# can be run from one session (e.g. for an inversion, see examples/invert_flux.jl).

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
        tracers_per_sill::Integer = 2,
        zircon::Bool = true, zircon_tracers::Integer = 1, nx_zircon::Integer = 50,
        verbose::Bool = false
    )
    Δt = Δt_yr * SecYear
    nz = round(Int, H * 1.0e3 / Δz) + 1
    Tbot = Ttop + H * γ
    Params, BC, N, Δ, T, z = init_model(;
        nz, L = H * 1.0e3, Geotherm = γ, Ttop, Tbot, Δt, R_lat = R_sill,
        ρ = 2700.0, Q_L,
        Conductivity = ConstantConductivity(k = 3.0),
        HeatCapacity = ConstantHeatCapacity(),
        Melting = melting
    )
    MatParam = Params.MatParam
    Params.Told .= T
    T_background = copy(T)
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

    tracers = init_tracers(tracer_top, tracer_bot)
    erupted = Tracer[]
    rocks = zero(T)
    F = zero(T)
    rng = MersenneTwister(seed)
    erupt_state = EruptionState()
    events = Any[]
    margin = 5 * Δ[1]

    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))

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

        compute_Q_magma!(
            Params, MatParam, zv; Tsill, ȧ = ȧ_step,
            Silltop = source_top, Sillbot = source_bot, r = R_sill
        )
        rocks_adv = conservative_advection(rocks, Params.w .* Δt, zv)
        rocks = rocks_adv
        add_uniform_content!(rocks, zv, zone_lo, zone_hi, Δh)
        advect_w!(Params)
        compute_Q_magma!(
            Params, MatParam, zv; Tsill, ȧ = ȧ_step,
            Silltop = source_top, Sillbot = source_bot, r = R_sill
        )
        advect_tracers!(tracers, Params)
        for _ in 1:n_injections
            add_zone_tracers!(tracers, source_top, source_bot, Tsill; n = tracers_per_sill)
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
