using GeoParams
using ForwardDiff, SparseArrays, SparseDiffTools, LinearAlgebra, Interpolations, Random
using ZirconGrowth

const SecYear = 3600 * 24 * 365.25

av(x) = (x[2:end]+x[1:end-1])/2

"""
    grid_cell_edges(z)

Control-volume edges for a node-centered, strictly ascending grid. The physical domain
is `[z[1], z[end]]`; boundary nodes therefore own half cells on a uniform grid.
"""
function grid_cell_edges(z)
    length(z) >= 2 || throw(ArgumentError("z must contain at least two points"))
    zv = collect(float.(z))
    all(isfinite, zv) || throw(ArgumentError("z must be finite"))
    all(>(0), diff(zv)) || throw(ArgumentError("z must be strictly ascending"))
    edges = similar(zv, length(zv) + 1)
    edges[1] = zv[1]
    edges[2:end-1] .= av(zv)
    edges[end] = zv[end]
    return edges
end

function interval_cell_widths(z, z_lo=first(z), z_hi=last(z))
    edges = grid_cell_edges(z)
    first(z) <= z_lo <= z_hi <= last(z) || throw(ArgumentError(
        "integration interval [$z_lo, $z_hi] must lie inside [$(first(z)), $(last(z))]"))
    return [max(0.0, min(edges[i + 1], z_hi) - max(edges[i], z_lo))
            for i in eachindex(z)]
end

"""
    integrated_content(field, z[, z_lo, z_hi])

Integrate a node-centered control-volume field over the physical grid domain or a
clipped depth interval. This gives boundary nodes half-cell weight on a uniform grid,
so a constant field integrates to `z_hi - z_lo` rather than one extra grid spacing.
"""
function integrated_content(field, z, z_lo=first(z), z_hi=last(z))
    length(field) == length(z) ||
        throw(DimensionMismatch("field and z must have equal length"))
    all(isfinite, field) || throw(ArgumentError("field must be finite"))
    isfinite(z_lo) && isfinite(z_hi) ||
        throw(ArgumentError("integration bounds must be finite"))
    widths = interval_cell_widths(z, z_lo, z_hi)
    total = zero(promote_type(eltype(field), eltype(widths)))
    for i in eachindex(field)
        widths[i] > 0 && (total += field[i] * widths[i])
    end
    return total
end

"""
    add_uniform_content!(field, z, z_lo, z_hi, amount)

Add an exactly integrated `amount` to `field`, uniformly over `[z_lo, z_hi]`, using
the same control volumes as [`integrated_content`](@ref).
"""
function add_uniform_content!(field, z, z_lo, z_hi, amount)
    length(field) == length(z) ||
        throw(DimensionMismatch("field and z must have equal length"))
    all(isfinite, (z_lo, z_hi, amount)) ||
        throw(ArgumentError("bounds and amount must be finite"))
    first(z) <= z_lo < z_hi <= last(z) || throw(ArgumentError(
        "addition interval [$z_lo, $z_hi] must lie inside [$(first(z)), $(last(z))]"))
    amount >= 0 || throw(ArgumentError("amount must be nonnegative"))
    edges = grid_cell_edges(z)
    concentration = amount / (z_hi - z_lo)
    for i in eachindex(field)
        overlap = min(edges[i + 1], z_hi) - max(edges[i], z_lo)
        overlap > 0 && (field[i] += concentration * overlap / (edges[i + 1] - edges[i]))
    end
    return field
end

function nonnegative_debit(before, after, label)
    debit = before - after
    tolerance = 128eps(max(abs(before), abs(after), 1.0))
    debit >= 0 && return debit
    debit >= -tolerance && return 0.0
    throw(ArgumentError("$label increased by $(-debit) during an outflow-only remap"))
end


"""
    init_model(; nz=101, L=40e3, Geotherm=0, Ttop=400.0, Tbot=0.0,
               Δt=1e3*SecYear, MatParam=nothing, ρ=nothing, Q_L=nothing,
               Conductivity=nothing, HeatCapacity=nothing, Melting=nothing)

Create initial model setup.

The material parameters are assembled here from the individual constitutive laws, so every
entry point (GUI, scripts, tests) shares one definition of the host-rock properties. `ρ`
[kg/m³] is the host-rock *thermal* density, the one that enters heat storage and
conduction; it is a separate role from the melt, crystal and lithostatic densities on
[`EruptionParams`](@ref), which [`check_density_consistency`](@ref) keeps in step. Pass a
ready-made `MatParam` tuple only to reuse one built elsewhere; combining it with any
component keyword is an error because those components would otherwise be ignored.
"""
function init_model(;nz=101, L=40e3, Geotherm=0, Ttop=400.0, Tbot=0.0, Δt=1e3*SecYear, MatParam=nothing,
                    ρ=nothing, Q_L=nothing, Conductivity=nothing,
                    HeatCapacity=nothing, Melting=nothing)
    if isnothing(MatParam)
        ρ = something(ρ, 2700.0)
        Q_L = something(Q_L, 2.55e5)
        Conductivity = something(Conductivity, T_Conductivity_Whittington())
        HeatCapacity = something(HeatCapacity, T_HeatCapacity_Whittington())
        Melting = something(Melting, MeltingParam_Assimilation())
        MatParam     = (SetMaterialParams(; Name="RockMelt", Phase=0,
                            Density         = ConstantDensity(; ρ=ρ*kg/m^3),
                            LatentHeat      = ConstantLatentHeat(Q_L=Q_L*J/kg),
                            RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                            Conductivity, HeatCapacity, Melting),
                    )
    elseif any(x -> !isnothing(x), (ρ, Q_L, Conductivity, HeatCapacity, Melting))
        throw(ArgumentError("MatParam cannot be combined with component material keywords"))
    end

    # Numerics
    Told        =   zeros(nz)
    T           =   zeros(nz)
    ρ           =   zeros(nz)
    Cp          =   zeros(nz)
    dϕdT        =   zeros(nz)
    ϕ           =   zeros(nz)
    Hl          =   zeros(nz)
    Q           =   zeros(nz)     # volumetric heat source term [W/m^3]
    w           =   zeros(nz)     # vertical advection velocity [m/s]
    k           =   zeros(nz-1)
    dz          =   L/(nz-1)
    z           =   -L:dz:0
    T           =   -Geotherm/1e3.*Vector(z) .+ Ttop
    Told        =   -Geotherm/1e3.*Vector(z) .+ Ttop

    Phases      =   fill(0,nz)
    Phases_c    =   fill(0,nz-1)

    Params      =   (; Δt, k, ρ, Cp, dϕdT, ϕ, Hl, Q, w, Told, Phases, Phases_c, MatParam, z)
    N           =   (nz,)
    BC          =   (; Ttop, Tbot)
    Δ           =   (dz,)

    return Params, BC, N, Δ, T, z
end

"""
    update_properties!(Params, MatParam)

Evaluate all temperature-dependent material properties (k, Cp, ρ, dϕ/dT, ϕ, latent
heat) from `Params.Told` and cache them in `Params`. These depend only on the *old*
temperature, not on the Newton unknown `T`, so the residual is linear in `T` and the
properties stay constant across the solve. Call this once per timestep (done in
`nonlinear_solution`) instead of recomputing the (expensive) GeoParams evaluations on
every residual / Jacobian / line-search evaluation.
"""
function update_properties!(Params, MatParam)
    args   = (T = Params.Told .+ 273.15,)
    args_c = (T = av(Params.Told) .+ 273.15,)
    compute_conductivity!(Params.k, MatParam, Params.Phases_c, args_c)
    compute_heatcapacity!(Params.Cp, MatParam, Params.Phases, args)
    compute_density!(Params.ρ, MatParam, Params.Phases, args)
    compute_dϕdT!(Params.dϕdT, MatParam, Params.Phases, args)
    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, args)
    compute_latent_heat!(Params.Hl, Params.MatParam, Params.Phases, args)
    return Params
end

"""
    Res!(F::AbstractArray, T::AbstractArray, Δ, N, BC)
"""
function Res!(F::AbstractVector{_T}, T::AbstractVector{_T}, Δ::NTuple, N::NTuple, BC::NamedTuple, Params::NamedTuple, MatParam) where _T<:Number

    dz     = Δ[1]       # grid spacing
    nz     = N[1]       # grid size

    # Material properties (k, Cp, ρ, dϕ/dT, ϕ, Hl) are frozen at Params.Told and cached
    # once per step by update_properties!; they don't depend on the unknown T (the
    # residual is linear in T), so they are read here, not recomputed on every eval.
    I          = 2:nz-1
    #  ρ(Cp + Hₗ∂ϕ/∂T) ∂T/∂t = ∂/∂z(k ∂T/∂z) + Q
    # (host-rock advection, if any, is applied to Params.Told beforehand via a
    # semi-Lagrangian remap - see `advect_w!` - rather than as a term here)
    F[2:end-1] = Params.ρ[I].*(Params.Cp[I]  + Params.Hl[I].*Params.dϕdT[I]).*(T[I]-Params.Told[I])/Params.Δt  -   diff(Params.k .* diff(T)/dz)/dz   .-   Params.Q[I];

    F[1]  = T[1]  - BC.Tbot
    F[nz] = T[nz] - BC.Ttop

    return F
end

"""
    LineSearch(func::Function, F, x, δx; α=[0.01 0.05 0.1 0.25 0.5 0.75 1.0])

Pick the step size out of `α` that minimizes the residual norm of `func` at `x + α*δx`,
and return it together with that norm.
"""
function LineSearch(func::Function, F, x, δx;  α = [0.01 0.05 0.1 0.25 0.5 0.75 1.0])
    Fnorm = zero(α)
    N     = length(x)
    for i in eachindex(α)
        func(F, x .+ α[i].*δx)
        Fnorm[i] = norm(F)/N
    end
    _, i_opt = findmin(Fnorm)
    return α[i_opt], Fnorm[i_opt]
end


"""
    Usol = nonlinear_solution(Fup::Vector, U::Vector{<:AbstractArray}, J, colors; tol=1e-8, maxit=100)

Computes a nonlinear solution using a Newton method with line search.
`U` needs to be a vector of abstract arrays, which contains the initial guess of every field
`J` is the sparse jacobian matrix, and `colors` the coloring matrix, usually computed with `matrix_colors(J)`
"""
function nonlinear_solution(Fup::Vector, T::Vector, J, colors; tol=1e-8, maxit=100, verbose=true,
                            Δ, N, BC, Params, MatParam)

    Res_closed! = (F,T) -> Res!(F, T, Δ, N, BC, Params, MatParam)

    update_properties!(Params, MatParam)   # frozen-coefficient props: once per step, not per eval

    r   = zero(Fup)
    err = 1e3; it=0;
    while err>tol && it<maxit
        Res_closed!(r,T)     # compute residual

        forwarddiff_color_jacobian!(J, Res_closed!, T, colorvec = colors) # compute jacobian in an in-place manner

        dT      =   J\-r    # solve linear system:
        α, err  =   LineSearch(Res_closed!, r, T, dT); # optimal step size
        T       +=   α*dT   # update solution
        it      +=1;
        if verbose; println("   Nonlinear iteration $it: error = $err, α=$α"); end
    end

    converged = err <= tol

    return T, converged, it
end

"""
    FluxHistory(mode; base, peak=base, t_start=0, t_end=0, times=[], rates=[])

Validated magma-accretion history in internal units (seconds and m/s). Supported modes:
`:constant`, `:ramp` (linear from `base` to `peak`), `:pulse` (`peak` between
`t_start` and `t_end`, `base` otherwise), and `:table` (piecewise-linear `times`,`rates`
with flat extrapolation).
"""
struct FluxHistory
    mode    :: Symbol
    base    :: Float64
    peak    :: Float64
    t_start :: Float64
    t_end   :: Float64
    times   :: Vector{Float64}
    rates   :: Vector{Float64}
end

function FluxHistory(mode; base=0.0, peak=base, t_start=0.0, t_end=0.0,
                     times=Float64[], rates=Float64[])
    mode in (:constant, :ramp, :pulse, :table) ||
        throw(ArgumentError("flux mode must be :constant, :ramp, :pulse, or :table"))
    base, peak, t_start, t_end = Float64.((base, peak, t_start, t_end))
    times, rates = Float64.(times), Float64.(rates)
    all(isfinite, (base, peak, t_start, t_end)) && all(isfinite, times) && all(isfinite, rates) ||
        throw(ArgumentError("flux-history values must be finite"))
    base >= 0 && peak >= 0 && all(>=(0), rates) ||
        throw(ArgumentError("flux rates must be nonnegative"))
    if mode in (:ramp, :pulse)
        0 <= t_start < t_end ||
            throw(ArgumentError("ramp/pulse times must satisfy 0 ≤ t_start < t_end"))
    elseif mode == :table
        length(times) == length(rates) ||
            throw(DimensionMismatch("table times and rates must have equal length"))
        length(times) >= 2 || throw(ArgumentError("flux table must contain at least two rows"))
        all(>(0), diff(times)) || throw(ArgumentError("flux-table times must be strictly increasing"))
    end
    return FluxHistory(mode, base, peak, t_start, t_end, times, rates)
end

function (history::FluxHistory)(time)
    isfinite(time) || throw(ArgumentError("time must be finite"))
    if history.mode == :constant
        return history.base
    elseif history.mode == :ramp
        time <= history.t_start && return history.base
        time >= history.t_end && return history.peak
        f = (time - history.t_start) / (history.t_end - history.t_start)
        return history.base + f*(history.peak - history.base)
    elseif history.mode == :pulse
        return history.t_start <= time < history.t_end ? history.peak : history.base
    end
    time <= history.times[1] && return history.rates[1]
    time >= history.times[end] && return history.rates[end]
    i = searchsortedlast(history.times, time)
    f = (time - history.times[i]) / (history.times[i + 1] - history.times[i])
    return history.rates[i] + f*(history.rates[i + 1] - history.rates[i])
end

"""
    load_flux_history(path; time_scale=1000SecYear, rate_scale=1/SecYear)

Read a two-column text/CSV table (`time_kyr, flux_m_per_yr` by default) and return a
piecewise-linear [`FluxHistory`](@ref). One header row plus blank and `#` comment lines are
accepted; malformed data fail with the source line number.
"""
function load_flux_history(path; time_scale=1000SecYear, rate_scale=1/SecYear)
    isfile(path) || throw(ArgumentError("flux table does not exist: $path"))
    all(isfinite, (time_scale, rate_scale)) && time_scale > 0 && rate_scale > 0 ||
        throw(ArgumentError("table unit scales must be finite and positive"))
    times, rates = Float64[], Float64[]
    header_skipped = false
    for (line_number, raw) in enumerate(eachline(path))
        line = strip(raw)
        (isempty(line) || startswith(line, '#')) && continue
        fields = split(replace(line, ',' => ' '))
        if length(fields) != 2
            throw(ArgumentError("flux table line $line_number must contain exactly two columns"))
        end
        time = tryparse(Float64, fields[1])
        rate = tryparse(Float64, fields[2])
        if time === nothing || rate === nothing
            if isempty(times) && !header_skipped
                header_skipped = true
                continue
            end
            throw(ArgumentError("flux table line $line_number contains nonnumeric data"))
        end
        push!(times, time*time_scale)
        push!(rates, rate*rate_scale)
    end
    return FluxHistory(:table; times, rates)
end

function _integrate_linear_flux(history::FluxHistory, time, Δt, breakpoints)
    stop = time + Δt
    interior = [t for t in breakpoints if time < t < stop]
    points = vcat(time, interior, stop)
    return sum((history(points[i]) + history(points[i + 1])) *
               (points[i + 1] - points[i]) / 2 for i in 1:length(points)-1)
end

function injected_thickness(history::FluxHistory, time, Δt)
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    isfinite(time) || throw(ArgumentError("time must be finite"))
    if history.mode == :constant
        return history.base*Δt
    elseif history.mode == :pulse
        overlap = max(0.0, min(time + Δt, history.t_end) - max(time, history.t_start))
        return history.base*Δt + (history.peak - history.base)*overlap
    elseif history.mode == :ramp
        return _integrate_linear_flux(history, time, Δt, (history.t_start, history.t_end))
    end
    return _integrate_linear_flux(history, time, Δt, history.times)
end

"""
    injected_thickness(ȧ, time, Δt) -> Δh

Magma thickness [m] accreted during the step `(time, time + Δt]` for the accretion-rate
history `ȧ` [m/s], integrated by the midpoint rule (exact for a constant rate). `ȧ` is
either a number or a callable `ȧ(t)`.

This is the single forcing quantity both emplacement branches derive from: the discrete
branch counts sills per accumulated thickness ([`sills_due`](@ref)) and the smeared branch
uses the step-mean rate `Δh/Δt` ([`compute_Q_magma!`](@ref)), so the two cannot drift apart
under a variable rate.
"""
function injected_thickness(ȧ, time, Δt)
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    isfinite(time) || throw(ArgumentError("time must be finite"))
    rate = ȧ isa Number ? ȧ : ȧ(time + Δt/2)
    isfinite(rate) && rate >= 0 ||
        throw(ArgumentError("accretion rate must be finite and nonnegative, got $rate"))
    return rate*Δt
end

"""
    sills_due(A, ΔA, d) -> Int

Number of sills of aperture `d` [m] completed while the cumulative injected thickness grows
from `A` to `A + ΔA`, i.e. `floor((A+ΔA)/d) - floor(A/d)`. More than one sill is returned
when a step delivers more than one aperture, and none while the accretion rate is zero.

Emplacement is keyed to injected thickness rather than to elapsed time so that a varying
accretion rate changes the event *frequency* at fixed aperture. For a constant rate
`ȧ = d/interval` this reproduces the event times of an interval-keyed schedule.
"""
function sills_due(A, ΔA, d)
    all(isfinite, (A, ΔA, d)) || throw(ArgumentError("A, ΔA, and d must be finite"))
    A >= 0 || throw(ArgumentError("A must be nonnegative"))
    ΔA >= 0 || throw(ArgumentError("ΔA must be nonnegative"))
    d > 0 || throw(ArgumentError("sill aperture must be positive"))

    sill_index(a) = floor(Int, a / d + 8eps(a / d))
    return sill_index(A + ΔA) - sill_index(A)
end

"""
    mean_sill_velocity(z, ȧ, zbot, ztop; r=5e3)

Time-averaged host-rock velocity generated by sills whose centers are uniformly
distributed between `zbot` and `ztop`. Each sill has full aperture `d`, so each wall
moves by `d/2`; multiplying by the event rate `ȧ/d` eliminates `d` from the mean.
The elastic decay is the same as [`crack_perp_displacement`](@ref).
"""
function mean_sill_velocity(z, ȧ, zbot, ztop; r=5e3)
    ȧ >= 0 || throw(ArgumentError("ȧ must be nonnegative"))
    ztop > zbot || throw(ArgumentError("ztop must be greater than zbot"))
    r > 0 || throw(ArgumentError("r must be positive"))

    H = ztop - zbot
    primitive(x) = abs(x) - sqrt(r^2 + x^2) + r
    return (ȧ / (2H)) .* (primitive.(z .- zbot) .- primitive.(z .- ztop))
end

"""
    compute_Q_magma!(Params, MatParam, z; Tsill, ȧ, Silltop, Sillbot)

Smears repeated sill injection into a steady volumetric heat source `Params.Q` [W/m^3],
following

    Q_magma(z,t) = ρₘ ȧ/H [ cp(Tₘ - T(z,t)) + L(1-ϕ(T)) ]

over the injection zone `z ∈ [-Sillbot, -Silltop]` (in m) of thickness `H = Sillbot-Silltop`,
and zero elsewhere. `ȧ` is the step-mean accretion rate [m/s] supplied by the shared
flux history.
The first term is the sensible heat magma surrenders cooling from `Tsill` to the local
temperature; the second is the latent heat released crystallizing from ϕ=1 (injected liquid)
down to the local melt fraction ϕ(T). ρ, cp and ϕ are (re-)evaluated here from `Params.Told`,
consistent with how the rest of the residual is linearized.

ρₘ is the host-rock thermal density of `MatParam` at the local temperature, not the melt
density `EruptionParams.ρ_melt`: the source therefore carries the enthalpy of a magma with
the host rock's density, an approximation at the ~10% level of the injected heat.

Also sets `Params.w` to the ensemble-mean host-rock velocity produced by discrete
sills with uniformly distributed centers in the injection zone. This uses the same
elastic displacement profile and full-aperture convention as [`insert_sill`](@ref),
including the nonzero velocity within the zone away from its midpoint.
"""
function compute_Q_magma!(Params, MatParam, z; Tsill, ȧ, Silltop, Sillbot, r=5e3)
    all(isfinite, (Tsill, ȧ, Silltop, Sillbot, r)) ||
        throw(ArgumentError("source parameters must be finite"))
    ȧ >= 0 || throw(ArgumentError("ȧ must be nonnegative"))
    Sillbot > Silltop || throw(ArgumentError("Sillbot must be greater than Silltop"))
    r > 0 || throw(ArgumentError("r must be positive"))

    zbot, ztop = -Sillbot*1e3, -Silltop*1e3
    first(z) <= zbot < ztop <= last(z) || throw(ArgumentError(
        "injection zone [$zbot, $ztop] m must lie inside [$(first(z)), $(last(z))] m"))

    args = (T = Params.Told .+ 273.15,)
    compute_heatcapacity!(Params.Cp, MatParam, Params.Phases, args)
    compute_density!(Params.ρ, MatParam, Params.Phases, args)
    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, args)

    Q_L = NumValue(MatParam[1].LatentHeat[1].Q_L)
    H   = (Sillbot - Silltop)*1e3

    Params.Q .= 0.0
    ind = findall(zbot .<= z .<= ztop)
    isempty(ind) && throw(ArgumentError("injection zone contains no grid points"))
    Params.Q[ind] .= Params.ρ[ind].*(ȧ/H).*( Params.Cp[ind].*(Tsill .- Params.Told[ind]) .+ Q_L.*(1.0 .- Params.ϕ[ind]) )

    Params.w .= mean_sill_velocity(z, ȧ, zbot, ztop; r)

    return Params.Q
end

"""
    crack_perp_displacement(z, d; r=5e3)

Wall displacement `d(1 - |z|/√(r²+z²))` of an elastic crack of radius `r` at
perpendicular distance `z` from its plane: `d` at the crack face, decaying to zero far
away. Used both to open a sill ([`insert_sill`](@ref)) and to close an erupted band
([`collapse_displacement`](@ref)).
"""
crack_perp_displacement(z, d; r=5e3) = d.*(1.0 .- abs.(z)./(sqrt.(r^2 .+ z.^2)))

"""
    T_adv, rock_adv, h_out = insert_sill(T, rocks, z; Sill_thick=400, Sill_z0=-20e3, Sill_T=1200, SillType=:elastic)

Adds a sill to the setup, advecting the temperature field `T` and the injected-magma
indicator `rocks` on a grid `z` apart to make room for it, then control-volume mixing the
sill interval at `Sill_T` and adding its `Sill_phase` content. Optional parameters are the sill thickness `Sill_thick`, the sill
center `Sill_z0`, the sill temperature `Sill_T`. Advection is done by `SillType`, which can
be `:constant` (where rocks above/below are moved with constant displacement) or `:elastic`,
where the displacement decreases with distance from the sill.

The third return value `h_out` is the injected-magma content [m] pushed out through a domain
boundary by that displacement, the boundary term of the magma-volume budget
([`MassBudget`](@ref)).
"""
function insert_sill(T,rocks, z; Sill_thick=400, Sill_z0=-20e3, Sill_T=1200, Sill_phase=1.0, SillType=:elastic)
    length(T) == length(rocks) == length(z) ||
        throw(DimensionMismatch("T, rocks, and z must have equal length"))
    all(isfinite, (Sill_thick, Sill_z0, Sill_T, Sill_phase)) ||
        throw(ArgumentError("sill parameters must be finite"))
    Sill_thick > 0 || throw(ArgumentError("Sill_thick must be positive"))
    Sill_phase >= 0 || throw(ArgumentError("Sill_phase must be nonnegative"))
    SillType in (:constant, :elastic) ||
        throw(ArgumentError("SillType must be :constant or :elastic"))
    z_lo, z_hi = Sill_z0 - Sill_thick/2, Sill_z0 + Sill_thick/2
    first(z) <= z_lo < z_hi <= last(z) || throw(ArgumentError(
        "sill interval [$z_lo, $z_hi] m must lie inside [$(first(z)), $(last(z))] m"))

    # find points above & below sill emplacement level
    z_shift = Vector(z) .-  Sill_z0;
    Displ   = zero(z_shift)

    # shift points above
    id_above = findall(z_shift.>0)
    id_below = findall(z_shift.<0)

    # Assume constant displacement - in elastic case this should decrease with distance from sill
    if SillType==:constant
        Displ[id_above]  .= Sill_thick/2
        Displ[id_below]  .= -Sill_thick/2
    elseif SillType==:elastic
        R = 5e3;
        Displ[id_above]  .=  crack_perp_displacement(z_shift[id_above], Sill_thick/2; r=R)
        Displ[id_below]  .= -crack_perp_displacement(z_shift[id_below], Sill_thick/2; r=R)
    end

    # use WENO5 to advect the temperature field
    T_adv = semilagrangian_advection(T, Displ, z)

    # Mix the new sill into each overlapped control volume. This remains defined when a
    # sill is thinner than the grid spacing or falls between nodes.
    edges = grid_cell_edges(z)
    sill_widths = interval_cell_widths(z, z_lo, z_hi)
    sill_fraction = sill_widths ./ diff(edges)
    T_adv .= (1 .- sill_fraction) .* T_adv .+ sill_fraction .* Sill_T

    # move host rock and previously injected material with the same (elastic or constant)
    # displacement as T. The indicator stores injected-magma content per grid control volume;
    # values can exceed one where emplacement compresses previously injected material.
    rock_adv = conservative_advection(rocks, Displ, z)
    h_out = nonnegative_debit(integrated_content(rocks, z),
        integrated_content(rock_adv, z), "injected-magma content")
    add_uniform_content!(rock_adv, z, z_lo, z_hi, Sill_phase*Sill_thick)

    return T_adv, rock_adv, h_out
end

"""
    Tadv = semilagrangian_advection(T, Displ, z)
Do semilagrangian_advection
"""
function semilagrangian_advection(T, Displ, z)

    z_new = z + Displ # advect grid
    all(>(0), diff(z_new)) ||
        throw(ArgumentError("displacement folds the grid; advected coordinates must be strictly ascending"))
    interp_linear = linear_interpolation(z_new, T);
    T_adv = interp_linear.(z)

    return T_adv
end

"""
    conservative_advection(field, Displ, z)

Finite-volume (mass-conserving) advection of a per-cell `field` by the displacement
`Displ` [m] on grid `z`. Unlike [`semilagrangian_advection`](@ref) — which
interpolates the *value* and lets integrated content drift wherever the displacement
stretches or compresses the grid — this moves each control volume's content: cell
edges are displaced (Displ interpolated to the edges) and every parcel deposits its
conserved content onto the fixed cells it overlaps. [`integrated_content`](@ref) is preserved exactly
(up to parcels pushed past the domain boundary).

Use this for the injected-rock phase indicator, whose integral is the injected crust
thickness: repeated non-conservative advection erodes thin grey bands (a one-way loss
of ~2-3% per injection). Temperature keeps using `semilagrangian_advection` — an
intensive field where the stretch-induced drift is small and expected.
"""
function conservative_advection(field, Displ, z)
    length(field) == length(Displ) == length(z) ||
        throw(DimensionMismatch("field, Displ, and z must have equal length"))
    all(isfinite, field) && all(isfinite, Displ) ||
        throw(ArgumentError("field and Displ must be finite"))
    n  = length(z)
    ze = grid_cell_edges(z)
    widths = diff(ze)
    de = linear_interpolation(z, Displ; extrapolation_bc=Line()).(ze)
    xe = ze .+ de
    all(>(0), diff(xe)) ||
        throw(ArgumentError("displacement folds a control volume; displaced edges must be strictly ascending"))
    out = zeros(promote_type(eltype(field), eltype(ze)), n)
    for j in 1:n
        L, R = xe[j], xe[j+1]
        w = R - L
        density = field[j]*widths[j]/w
        i1 = clamp(searchsortedlast(ze, L), 1, n)
        i2 = clamp(searchsortedfirst(ze, R) - 1, 1, n)
        for i in i1:i2
            ov = min(R, ze[i+1]) - max(L, ze[i])
            ov > 0 && (out[i] += density*ov/widths[i])
        end
    end
    return out
end

function largest_contiguous_range(mask)
    best_len = 0
    best_lo  = 0
    best_hi  = 0
    i = 1
    n = length(mask)
    while i <= n
        if mask[i]
            run_lo = i
            j = i
            while j <= n && mask[j]
                j += 1
            end
            run_len = j - run_lo
            if run_len > best_len
                best_len = run_len
                best_lo  = run_lo
                best_hi  = j - 1
            end
            i = j
        else
            i += 1
        end
    end
    return best_len == 0 ? nothing : best_lo:best_hi
end

"""
    find_eruptible_region(ϕ, z; ϕ_threshold=0.5)

Find the envelope `[z_bot, z_top]` of the largest contiguous run where `ϕ` exceeds
`ϕ_threshold`. Separate melt lenses are not combined. Returns `nothing` if no point
exceeds the threshold.
"""
function find_eruptible_region(ϕ, z; ϕ_threshold=0.5)
    length(ϕ) == length(z) || throw(DimensionMismatch("ϕ and z must have equal length"))
    run = largest_contiguous_range(ϕ .> ϕ_threshold)
    run === nothing && return nothing

    return z[first(run)], z[last(run)]
end

"""
    collapse_advection(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2)

Move every grid point's *position* toward the eruption zone `[Erupt_z0-half, Erupt_z0+half]`
using the same elastic decay law `insert_sill` uses to open a sill (largest right at the
band edge, decaying away from it with radius `R`), then interpolate the original field `T`
at those moved positions back onto the fixed grid `z`. This is the direct inverse of
`insert_sill`'s opening displacement: instead of pushing host rock apart to make room for
new material, it pulls host rock together to close a gap left by erupted material.

Points outside the band move toward `Erupt_z0` by exactly enough to fully close their side
of the gap (not just asymptotically approach it - the naive `insert_sill`-style amplitude
falls fractionally short once evaluated at an actual grid point rather than the idealized
edge, which would otherwise leave a sliver of the band uncollapsed). Points inside the band
are swept toward `Erupt_z0` too, packed into a tiny window around it so the displaced grid
stays strictly monotonic (required for the interpolation) without colliding exactly on top
of each other.
"""
function collapse_advection(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)
    zv = Vector(z)
    n = length(zv)
    half = Erupt_thick/2

    Displ = collapse_displacement(zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, R=R, method=method)
    # for the hybrid the top boundary is unpinned (the surface subsides by
    # Erupt_thick): flat extrapolation fills the vacated cells above the subsided
    # surface with the boundary value (Ttop)
    itp = linear_interpolation(zv .+ Displ, Vector(T); extrapolation_bc=Flat())
    T_new = itp.(zv)

    # the handful of grid points exactly at the collapse center can still backtrace onto
    # their own pre-collapse value due to floating-point collisions in the interpolation;
    # patch them with a clean linear interpolation between the nearest already-correctly
    # -advected neighbors just outside the band
    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    if !isempty(ind) && ind[1] > 1 && ind[end] < n
        lo, hi = ind[1]-1, ind[end]+1
        T_new[ind] .= range(T_new[lo], T_new[hi]; length=length(ind)+2)[2:end-1]
    end

    return T_new
end

"""
    collapse_displacement(zv; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)

Build the grid-point displacement field used by `collapse_advection` to close an
erupted band: elastic decay away from the band edges (`crack_perp_displacement`,
radius `R`), rescaled so the nearest exterior point on each side fully closes its
side of the gap, with the interior (band) points packed into a tiny strictly ordered
window around `Erupt_z0`.

`method=:elastic`: both domain boundaries are pinned - the closure is fully absorbed
by stretching the walls. `method=:hybrid`: the floor side is unchanged (elastic rise
toward the vent), but the roof displacement transitions from `-half` at the wall face
to a rigid `-Erupt_thick` subsidence far above (transition radius `Erupt_thick`, wide
enough to keep the warped grid monotonic), so the free surface sinks by the erupted
thickness and the removed volume exits through the unpinned top boundary. The
compression paying for the floor-side stretch is thereby concentrated in the roof
just above the vent.
"""
function collapse_displacement(zv; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)
    method in (:elastic, :hybrid) ||
        throw(ArgumentError("collapse_displacement supports only :elastic and :hybrid, got $method"))
    half = Erupt_thick/2
    z_shift = zv .- Erupt_z0
    Displ   = zero(z_shift)

    id_above  = findall(z_shift .> half)
    id_below  = findall(z_shift .< -half)
    id_inside = findall(abs.(z_shift) .<= half)

    dist_above = z_shift[id_above]  .- half   # distance outward from the idealized band edge
    dist_below = -z_shift[id_below] .- half

    # anchor distance: how far the nearest grid point on each side actually needs to move
    # to fully reach Erupt_z0, used to rescale the decay law so that point's gap fully closes
    anchor_above = isempty(id_above) ? half : z_shift[id_above[1]]
    anchor_below = isempty(id_below) ? half : -z_shift[id_below[end]]
    scale_above = anchor_above / crack_perp_displacement(0.0, half; r=R)
    scale_below = anchor_below / crack_perp_displacement(0.0, half; r=R)

    if method == :hybrid
        # roof: from -half at the face (walls meet at Erupt_z0) to -Erupt_thick far
        # above (rigid caldera subsidence of the whole overburden)
        Displ[id_above] .= -(Erupt_thick .- crack_perp_displacement(dist_above, half; r=Erupt_thick))
    else
        Displ[id_above] .= -scale_above .* crack_perp_displacement(dist_above, half; r=R)
    end
    Displ[id_below] .=  scale_below .* crack_perp_displacement(dist_below, half; r=R)

    # every point that ends up inside (or right at the edge of) the old band's footprint -
    # both the exterior anchor points (which land exactly on Erupt_z0) and the interior
    # (melt-zone) points - gets packed into a strictly ordered, tiny window around Erupt_z0,
    # so the displaced grid stays strictly monotonic everywhere (required by the
    # interpolation) without any two points colliding on the exact same position
    id_band = sort(vcat(id_inside, isempty(id_above) ? Int[] : id_above[1], isempty(id_below) ? Int[] : id_below[end]))
    unique!(id_band)
    if !isempty(id_band)
        ε = 1e-3*minimum(diff(zv))
        order = sortperm(z_shift[id_band])   # ascending z_shift == ascending zv on id_band
        targets = length(id_band) == 1 ? [0.0] : collect(range(-ε, ε; length=length(id_band)))
        Displ[id_band[order]] .= targets .- z_shift[id_band[order]]
    end

    # pin the bottom boundary (fixed BC) so the warped grid never contracts past it;
    # the top is pinned only for :elastic - for :hybrid the surface subsides and the
    # vacated cells are filled by extrapolation in collapse_advection
    Displ[1] = 0.0
    if method != :hybrid
        Displ[end] = 0.0
    end

    return Displ
end

"""
    T_new, rock_new, h_out = erupt_melt!(T, rocks, z; Erupt_z0, Erupt_thick, method=:caldera)

Erupt a melt region of thickness `Erupt_thick` [m] centered at `Erupt_z0` [m], the
inverse of `insert_sill`. `h_out` is the injected-magma content [m] the closure takes out of
the column - the erupted band's own content plus anything the closure displaces past a
domain boundary - and is the withdrawal term of the magma-volume budget
([`MassBudget`](@ref)). Two closure mechanisms are available via `method`:

- `:caldera` (default) - roof subsidence: the erupted band's cells are deleted (the
  magma and its heat leave the system) and the entire roof block above the band drops
  rigidly by the band thickness onto the chamber floor, so the free surface subsides
  by the erupted thickness. The vacated cells at the top are filled with the surface
  temperature (the Dirichlet boundary value) - the caldera floor. No rock parcel
  changes temperature: the roof block's profile is translated, not deformed, so the
  column's total heat drops by exactly the erupted band's content (minus the
  negligible cold surface fill) without any artificial cooling around the vent.
  `rocks` subsides the same way, so its integrated content drops by the erupted band's
  intruded-magma content.

- `:elastic` - elastic collapse: host rock on both sides moves toward the vent with
  `collapse_advection`'s elastic decay law (largest at the band edges, decaying with
  distance) and is re-interpolated onto the regular grid; `rocks` follows the same
  displacement (re-binarized by rounding). Temperatures are transported as intensive
  values, so the stretched walls re-cover the vent at their own temperature - the
  column's total heat drops by much less than the erupted band's content, and
  the integrated `rocks` content drops by less than the band's intruded-magma content.

- `:hybrid` - elastic + caldera: the floor rises elastically toward the vent as in
  `:elastic`, while the roof face drops to meet it and the roof displacement
  transitions to a rigid `-Erupt_thick` subsidence away from the vent, so the free
  surface sinks by the erupted thickness and the removed volume exits through the
  top. Deformation stays concentrated near the vent, temperatures are transported as
  intensive values (no dilution cooling), and the heat debit is approximately the
  erupted band's content (exact when the near-vent material is locally uniform: the
  floor-side stretch duplication is paid by compression of the equally hot roof just
  above the vent).
"""
function erupt_melt!(T, rocks, z; Erupt_z0, Erupt_thick, method=:caldera)
    method in (:caldera, :elastic, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    half = Erupt_thick/2
    zv = Vector(z)
    n  = length(zv)
    magma_out(rock_new) = nonnegative_debit(integrated_content(rocks, zv),
        integrated_content(rock_new, zv), "injected-magma content")

    if method == :elastic || method == :hybrid
        T_new    = collapse_advection(Vector(T), zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=method)
        # advect the grey phase indicator conservatively with the SAME displacement
        # collapse_advection applies to T, not round(collapse_advection(rocks)): the latter
        # interpolates-then-rounds the 0/1 field under the converging closure, which inflates
        # the grey where the walls stretch (e.g. 700 m -> 1700 m) or annihilates it near the
        # vent (700 m -> 0), and lands it off the co-moving temperature. The grey inside the
        # erupted band leaves with the eruption (zeroed before the remap, else collapse_
        # displacement packs it into one cell as a spike); the surviving grey is remapped
        # conservatively and co-moves with T. Field stays fractional, as after insert_sill.
        Displ    = collapse_displacement(zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=method)
        src      = copy(Vector(rocks))
        src[findall(abs.(zv .- Erupt_z0) .<= half)] .= 0.0   # erupted band grey leaves the system
        rock_new = conservative_advection(src, Displ, zv)
        return T_new, rock_new, magma_out(rock_new)
    end

    T_new    = copy(Vector(T))
    rock_new = copy(Vector(rocks))

    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    isempty(ind) && return T_new, rock_new, 0.0

    n_band = length(ind)
    i0 = ind[1]
    T_new[i0:n-n_band]    .= T_new[i0+n_band:n]      # roof block drops onto the floor
    rock_new[i0:n-n_band] .= rock_new[i0+n_band:n]
    T_new[n-n_band+1:n]    .= T_new[n]               # subsided surface, filled at Ttop
    rock_new[n-n_band+1:n] .= 0.0

    return T_new, rock_new, magma_out(rock_new)
end

"""
    melt_thickness(ϕ, z, z_lo, z_hi)

Melt content `∫ϕ dz` [m] of the depth interval `[z_lo, z_hi]` - the dense-rock
equivalent thickness of the melt held in that band. This is what actually leaves the
column when the band erupts: the crystal framework stays, so the vent closure
amplitude and the erupted volume (`A_sill * melt_thickness`) are based on the melt
content rather than on the bulk band thickness.
"""
function melt_thickness(ϕ, z, z_lo, z_hi)
    return integrated_content(ϕ, z, z_lo, z_hi)
end

"""
    column_enthalpy(T, z, MatParam, Phases) -> H  [J/m²]

Total heat content of the column per unit area, `∫ ρ(cₚT + Lϕ) dz`: sensible heat plus
the latent heat stored in the melt fraction. Properties are (re)evaluated from the
passed `T` so a before/after pair is self-consistent.

This is the diagnostic for the elastic/hybrid eruption closures: `:caldera` translates
the roof rigidly, so `H` drops by exactly the erupted band's content, but `:elastic`
transports `T` *intensively* (stretched walls re-cover the vent at their own temperature)
so it conserves energy only approximately. Measure `H` before and after `erupt_melt!` to
quantify that drift before trusting reservoir Tt-paths / zircon spectra from those methods.
"""
function column_enthalpy(T, z, MatParam, Phases)
    args = (T = T .+ 273.15,)
    ρ    = similar(T); Cp = similar(T); Hl = similar(T); ϕ = similar(T)
    compute_density!(ρ, MatParam, Phases, args)
    compute_heatcapacity!(Cp, MatParam, Phases, args)
    compute_latent_heat!(Hl, MatParam, Phases, args)
    compute_meltfraction!(ϕ, MatParam, Phases, args)
    return integrated_content(ρ .* (Cp .* T .+ Hl .* ϕ), z)
end

"""
    conductive_boundary_energy(T, k, z, Δt) -> E [J/m²]

Net conductive heat entering the column during one timestep. This is the exact
boundary term obtained by summing the finite-difference diffusion operator over the
interior nodes; positive values heat the column.
"""
function conductive_boundary_energy(T, k, z, Δt)
    length(T) == length(z) || throw(DimensionMismatch("T and z must have equal length"))
    length(k) == length(z) - 1 || throw(DimensionMismatch("k must have length(z) - 1 entries"))
    Δt >= 0 || throw(ArgumentError("Δt must be nonnegative"))
    Δz = z[2] - z[1]
    Δz > 0 || throw(ArgumentError("z must be strictly ascending"))
    return (k[end] * (T[end] - T[end-1]) - k[1] * (T[2] - T[1])) / Δz * Δt
end

"""
    source_energy(Q, z, Δt) -> E [J/m²]

Heat supplied by the volumetric source during one timestep, using the same interior
nodes as the thermal residual.
"""
function source_energy(Q, z, Δt)
    length(Q) == length(z) || throw(DimensionMismatch("Q and z must have equal length"))
    Δt >= 0 || throw(ArgumentError("Δt must be nonnegative"))
    return sum(@view Q[2:end-1]) * (z[2] - z[1]) * Δt
end

"""
    magma_heat_input(T_host, Tsill, h, MatParam; phase=0) -> E [J/m²]

Sensible plus latent heat supplied by `h` meters of initially liquid magma relative
to host material at `T_host`. This is the discrete-sill counterpart of the integrand
used by [`compute_Q_magma!`](@ref), and shares its density approximation: the magma is
weighted by the host-rock thermal density of `MatParam` at `T_host`, not by
`EruptionParams.ρ_melt`.
"""
function magma_heat_input(T_host, Tsill, h, MatParam; phase=0)
    h >= 0 || throw(ArgumentError("h must be nonnegative"))
    T = [float(T_host)]
    phases = [phase]
    args = (T = T .+ 273.15,)
    ρ = similar(T); Cp = similar(T); Hl = similar(T); ϕ = similar(T)
    compute_density!(ρ, MatParam, phases, args)
    compute_heatcapacity!(Cp, MatParam, phases, args)
    compute_latent_heat!(Hl, MatParam, phases, args)
    compute_meltfraction!(ϕ, MatParam, phases, args)
    return h * ρ[1] * (Cp[1] * (Tsill - T_host) + Hl[1] * (1 - ϕ[1]))
end

"""
    erupted_melt_enthalpy(T, ϕ, z, z_lo, z_hi, h, MatParam, Phases) -> E [J/m²]

Enthalpy carried by `h` meters of melt sampled from the eruptible interval. Local
enthalpy is weighted by melt content `ϕΔz`; the liquid carries sensible heat and the
full latent heat of fusion. Temperature uses the same 0 °C reference as
[`column_enthalpy`](@ref).
"""
function erupted_melt_enthalpy(T, ϕ, z, z_lo, z_hi, h, MatParam, Phases)
    h >= 0 || throw(ArgumentError("h must be nonnegative"))
    length(T) == length(ϕ) == length(z) == length(Phases) ||
        throw(DimensionMismatch("T, ϕ, z, and Phases must have equal length"))
    ind = findall(z_lo .<= z .<= z_hi)
    isempty(ind) && return 0.0
    edges = grid_cell_edges(z)
    geometric_weights = [max(0.0, min(edges[i + 1], z_hi) - max(edges[i], z_lo)) for i in ind]
    weights = ϕ[ind] .* geometric_weights
    total = sum(weights)
    total > 0 || return 0.0
    args = (T = T .+ 273.15,)
    ρ = similar(T); Cp = similar(T); Hl = similar(T)
    compute_density!(ρ, MatParam, Phases, args)
    compute_heatcapacity!(Cp, MatParam, Phases, args)
    compute_latent_heat!(Hl, MatParam, Phases, args)
    return h * sum(weights .* ρ[ind] .* (Cp[ind] .* T[ind] .+ Hl[ind])) / total
end

"""
    EnthalpyBudget(initial_storage)

Cumulative column-energy diagnostic per unit area. `residual` is

`storage - initial_storage - boundary - injected - source + erupted`.

It deliberately exposes heat created or retained by intensive temperature remapping;
it does not correct transport.
"""
Base.@kwdef mutable struct EnthalpyBudget
    initial_storage :: Float64
    storage         :: Float64
    boundary        :: Float64 = 0.0
    injected        :: Float64 = 0.0
    source          :: Float64 = 0.0
    erupted         :: Float64 = 0.0
    residual        :: Float64 = 0.0
end

EnthalpyBudget(initial_storage) = EnthalpyBudget(; initial_storage, storage=initial_storage)

"""
    update_enthalpy_budget!(budget, storage; boundary=0, injected=0, source=0, erupted=0)

Record the current column storage and accumulate the energy exchanged during one timestep,
refreshing `budget.residual` (see [`EnthalpyBudget`](@ref)).
"""
function update_enthalpy_budget!(budget::EnthalpyBudget, storage;
                                 boundary=0.0, injected=0.0, source=0.0, erupted=0.0)
    all(isfinite, (storage, boundary, injected, source, erupted)) ||
        throw(ArgumentError("enthalpy-budget terms must be finite"))
    budget.storage = storage
    budget.boundary += boundary
    budget.injected += injected
    budget.source += source
    budget.erupted += erupted
    budget.residual = storage - budget.initial_storage - budget.boundary -
                      budget.injected - budget.source + budget.erupted
    return budget
end

enthalpy_budget_snapshot(budget::EnthalpyBudget) = (;
    storage=budget.storage,
    storage_change=budget.storage - budget.initial_storage,
    boundary=budget.boundary,
    injected=budget.injected,
    source=budget.source,
    erupted=budget.erupted,
    residual=budget.residual,
)

"""
    MassBudget(magma, melt)

Cumulative per-unit-area thickness budget [m] of the two quantities repeated injection
supplies: intruded magma and stored melt. They share the injected term but close
differently, and only the first one closes to zero.

- `residual = injected - withdrawn - (magma - initial_magma)` is the **magma-volume
  budget**. `magma` is the injected-rock control-volume content and `withdrawn` is the
  magma removed by eruption or displaced past a domain boundary. Exact additive injection
  makes this balance close up to remapping roundoff.
- `melt_residual = injected - erupted - (melt - initial_melt)` is the **melt-content
  residual**. It is not a crystallization measurement: it combines crystallization,
  host-rock melting, boundary transport, and any mismatch between booked and physically
  represented melt withdrawal.

Like [`EnthalpyBudget`](@ref) this diagnoses transport and never corrects it.
"""
Base.@kwdef mutable struct MassBudget
    initial_magma :: Float64
    initial_melt  :: Float64
    magma         :: Float64
    melt          :: Float64
    injected      :: Float64 = 0.0
    withdrawn     :: Float64 = 0.0
    erupted       :: Float64 = 0.0
    residual      :: Float64 = 0.0
    melt_residual :: Float64 = 0.0
end

MassBudget(magma, melt) = MassBudget(; initial_magma=magma, initial_melt=melt, magma, melt)

"""
    update_mass_budget!(budget, magma, melt; injected=0, withdrawn=0, erupted=0)

Record the current intruded-magma content `magma` [m] and stored melt `melt` [m] and
accumulate the thicknesses exchanged during one timestep, refreshing `budget.residual` and
`budget.melt_residual` (see [`MassBudget`](@ref)).
"""
function update_mass_budget!(budget::MassBudget, magma, melt;
                             injected=0.0, withdrawn=0.0, erupted=0.0)
    all(isfinite, (magma, melt, injected, withdrawn, erupted)) ||
        throw(ArgumentError("mass-budget terms must be finite"))
    all(>=(0), (magma, melt, injected, withdrawn, erupted)) ||
        throw(ArgumentError("mass-budget terms must be nonnegative"))
    budget.magma = magma
    budget.melt = melt
    budget.injected += injected
    budget.withdrawn += withdrawn
    budget.erupted += erupted
    budget.residual = budget.injected - budget.withdrawn - (magma - budget.initial_magma)
    budget.melt_residual = budget.injected - budget.erupted - (melt - budget.initial_melt)
    return budget
end

mass_budget_snapshot(budget::MassBudget) = (;
    magma=budget.magma,
    melt=budget.melt,
    magma_change=budget.magma - budget.initial_magma,
    melt_change=budget.melt - budget.initial_melt,
    injected=budget.injected,
    withdrawn=budget.withdrawn,
    erupted=budget.erupted,
    residual=budget.residual,
    melt_residual=budget.melt_residual,
)

"""
    erupt_displacement(zm, Erupt_z0, Erupt_thick; method=:caldera, R=Erupt_thick/2)

New position of a material point at depth `zm` under the eruption closure of
`erupt_melt!`, for markers and tracers riding on the host rock. `:caldera`: points
above the erupted band drop rigidly by `Erupt_thick`, points inside land on the
chamber floor, points below stay put. `:elastic`: points inside the band collapse
onto `Erupt_z0`, points outside move toward it with the same elastic decay law as
the host rock in `collapse_advection`. `:hybrid`: like `:elastic` below the band,
while above it the drop transitions from `Erupt_thick/2` at the wall face to the
full rigid `Erupt_thick` subsidence far above the vent.
"""
function erupt_displacement(zm, Erupt_z0, Erupt_thick; method=:caldera, R=Erupt_thick/2)
    method in (:caldera, :elastic, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    half = Erupt_thick/2
    s = zm - Erupt_z0
    if method == :elastic
        abs(s) <= half && return Erupt_z0
        return zm - sign(s)*crack_perp_displacement(abs(s) - half, half; r=R)
    elseif method == :hybrid
        abs(s) <= half && return Erupt_z0
        s < -half && return zm + crack_perp_displacement(-s - half, half; r=R)
        return zm - (Erupt_thick - crack_perp_displacement(s - half, half; r=Erupt_thick))
    end
    zm > Erupt_z0 + half && return zm - Erupt_thick
    zm >= Erupt_z0 - half && return Erupt_z0 - half
    return zm
end

"""
    collapse_markers!(markers, Erupt_z0, Erupt_thick; method=:caldera)

Move marker positions (e.g. the Q_magma injection-zone boundary markers drawn in the
melt-fraction plot) with the eruption closure, consistent with `erupt_melt!` (see
`erupt_displacement` for the two `method`s).
"""
function collapse_markers!(markers, Erupt_z0, Erupt_thick; method=:caldera)
    for (i, zm) in enumerate(markers)
        markers[i] = erupt_displacement(zm, Erupt_z0, Erupt_thick; method=method)
    end
    return markers
end

"""
    collapse_tracers!(tracers, Erupt_z0, Erupt_thick; method=:caldera)

Move the passive tracers with the eruption closure, consistent with `erupt_melt!`
(see `erupt_displacement` for the two `method`s). Cargo tracers selected by
`extract_erupted_tracers!` are removed first; unselected tracers inside the collapsing
band are placed on the closed vent.
"""
function collapse_tracers!(tracers, Erupt_z0, Erupt_thick; method=:caldera)
    for tracer in tracers
        tracer.z = erupt_displacement(tracer.z, Erupt_z0, Erupt_thick; method=method)
    end
    return tracers
end

"""
    extract_erupted_tracers!(rng, tracers, ϕ, z, z_lo, z_hi, h_erupt;
                             eligible_phase)

Remove a melt-representative sample of tracers from the full eruptible mush
`[z_lo, z_hi]`. An eligible tracer in grid cell `i` represents melt thickness
`ϕ[i]w[i]/n[i]`, where `w[i]` is that control volume's overlap with the interval and
`n[i]` is the number of eligible tracers in that cell, so
changing the tracer-seeding density does not change the represented cargo. Sampling
is without replacement and weighted by represented melt thickness until `h_erupt`
is approximated.

Pass `eligible_phase=nothing` to include host-rock and injected tracers, or a phase
number to restrict cargo to that phase. The keyword is required so phase eligibility
cannot be ignored accidentally. Returns `(erupted, h_cargo)`, where `h_cargo` is the
melt thickness represented by the selected tracers. The caller owns `rng`; stochastic
tests should pass a seeded generator. When all phases are eligible, the tracer weights
are normalized to the full melt content of the interval so empty tracer cells do not
make the sampled cargo underrepresent the withdrawn state.
"""
function extract_erupted_tracers!(rng, tracers, ϕ, z, z_lo, z_hi, h_erupt; eligible_phase)
    h_erupt >= 0 || throw(ArgumentError("h_erupt must be nonnegative"))
    length(ϕ) == length(z) || throw(DimensionMismatch("ϕ and z must have equal length"))
    length(z) >= 2 || throw(ArgumentError("z must contain at least two grid points"))

    Δz = z[2] - z[1]
    Δz > 0 || throw(ArgumentError("z must be strictly ascending"))
    candidate_indices = Int[]
    cell_indices = Int[]
    counts = zeros(Int, length(z))
    for (j, tracer) in pairs(tracers)
        z_lo <= tracer.z <= z_hi || continue
        (isnothing(eligible_phase) || tracer.phase == eligible_phase) || continue
        i = clamp(round(Int, (tracer.z - z[1]) / Δz) + 1, 1, length(z))
        ϕ[i] > 0 || continue
        push!(candidate_indices, j)
        push!(cell_indices, i)
        counts[i] += 1
    end

    isempty(candidate_indices) && return tracers[Int[]], 0.0
    widths = interval_cell_widths(z, z_lo, z_hi)
    weights = [ϕ[i] * widths[i] / counts[i] for i in cell_indices]
    if isnothing(eligible_phase)
        weights .*= melt_thickness(ϕ, z, z_lo, z_hi) / sum(weights)
    end
    total = sum(weights)
    target = min(h_erupt, total)
    target > 0 || return tracers[Int[]], 0.0

    order = target == total ? eachindex(weights) : sortperm(randexp(rng, length(weights)) ./ weights)
    selected = Int[]
    h_cargo = 0.0
    last_weight = 0.0
    for k in order
        push!(selected, candidate_indices[k])
        last_weight = weights[k]
        h_cargo += last_weight
        h_cargo >= target && break
    end
    if length(selected) > 1
        if abs(h_cargo - last_weight - target) < abs(h_cargo - target)
            pop!(selected)
            h_cargo -= last_weight
        end
    end

    is_erupted = falses(length(tracers))
    is_erupted[selected] .= true
    erupted = tracers[is_erupted]
    deleteat!(tracers, is_erupted)
    return erupted, h_cargo
end

"""
    EruptionEvent(; ...)

One realized eruption and its four independently supplied thickness accounts. Separate
trigger and realization times preserve the delay when sub-grid D&H drainage aggregates.
Construction fails unless requested, state-withdrawn, cargo-represented, and booked
thickness agree. State and booking must agree to floating-point precision; cargo may
differ by the explicitly declared tracer-resolution tolerance `cargo_atol`.

`magma_removed` is the intruded-magma content the closure took out of the column.
`melt_removed` is the measured change in full-column melt storage across the instantaneous
closure. It can differ from `requested` because the fixed-grid closure transports
temperature intensively; this diagnostic exposes that mismatch instead of relabeling the
requested closure amplitude as a measured state withdrawal.
"""
struct EruptionEvent
    trigger_time        :: Float64
    realization_time    :: Float64
    requested           :: Float64
    state_withdrawn     :: Float64
    cargo_represented   :: Float64
    booked              :: Float64
    z_lo                :: Float64
    z_hi                :: Float64
    z_centroid          :: Float64
    trigger             :: Symbol
    closure             :: Symbol
    aggregated          :: Bool
    cargo_count         :: Int
    enthalpy_before     :: Float64
    enthalpy_after      :: Float64
    erupted_enthalpy    :: Float64
    enthalpy_residual   :: Float64
    magma_removed       :: Float64
    melt_removed        :: Float64
end

function EruptionEvent(; trigger_time, realization_time, requested, state_withdrawn,
                       cargo_represented, booked,
                       z_lo, z_hi, z_centroid, trigger, closure, aggregated=false,
                       cargo_count=0, enthalpy_before, enthalpy_after, erupted_enthalpy,
                       magma_removed=0.0, melt_removed=0.0, cargo_atol=0.0)
    thicknesses = (requested, state_withdrawn, cargo_represented, booked)
    all(isfinite, thicknesses) || throw(ArgumentError("eruption thicknesses must be finite"))
    all(x -> x >= 0, thicknesses) ||
        throw(ArgumentError("eruption thicknesses must be nonnegative"))
    cargo_atol >= 0 || throw(ArgumentError("cargo_atol must be nonnegative"))
    all(isfinite, (trigger_time, realization_time, z_lo, z_hi, z_centroid,
                   enthalpy_before, enthalpy_after, erupted_enthalpy,
                   magma_removed, melt_removed)) ||
        throw(ArgumentError("eruption event values must be finite"))
    magma_removed >= 0 || throw(ArgumentError("magma_removed must be nonnegative"))
    cargo_count >= 0 || throw(ArgumentError("cargo_count must be nonnegative"))
    scale = max(maximum(thicknesses), 1.0)
    atol = 64eps(scale)
    isapprox(state_withdrawn, requested; rtol=0, atol) &&
    isapprox(booked, requested; rtol=0, atol) &&
    isapprox(cargo_represented, requested; rtol=0, atol=cargo_atol + atol) ||
        throw(ArgumentError("eruption thickness mismatch: requested=$requested, " *
            "state-withdrawn=$state_withdrawn, cargo-represented=$cargo_represented, " *
            "booked=$booked (cargo tolerance=$cargo_atol)"))
    closure in (:caldera, :elastic, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $closure"))
    Hres = enthalpy_after - enthalpy_before + erupted_enthalpy
    trigger_time <= realization_time ||
        throw(ArgumentError("trigger_time must not exceed realization_time"))
    return EruptionEvent(trigger_time, realization_time, requested, state_withdrawn,
        cargo_represented, booked, z_lo, z_hi, z_centroid, Symbol(trigger), closure,
        aggregated, cargo_count, enthalpy_before, enthalpy_after, erupted_enthalpy, Hres,
        magma_removed, melt_removed)
end

"""
    realize_eruption!(rng, T, rocks, tracers, ϕ, z, MatParam, Phases; ...) ->
        (T_new, rocks_new, cargo, event)

Withdraw the thermal/rock state and represented tracer cargo as one operation, then
construct the fail-fast [`EruptionEvent`](@ref). The default cargo tolerance is half a
grid cell, the sampler's maximum nearest-whole-tracer rounding error when the eligible
population represents the full mush.
"""
function realize_eruption!(rng, T, rocks, tracers, ϕ, z, MatParam, Phases;
                           realization_time, trigger_time=realization_time,
                           h_requested, h_booked=h_requested, z_lo, z_hi,
                           trigger, closure, aggregated=false, eligible_phase,
                           cargo_atol=(z[2] - z[1]) / 2)
    h_requested > 0 || throw(ArgumentError("h_requested must be positive"))
    z_hi > z_lo || throw(ArgumentError("z_hi must be greater than z_lo"))
    widths = interval_cell_widths(z, z_lo, z_hi)
    weights = ϕ .* widths
    ind = findall(z_lo .<= z .<= z_hi)
    h_melt = sum(weights[ind])
    h_melt > 0 || throw(ArgumentError("eruptible interval contains no melt"))
    z_centroid = sum(weights[ind] .* z[ind]) / h_melt
    H_before = column_enthalpy(T, z, MatParam, Phases)
    H_erupted = erupted_melt_enthalpy(T, ϕ, z, z_lo, z_hi, h_requested, MatParam, Phases)
    T_new, rocks_new, magma_removed = erupt_melt!(T, rocks, z;
        Erupt_z0=(z_lo + z_hi) / 2, Erupt_thick=h_requested, method=closure)
    cargo, h_cargo = extract_erupted_tracers!(rng, tracers, ϕ, z, z_lo, z_hi,
        h_requested; eligible_phase)
    H_after = column_enthalpy(T_new, z, MatParam, Phases)
    ϕ_after = similar(ϕ)
    compute_meltfraction!(ϕ_after, MatParam, Phases, (T = T_new .+ 273.15,))
    melt_removed = integrated_content(ϕ, z) - integrated_content(ϕ_after, z)
    event = EruptionEvent(; trigger_time, realization_time, requested=h_requested,
        state_withdrawn=h_requested, cargo_represented=h_cargo, booked=h_booked,
        z_lo, z_hi, z_centroid, trigger, closure, aggregated,
        cargo_count=length(cargo), enthalpy_before=H_before, enthalpy_after=H_after,
        erupted_enthalpy=H_erupted, magma_removed, melt_removed, cargo_atol)
    return T_new, rocks_new, cargo, event
end

# =====================================================================================
#  OVERPRESSURE TRIGGER  (Degruyter & Huber 2014 / Townsend 2021, 3-phase)
# =====================================================================================
# A physically richer alternative to the GUI's single-β_eff elastic box model: a lumped
# 3-phase (melt/crystal/gas) chamber whose overpressure ΔP = P - P_lith evolves from a
# master ODE driven by the column's own Ṫ and ϕ̇. Gas exsolution/water solubility set the
# compressibility, and a depth cutoff stalls dikes from chambers deeper than z_erupt_max.
# This is a *trigger*: the actual erupted band is removed with erupt_melt! as elsewhere.

"""
    EruptionParams

Tunable parameters for the 3-phase overpressure trigger. Thermodynamic / solubility
constants follow Degruyter & Huber (2014); treat the defaults as order-of-magnitude
values and check the paper's Table 1 for a specific system.

- `ϕ_erupt`  : mobile-melt threshold; the eruptible chamber is the mush ϕ ≥ ϕ_erupt (0.5)
- `ΔP_crit`  : roof-failure overpressure [Pa] (D&H: ~10-40 MPa)
- `ϕ_g_crit` : gas-volume-fraction lock-up (shut-off) [-]
- `ΔP_relax` : overpressure left after an eruption relaxes the chamber [Pa]
- `z_erupt_max`: max chamber-centroid depth [m] at which eruptions can still occur
- `β_r`,`η_r`: host-rock stiffness [Pa] and wall relaxation viscosity [Pa·s]
- `ρ_melt`,`ρ_x`: melt / crystal densities [kg/m³]
- `m_w`      : total water mass fraction of the magma [-]
- `z_gas_max`: maximum depth [m] of the shallow zone in which an exsolved gas phase is
  represented; deeper magma retains its water in the condensed phase and never calls RK
- `A_visc`,`B_gas`,`G_act`: Arrhenius wall-relaxation-viscosity constants (D&H 2014 Table 1),
  `η(T) = A_visc·exp(G_act/(B_gas·T))`; used to compute `η_r` from the crustal wall T.
- `ρ_crust`,`g`: crustal density [kg/m³] and gravity [m/s²] for the lithostatic reference
"""
Base.@kwdef mutable struct EruptionParams
    ϕ_erupt  :: Float64 = 0.5
    ΔP_crit  :: Float64 = 20e6
    ϕ_g_crit :: Float64 = 0.5
    ΔP_relax :: Float64 = 0.0
    z_erupt_max :: Float64 = 10e3
    β_r      :: Float64 = 1e10
    η_r      :: Float64 = 1e19
    ρ_melt   :: Float64 = 2400.0
    ρ_x      :: Float64 = 2700.0
    m_w      :: Float64 = 0.05
    z_gas_max :: Float64 = 10e3
    A_visc   :: Float64 = 4.25e7        # D&H Table 1: viscosity-law prefactor [Pa·s]
    B_gas    :: Float64 = 8.314         # molar gas constant [J/mol/K]
    G_act    :: Float64 = 141e3         # activation energy for creep [J/mol]
    ρ_crust  :: Float64 = 2700.0
    g        :: Float64 = 9.81
end

function validate_eruption_params(ep::EruptionParams)
    all(isfinite, (ep.ϕ_erupt, ep.ΔP_crit, ep.ϕ_g_crit, ep.ΔP_relax,
        ep.z_erupt_max, ep.β_r, ep.η_r, ep.ρ_melt, ep.ρ_x, ep.m_w,
        ep.z_gas_max, ep.A_visc, ep.B_gas, ep.G_act, ep.ρ_crust, ep.g)) ||
        throw(ArgumentError("eruption parameters must be finite"))
    0 <= ep.ϕ_erupt <= 1 || throw(DomainError(ep.ϕ_erupt, "ϕ_erupt must lie in [0, 1]"))
    0 <= ep.ϕ_g_crit <= 1 || throw(DomainError(ep.ϕ_g_crit, "ϕ_g_crit must lie in [0, 1]"))
    ep.ΔP_crit > 0 || throw(ArgumentError("ΔP_crit must be positive"))
    0 <= ep.ΔP_relax < ep.ΔP_crit ||
        throw(ArgumentError("ΔP_relax must lie in [0, ΔP_crit)"))
    ep.z_erupt_max >= 0 && ep.z_gas_max >= 0 ||
        throw(ArgumentError("depth limits must be nonnegative"))
    all(>(0), (ep.β_r, ep.η_r, ep.ρ_melt, ep.ρ_x, ep.A_visc,
               ep.B_gas, ep.ρ_crust, ep.g)) ||
        throw(ArgumentError("mechanical, density, and viscosity parameters must be positive"))
    ep.G_act >= 0 || throw(ArgumentError("G_act must be nonnegative"))
    0 <= ep.m_w <= 1 || throw(DomainError(ep.m_w, "m_w must lie in [0, 1]"))
    return ep
end

"""
    check_density_consistency(MatParam, ep::EruptionParams; T_ref=800.0, rtol=1e-3, phase=0)

Verify that the host-rock thermal density of `MatParam` (evaluated at `T_ref` [°C])
agrees with the lithostatic density `ep.ρ_crust`. Chamber crystals and melt are distinct
materials, so `ep.ρ_x` and `ep.ρ_melt` are deliberately not compared.

Call once at model setup, where the two parameter objects are built.
"""
function check_density_consistency(MatParam, ep::EruptionParams; T_ref=800.0, rtol=1e-3, phase=0)
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    ρ = zeros(1)
    compute_density!(ρ, MatParam, [phase], (T = [T_ref + 273.15],))
    isapprox(ρ[1], ep.ρ_crust; rtol) || throw(ArgumentError(
        "host-rock density mismatch: MatParam gives $(ρ[1]) kg/m³ at $T_ref °C, " *
        "EruptionParams.ρ_crust is $(ep.ρ_crust) kg/m³ (rtol=$rtol)"))
    return ρ[1]
end

"""
    EruptionState

Mutable chamber state carried across timesteps: absolute pressure `P`, lithostatic
reference `P_lith`, gas volume fraction `ϕ_g`, and the previous mush (T,ϕ) for the
fixed-P density-rate source term. `h_erupt` is the melt drained by the pressure ODE
during the latest thermal step; `h_pending` is drained melt not yet withdrawn from the
grid because the accumulated thickness is still sub-grid.
"""
Base.@kwdef mutable struct EruptionState
    P        :: Float64 = 0.0
    P_lith   :: Float64 = 0.0
    ϕ_g      :: Float64 = 0.0
    T_prev   :: Float64 = NaN          # mush-mean T [K] at previous step
    ϕ_prev   :: Float64 = NaN          # mush-mean ϕ at previous step
    inv_βm   :: Float64 = 0.0          # magma compressibility 1/β_m at the last step [1/Pa]
    h_erupt  :: Float64 = 0.0          # melt thickness [m] drained by eruptions during the last step
    h_pending :: Float64 = 0.0          # drained melt [m] awaiting a grid-resolvable withdrawal
    pending_since :: Float64 = NaN      # time [s] of the first trigger represented by h_pending
    m_diss   :: Float64 = 0.0          # dissolved H₂O mass fraction (per magma mass), Liu 2005
    X_g      :: Float64 = 0.0          # exsolved H₂O gas mass fraction (per magma mass), D&H eq.18
    ρ_gas    :: Float64 = 0.0          # exsolved-gas density [kg/m³] at the last step (modified RK)
    η_r      :: Float64 = 0.0          # wall relaxation viscosity used at the last step [Pa·s]
    ϕ_mush   :: Float64 = 0.0          # mush-mean melt fraction driving the last step
    init     :: Bool    = false
end

"""
    pending_withdrawal!(state, h_requested, h_melt, Δz; time) -> h_realizable

Queue `h_requested` melt thickness drained by the D&H pressure model. Return zero while
the queued withdrawal is at most `2Δz`; otherwise return the grid-resolvable thickness,
bounded by the chamber's current `h_melt`. The pending amount is not removed until
[`commit_pending_withdrawal!`](@ref) is called after the physical state withdrawal.
`time` is required when starting a new queue so the realized event retains its first
trigger time.
"""
function pending_withdrawal!(state::EruptionState, h_requested, h_melt, Δz; time=NaN)
    h_requested >= 0 || throw(ArgumentError("h_requested must be nonnegative"))
    h_melt >= 0 || throw(ArgumentError("h_melt must be nonnegative"))
    Δz > 0 || throw(ArgumentError("Δz must be positive"))
    if h_requested > 0 && state.h_pending == 0
        isfinite(time) || throw(ArgumentError("time must be finite when starting a pending withdrawal"))
        state.pending_since = time
    end
    state.h_pending += h_requested
    h_realizable = min(state.h_pending, h_melt)
    return h_realizable > 2Δz ? h_realizable : 0.0
end

"""
    commit_pending_withdrawal!(state, h_realized)

Debit a successfully completed physical withdrawal from `state.h_pending`.
"""
function commit_pending_withdrawal!(state::EruptionState, h_realized)
    0 <= h_realized <= state.h_pending ||
        throw(ArgumentError("h_realized must lie between zero and the pending withdrawal"))
    state.h_pending -= h_realized
    if state.h_pending <= 64eps(max(h_realized, 1.0))
        state.h_pending = 0.0
        state.pending_since = NaN
    end
    return state
end

"""
    rho_gas_RK(P, T_K) -> ρ_g [kg/m³]

Modified Redlich–Kwong parameterisation for H₂O gas density (Huber et al. 2010, D&H eq.
A.1), valid for 873.15–1173.15 K and 30–400 MPa. Calls outside that calibration box
throw instead of extrapolating. `P` is in Pa (used as bar internally), `T_K` in K (used
as °C).
"""
function rho_gas_RK(P, T_K)
    all(isfinite, (P, T_K)) || throw(ArgumentError("P and T_K must be finite"))
    30e6 <= P <= 400e6 || throw(DomainError(P,
        "modified RK H₂O EOS is calibrated only for 30–400 MPa"))
    873.15 <= T_K <= 1173.15 || throw(DomainError(T_K,
        "modified RK H₂O EOS is calibrated only for 873.15–1173.15 K"))
    Tc = T_K - 273.15
    Pb = P*1e-5                                # Pa -> bar
    return 1e3*(-112.528*Tc^-0.381 + 127.811*Pb^-1.135 + 112.04*Tc^-0.411*Pb^0.033)
end

"""
    water_gas_partition(P, T_K, ϕ_melt, ep; z_centroid) -> (m_diss, X_g, ρ_g, m_eq)

H₂O speciation + gas density at `P` [Pa], `T_K` [K], `ϕ_melt`. Dissolved water per *magma*
mass `m_diss = m_eq·ϕ_melt` from the Liu et al. (2005) saturation `m_eq` (per *melt* mass,
T-dependent, silicic, H₂O-only so `P_w=P`); exsolved-gas mass fraction `X_g = max(0, m_w −
m_diss)` (D&H 2014 eq. 18, water conserved); gas density `ρ_g` from the modified
Redlich–Kwong EOS ([`rho_gas_RK`](@ref)). Both [`mixture_density`](@ref) and the chamber
diagnostics on [`EruptionState`](@ref) read this, so the physics lives in one place.
CO₂ is not modelled (`X_co2≡0`); there is no CO₂ phase to track. Exsolved gas
is represented only where `abs(z_centroid) <= ep.z_gas_max`. Deeper chambers return
`X_g=ϕ_g=ρ_g=0`, keep `m_w` in the condensed phase, and do not evaluate either the
Liu law or RK EOS.
"""
function water_gas_partition(P, T_K, ϕ_melt, ep::EruptionParams; z_centroid=nothing)
    z_centroid === nothing && throw(ArgumentError(
        "z_centroid is required so gas physics cannot be applied silently in the lower crust"))
    all(isfinite, (P, T_K, ϕ_melt, z_centroid)) ||
        throw(ArgumentError("P, T_K, ϕ_melt, and z_centroid must be finite"))
    0 <= ϕ_melt <= 1 || throw(DomainError(ϕ_melt, "ϕ_melt must lie in [0, 1]"))
    0 <= ep.m_w <= 1 || throw(DomainError(ep.m_w, "m_w must lie in [0, 1]"))
    ep.z_gas_max >= 0 || throw(ArgumentError("z_gas_max must be nonnegative"))
    ep.m_w == 0 && return 0.0, 0.0, 0.0, NaN
    if abs(z_centroid) > ep.z_gas_max
        return ep.m_w, 0.0, 0.0, NaN
    end
    Pp = P
    # Liu et al. (2005): coefficients give wt%, the trailing 1e-2 → mass fraction (matches
    # reference exsolve_silicic.m). As the magma crystallizes (ϕ_melt↓) the melt dissolves
    # less, so X_g *rises* — second boiling, D&H's dominant pressurization for small chambers.
    Pm     = Pp*1e-6                          # Pa -> MPa
    # Liu 2005 is only calibrated to ~500 MPa; beyond that the -Pm^1.5 term drives the
    # saturation negative, which would make X_g blow past m_w (and corrupt ρ/ϕ_g). A
    # saturation can't be negative, and dissolved water can't exceed the total m_w — clamp
    # both, so m_diss and X_g stay in [0, m_w] (they are mass fractions).
    m_eq   = max(0.0, 1e-2*((354.94*sqrt(Pm) + 9.623*Pm - 1.5223*Pm^1.5)/T_K + 1.2439e-3*Pm^1.5))
    ρ_g    = rho_gas_RK(Pp, T_K)
    ρ_g > 0 || throw(DomainError(ρ_g, "RK gas density must be positive"))
    m_diss = min(m_eq*ϕ_melt, ep.m_w)         # dissolved H₂O per magma mass, ≤ total water
    X_g    = ep.m_w - m_diss                  # exsolved-gas mass fraction ∈ [0, m_w]
    return m_diss, X_g, ρ_g, m_eq
end

"""
    mixture_density(P, T_K, ϕ_melt, ep; z_centroid) -> (ρ, ϕ_g)

Three-phase (melt + crystal + exsolved gas) mixture density [kg/m³] and gas volume
fraction at pressure `P` [Pa], temperature `T_K` [K] and melt fraction `ϕ_melt`. The H₂O
speciation and gas density come from [`water_gas_partition`](@ref).
"""
function mixture_density(P, T_K, ϕ_melt, ep::EruptionParams; z_centroid=nothing)
    ρ_c   = ϕ_melt*ep.ρ_melt + (1-ϕ_melt)*ep.ρ_x   # condensed (melt+crystal) density
    z_centroid === nothing && throw(ArgumentError(
        "z_centroid is required so gas physics cannot be applied silently in the lower crust"))
    if abs(z_centroid) > ep.z_gas_max
        return ρ_c, 0.0
    end
    _, Xg, ρ_g, _ = water_gas_partition(P, T_K, ϕ_melt, ep; z_centroid)
    Vg    = Xg/ρ_g
    Vc    = (1-Xg)/ρ_c
    ρ     = 1.0/(Vg + Vc)
    ϕ_g   = Vg/(Vg + Vc)
    return ρ, ϕ_g
end

"""
    wall_relaxation_viscosity(ep, T_wall_K) -> η_r [Pa·s]

Arrhenius wall-relaxation viscosity at the chamber-wall temperature `T_wall_K` [K]
(D&H 2014 Appendix A.5, minimal wall-T tier): `η_r = A_visc·exp(G_act/(B_gas·T_wall))`,
clamped to `[1e17, 1e24]` so a cold wall can't freeze the overpressure ODE. As the crust
matures (wall heats) η_r drops and the chamber crosses from *storing* into *erupting* on
its own, removing the free GUI knob. This is the geometry-free tier; the full spherical
radial integral (D&H A.18–A.21) is deferred (1-D-Cartesian makes it only *effective*).
"""
function wall_relaxation_viscosity(ep::EruptionParams, T_wall_K)
    isfinite(T_wall_K) && T_wall_K > 0 ||
        throw(ArgumentError("T_wall_K must be finite and positive"))
    return clamp(ep.A_visc*exp(ep.G_act/(ep.B_gas*T_wall_K)), 1e17, 1e24)
end

"""
    eruptible_mush(ϕ, z; ϕ_erupt=0.5) -> (ind, V_e, z_centroid)

Indices of the largest contiguous mobile-mush chamber (`ϕ ≥ ϕ_erupt`), its bulk
thickness `V_e` [m per unit area], and its melt-weighted centroid depth. Disconnected
melt lenses are independent chambers and are never combined into one pressure reservoir.
"""
function eruptible_mush(ϕ, z; ϕ_erupt=0.5)
    length(ϕ) == length(z) || throw(DimensionMismatch("ϕ and z must have equal length"))
    run = largest_contiguous_range(ϕ .>= ϕ_erupt)
    run === nothing && return Int[], 0.0, 0.0
    ind = collect(run)
    z_lo, z_hi = z[first(run)], z[last(run)]
    V_e = z_hi - z_lo
    V_e > 0 || return ind, 0.0, z[first(run)]
    melt = integrated_content(ϕ, z, z_lo, z_hi)
    melt > 0 || return ind, 0.0, 0.0
    zc = integrated_content(ϕ .* z, z, z_lo, z_hi) / melt
    return ind, V_e, zc
end

"""
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid=nothing)

Integrate the master overpressure ODE over one thermal step. Equating the mechanical
(elastic+viscous shell) and thermodynamic volumetric rates:

    (1/β_r + 1/β_m) dP/dt = Ṁ_in/(ρV) - (1/ρ)dρ/dt|_{T,ϕ} - (P-P_lith)/η_r

where `1/β_m = (1/ρ)∂ρ/∂P` (gas-dominated compressibility, by finite difference), the
fixed-P density rate is finite-differenced from the previous mush (T,ϕ), and recharge
`Ṁ_in = ȧ·ρ_melt` [kg/m²/s] comes from the accretion rate `ȧ` [m/s].

**Sub-stepping + internal drain (mass-conserving eruptions).** The overpressure builds
much faster than the thermal Δt (recharge can add ≫ΔP_crit per thermal step — a stiff
ODE, gap §1.8), so a single Euler step overshoots and, cycled every step, erupts more
melt than was recharged. Instead we sub-step so ΔP moves ≲¼·ΔP_crit per sub-step; and
whenever ΔP crosses `ΔP_crit` (and not gas-locked, and — if `z_centroid` is given —
shallow enough), the chamber **drains** the stored volume `V_e·(ΔP-ΔP_relax)·(1/β_r+1/β_m)`
back to `ΔP_relax`. Each drain releases exactly the volume its overpressure represents, so
summed over the step the erupted melt equals the recharge that came in (mass conserved,
gap §1.2). The total drained melt thickness for the step is returned in `state.h_erupt`
(0 if the chamber only charged); the caller queues it with [`pending_withdrawal!`](@ref)
and books an eruption only after the accumulated thickness is physically withdrawn.
"""
function step_overpressure!(state::EruptionState, ep::EruptionParams,
                            T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid=nothing)
    validate_eruption_params(ep)
    all(isfinite, (T_mush_K, ϕ_mush, V_e, ȧ, Δt)) ||
        throw(ArgumentError("chamber-step inputs must be finite"))
    T_mush_K > 0 || throw(ArgumentError("T_mush_K must be positive"))
    0 <= ϕ_mush <= 1 || throw(DomainError(ϕ_mush, "ϕ_mush must lie in [0, 1]"))
    V_e >= 0 || throw(ArgumentError("V_e must be nonnegative"))
    ȧ >= 0 || throw(ArgumentError("ȧ must be nonnegative"))
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    z_centroid === nothing && throw(ArgumentError(
        "z_centroid is required to select shallow-gas or deep-condensed physics"))
    isfinite(z_centroid) || throw(ArgumentError("z_centroid must be finite"))
    ρ, ϕg   = mixture_density(state.P, T_mush_K, ϕ_mush, ep; z_centroid)
    state.ϕ_g   = ϕg
    state.h_erupt = 0.0
    # H₂O speciation diagnostics (dissolved / exsolved mass fractions, gas density) for tracking
    state.m_diss, state.X_g, state.ρ_gas, _ =
        water_gas_partition(state.P, T_mush_K, ϕ_mush, ep; z_centroid)
    state.η_r = ep.η_r    # record the wall viscosity the caller set (per-model, ep is shared)
    state.ϕ_mush = ϕ_mush # mush-mean melt fraction driving the split (for tracking)
    if !state.init || V_e <= 0
        state.T_prev = T_mush_K
        state.ϕ_prev = ϕ_mush
        state.init   = true
        return state
    end

    # magma compressibility 1/β_m = (1/ρ)∂ρ/∂P  (finite difference at fixed T,ϕ)
    dP       = max(1e3, 1e-4*max(state.P, 1e5))
    ρp, _    = mixture_density(state.P + dP, T_mush_K, ϕ_mush, ep; z_centroid)
    inv_βm   = (ρp - ρ)/(ρ*dP)
    state.inv_βm = inv_βm

    # thermodynamic source: -(1/ρ) dρ/dt at fixed P (T & ϕ change only). Held over the
    # sub-steps (T, ϕ only change on the thermal Δt).
    ρ_old, _ = mixture_density(state.P, state.T_prev, state.ϕ_prev, ep; z_centroid)
    dρdt_TP  = (ρ - ρ_old)/Δt
    Ṁ_in     = ȧ*ep.ρ_melt
    S        = Ṁ_in/(ρ*V_e) - dρdt_TP/ρ
    inv_βr   = 1.0/ep.β_r
    invβ     = inv_βr + inv_βm

    # sub-step count: keep |ΔP| change per sub-step ≲ ¼·ΔP_crit so the drain doesn't
    # overshoot (which is what over-counted erupted volume before)
    dPdt0    = (S - (state.P - state.P_lith)/ep.η_r) / invβ
    # clamp in float space BEFORE the Int cast: a soft (floored) η_r can make |dPdt0| huge,
    # and ceil(Int, >9.2e18) overflows Int64 (InexactError) before the clamp can cap it.
    nsub     = round(Int, clamp(ceil(abs(dPdt0)*Δt / (0.25*ep.ΔP_crit)), 1.0, 10_000.0))
    dt_sub   = Δt/nsub

    can_drain = ϕg < ep.ϕ_g_crit &&
                (z_centroid === nothing || abs(z_centroid) <= ep.z_erupt_max)
    h_out = 0.0
    for _ in 1:nsub
        dPdt      = (S - (state.P - state.P_lith)/ep.η_r) / invβ
        state.P  += dPdt*dt_sub
        if can_drain && (state.P - state.P_lith) >= ep.ΔP_crit
            h_out  += V_e*(state.P - state.P_lith - ep.ΔP_relax)*invβ
            state.P = state.P_lith + ep.ΔP_relax
        end
    end

    state.h_erupt = h_out
    state.T_prev  = T_mush_K
    state.ϕ_prev  = ϕ_mush
    return state
end

"""
    init_eruption!(state, ep, z_centroid)

Initialise the chamber at lithostatic pressure (zero overpressure) for a chamber whose
melt-weighted centroid sits at depth `z_centroid` [m, negative down]:
`P_lith = ρ_crust·g·|z_centroid|`, `P = P_lith`.
"""
function init_eruption!(state::EruptionState, ep::EruptionParams, z_centroid)
    state.P_lith = ep.ρ_crust*ep.g*abs(z_centroid)
    state.P      = state.P_lith
    return state
end

"""
    update_lithostatic!(state, ep, z_centroid)

Keep the lithostatic reference tracking the (moving) chamber centroid. First call
initialises at lithostatic (as [`init_eruption!`](@ref)); later calls shift `P` by the
change in `P_lith` so `ΔP = P - P_lith` stays continuous while the reference follows the
chamber. Call once per timestep before [`step_overpressure!`](@ref).
"""
function update_lithostatic!(state::EruptionState, ep::EruptionParams, z_centroid)
    P_lith = ep.ρ_crust*ep.g*abs(z_centroid)
    if state.P_lith == 0.0 && state.P == 0.0
        state.P = P_lith
    else
        state.P += P_lith - state.P_lith
    end
    state.P_lith = P_lith
    return state
end

"""
    overpressure_erupts(state, ep, z_centroid) -> Bool

Eruption criterion for the 3-phase trigger: roof failure `ΔP ≥ ΔP_crit`, not gas-locked
`ϕ_g < ϕ_g_crit`, and the chamber shallower than `z_erupt_max`. The caller removes the
band with `erupt_melt!` and then relaxes `state.P = state.P_lith + ep.ΔP_relax`.
"""
function overpressure_erupts(state::EruptionState, ep::EruptionParams, z_centroid)
    (state.P - state.P_lith) >= ep.ΔP_crit &&
        state.ϕ_g < ep.ϕ_g_crit &&
        abs(z_centroid) <= ep.z_erupt_max
end

"""
    advect_w!(Params)

Semi-Lagrangian advection of `Params.Told` by the vertical velocity field `Params.w`
[m/s] over one timestep `Params.Δt`, using the same `semilagrangian_advection` scheme
as discrete sill injection (`insert_sill`). Mimics the host-rock displacement caused by
continuous magma accretion (e.g. as set by `compute_Q_magma!`) in a way consistent with
how discrete sills displace the column, rather than via an upwind advection term in the
residual.
"""
function advect_w!(Params)
    Displ = Params.w .* Params.Δt
    Params.Told .= semilagrangian_advection(Params.Told, Displ, Params.z)
    return Params.Told
end

"""
    advect_markers!(markers, Params)

Advance a vector of passive marker depths `markers` [m] by one timestep using the
host-rock advection velocity `Params.w` [m/s], interpolated onto the (possibly
off-grid) marker positions. Used to visualize how far host rock outside the
injection zone has moved under `compute_Q_magma!`/`advect_w!`, analogous to the
`rocks` field tracked for discrete sill injection.
"""
function advect_markers!(markers, Params)
    w_interp = linear_interpolation(Params.z, Params.w; extrapolation_bc=Line())
    markers .+= w_interp.(markers) .* Params.Δt
    return markers
end

"""
    Tracer

Passive tracer carrying its own temperature-time history, in the same ragged
per-tracer-vector convention used by MagmaThermoKinematics.jl (and consumed
directly by ZirconGrowth.jl's `simulate_from_cooling_path(time_Myr, T_C)`):
`time_vec` is in Myr, `T_vec` and `T` are in °C. Melt fraction `phi` (0-1) is
tracked alongside temperature so eruptibility (whether a tracer is currently
within the eruptible melt-fraction window) can be evaluated later.

# Fields
- `z::Float64`: current depth [m]
- `T::Float64`: current temperature [°C]
- `phi::Float64`: current melt fraction [-]
- `phase::Int`: 0 = host rock, 1 = injected sill/magma material
- `time_vec::Vector{Float64}`: recorded times [Myr], growing over the run
- `T_vec::Vector{Float64}`: recorded temperatures [°C], growing over the run
"""
mutable struct Tracer
    z        :: Float64
    T        :: Float64
    phi      :: Float64
    phase    :: Int
    time_vec :: Vector{Float64}
    T_vec    :: Vector{Float64}
end

"""
    init_tracers(Silltop, Sillbot; n=20)

Seed `n` passive tracers uniformly across the injection zone `z ∈ [-Sillbot, -Silltop]`
(in km, as elsewhere in the GUI), with `phase=0` (host rock) and an as-yet-empty
temperature-time history.
"""
function init_tracers(Silltop, Sillbot; n=20)
    zs = range(-Sillbot*1e3, -Silltop*1e3, length=n)
    return [Tracer(z, 0.0, 0.0, 0, Float64[], Float64[]) for z in zs]
end

"""
    add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=5)

Seed `n` new passive tracers within a freshly-inserted sill centered at `Sill_z0` [m]
with thickness `Sill_thick` [m], at temperature `Sill_T` [°C] and `phase=1` (injected
material), and append them to `tracers`.
"""
function add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=5)
    zs = range(Sill_z0 - Sill_thick/2, Sill_z0 + Sill_thick/2, length=n)
    append!(tracers, [Tracer(z, Sill_T, 1.0, 1, Float64[], Float64[]) for z in zs])
    return tracers
end

"""
    add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=2)

Seed `n` new passive tracers at the center of the injection zone `z ∈ [-Sillbot, -Silltop]`
(in km), at temperature `Tsill` [°C] and `phase=1` (injected material), and append them to
`tracers`. Used with `compute_Q_magma!`/`advect_w!` to keep replenishing tracers at the
zone as the host rock is continuously advected away from its center, analogous to
`add_sill_tracers!` for discrete sill injection.
"""
function add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=2)
    z0 = -(Silltop + Sillbot)/2*1e3
    append!(tracers, [Tracer(z0, Tsill, 1.0, 1, Float64[], Float64[]) for _ in 1:n])
    return tracers
end

"""
    advect_tracers!(tracers, Params)

Advance each tracer's depth `tracer.z` by one timestep using the host-rock advection
velocity `Params.w` [m/s], exactly as `advect_markers!` does for the injection-zone
boundary markers. Use this for the `compute_Q_magma!`/`advect_w!` path; discrete sill
injection does not populate `Params.w` and should use `advect_tracers_sill!` instead.
"""
function advect_tracers!(tracers, Params)
    w_interp = linear_interpolation(Params.z, Params.w; extrapolation_bc=Line())
    for tracer in tracers
        tracer.z += w_interp(tracer.z) * Params.Δt
    end
    return tracers
end

"""
    advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic)

Displace each tracer's depth `tracer.z` by the same per-point displacement law that
`insert_sill` applies to the temperature/rock fields when a sill of full thickness
`Sill_thick` [m] is emplaced at `Sill_z0` [m]: zero inside the sill, and each wall
moves by at most `Sill_thick/2`.
Use this for the discrete-sill path; `compute_Q_magma!`/`advect_w!` does not displace
the column this way and should use `advect_tracers!` instead.
"""
function advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic, r=5e3)
    Sill_thick > 0 || throw(ArgumentError("Sill_thick must be positive"))
    SillType in (:constant, :elastic) ||
        throw(ArgumentError("SillType must be :constant or :elastic"))
    r > 0 || throw(ArgumentError("r must be positive"))
    for tracer in tracers
        z_shift = tracer.z - Sill_z0
        if abs(z_shift) <= Sill_thick/2
            continue   # inside the sill: no host-rock displacement to apply
        elseif SillType == :constant
            tracer.z += z_shift > 0 ? Sill_thick/2 : -Sill_thick/2
        elseif SillType == :elastic
            d = crack_perp_displacement(z_shift, Sill_thick/2; r=r)
            tracer.z += z_shift > 0 ? d : -d
        end
    end
    return tracers
end

"""
    update_tracers_T!(tracers, T, z, time_Myr, phi=nothing)

Interpolate the temperature field `T` (defined on grid `z` [m]) onto each tracer's
current depth, update `tracer.T`, and append `(time_Myr, tracer.T)` to the tracer's
`time_vec`/`T_vec` history. If the melt fraction field `phi` (on the same grid `z`)
is given, also interpolate it onto each tracer's depth and update `tracer.phi` —
used later to check whether a tracer is within the eruptible melt-fraction window.
"""
function update_tracers_T!(tracers, T, z, time_Myr, phi=nothing)
    T_interp = linear_interpolation(z, T; extrapolation_bc=Line())
    phi_interp = phi === nothing ? nothing : linear_interpolation(z, phi; extrapolation_bc=Line())
    for tracer in tracers
        tracer.T = T_interp(tracer.z)
        push!(tracer.time_vec, time_Myr)
        push!(tracer.T_vec, tracer.T)
        if phi_interp !== nothing
            tracer.phi = phi_interp(tracer.z)
        end
    end
    return tracers
end

"""
    volume_averaged_age(result::ZirconGrowth.SimulationResult) -> Float64

Compute the volume-averaged crystallisation age (in years, before the end of the
cooling path) of a single zircon crystal, exactly as
`MagmaThermoKinematics.volume_averaged_age` does: each concentric growth shell is
weighted by its volume (∝ `r[i+1]^3 - r[i]^3`) and assigned the age of its midpoint;
shells with zero or negative growth are excluded.
"""
function volume_averaged_age(result::ZirconGrowth.SimulationResult)
    t = result.time_years
    r = result.zircon_radius_um

    age_sum = 0.0
    vol_sum = 0.0
    t_end   = t[end]

    for i in 1:(length(r) - 1)
        dV = r[i+1]^3 - r[i]^3
        dV <= 0 && continue
        age_mid  = t_end - 0.5*(t[i] + t[i+1])
        age_sum += age_mid * dV
        vol_sum += dV
    end

    return vol_sum > 0 ? age_sum / vol_sum : 0.0
end

"""
    compute_zircon_ages(tracers; nx=100, elements=ZirconGrowth.default_element_data(),
                         return_results=false, T_zr_min=650.0, t_ref_Myr=nothing)

Run the ZirconGrowth.jl crystal-growth model on each tracer's `(time_vec, T_vec)`
cooling path (Myr, °C) and return a `NamedTuple` with one entry per successfully
simulated tracer:

- `age_years`: volume-averaged crystallisation age [yr] (see [`volume_averaged_age`](@ref))
- `zircon_radius_um`: final crystal radius [µm]

Skipped tracers: fewer than 2 recorded time steps, never hotter than `T_zr_min` [°C]
(too cold to ever grow zircon - saves the expensive simulation and keeps fake zero-age
entries out of the spectra), or no zircon growth beyond the seed radius.

All ages are referenced to a **common clock** `t_ref_Myr` (default: the latest recorded
time across `tracers`): an erupted-cargo tracer whose history stops at eruption gets the
eruption-to-reference offset added, so erupted-cargo and whole-reservoir spectra are
directly comparable. Each cooling path is passed to ZirconGrowth birth-relative, so the
internal uniform time grid resolves the tracer's actual history rather than padding back
to t=0.

Set `return_results=true` to also get the full `Vector{ZirconGrowth.SimulationResult}`.

Each tracer is independent, so the loop runs on all available Julia threads
(start Julia with `--threads auto` or `julia -t auto` for a proportional speedup;
otherwise this runs serially and can be slow for many tracers).
"""
function compute_zircon_ages(tracers; nx::Int=100,
                              elements::ZirconGrowth.ElementData=ZirconGrowth.default_element_data(),
                              return_results::Bool=false,
                              T_zr_min::Float64=650.0,
                              t_ref_Myr::Union{Nothing,Float64}=nothing)
    n                = length(tracers)
    age_years        = Vector{Union{Nothing,Float64}}(nothing, n)
    zircon_radius_um = Vector{Union{Nothing,Float64}}(nothing, n)
    _results         = return_results ? Vector{Union{Nothing,ZirconGrowth.SimulationResult}}(nothing, n) : nothing

    t_ref = isnothing(t_ref_Myr) ?
        maximum((tr.time_vec[end] for tr in tracers if length(tr.time_vec) >= 2); init=0.0) :
        t_ref_Myr

    Threads.@threads for i in eachindex(tracers)
        tracer = tracers[i]
        length(tracer.time_vec) < 2 && continue
        maximum(tracer.T_vec) < T_zr_min && continue   # never hot enough to grow zircon

        # birth-relative time: ZirconGrowth resamples onto a uniform grid starting at the
        # first control point of the path; passing absolute model time would flat-pad the
        # temperature back to t=0 and squeeze late-born tracers' real history
        time_Myr = Float64.(tracer.time_vec) .- Float64(tracer.time_vec[1])
        T_C      = Float64.(tracer.T_vec)

        params = ZirconGrowth.GrowthParams(time_Myr, T_C; nx=nx)
        res = ZirconGrowth.simulate_from_cooling_path(time_Myr, T_C; params=params, elements=elements)

        # no growth beyond the seed radius -> no zircon, no age
        res.zircon_radius_um[end] - res.zircon_radius_um[1] < 1e-3 && continue

        # common reference clock: add the offset from this tracer's last recorded time
        # (eruption time for erupted cargo) to the shared reference
        age_years[i]        = volume_averaged_age(res) + (t_ref - tracer.time_vec[end])*1e6
        zircon_radius_um[i] = res.zircon_radius_um[end]
        return_results && (_results[i] = res)
    end

    age_years        = Float64[v for v in age_years        if !isnothing(v)]
    zircon_radius_um = Float64[v for v in zircon_radius_um if !isnothing(v)]

    if return_results
        results = ZirconGrowth.SimulationResult[r for r in _results if !isnothing(r)]
        return (; age_years, zircon_radius_um, results)
    end

    return (; age_years, zircon_radius_um)
end
