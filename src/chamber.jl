# Lumped 3-phase magma chamber (Degruyter & Huber): overpressure evolution as an eruption trigger.

# =====================================================================================
#  OVERPRESSURE TRIGGER  (Degruyter & Huber 2014 / Townsend 2021, 3-phase)
# =====================================================================================
# A lumped 3-phase (melt/crystal/gas) chamber whose overpressure ΔP = P - P_lith evolves from a
# master ODE driven by the column's own Ṫ and ϕ̇. Gas exsolution/water solubility set the
# compressibility, and the dike reaches the surface only when overpressure plus magma
# buoyancy overcome the barrier stress (see [`dike_ascends`](@ref)).
# This is a *trigger*: the actual erupted band is removed with erupt_melt! as elsewhere.

"""
    EruptionParams

Tunable parameters for the 3-phase overpressure trigger. Defaults follow Degruyter & Huber
(2014) Table 1 and are order-of-magnitude values; check the paper for a specific system.

- `ϕ_erupt`  : mobile-melt threshold; the eruptible chamber is the mush ϕ ≥ ϕ_erupt
- `h_melt_min`: melt content `∫ϕ dz` [m] a connected mush must hold to count as a chamber.
  Must exceed the melt one injection delivers, or a freshly emplaced sill registers as its
  own chamber and erupts in the step it arrives.
- `ΔP_crit`  : roof-failure overpressure [Pa]
- `ϕ_g_crit` : gas-volume-fraction lock-up (shut-off) [-]
- `ΔP_relax` : overpressure left after an eruption relaxes the chamber [Pa]
- `σ_barrier`: stress [Pa] a dike must overcome to reach the surface ([`dike_ascends`](@ref))
- `w_dike`   : dike half-width [m]. Ascent reach goes as `w^4`
  ([`max_ascent_length`](@ref)), making this the most sensitive propagation parameter.
- `κ_magma`  : magma thermal diffusivity [m²/s]
- `melt_viscosity`: melt viscosity law η(T) (basaltic or rhyolitic)
- `μ_shear`: host-rock shear modulus [Pa]. Chamber compliance is derived from it and the
  chamber shape ([`host_compliance`](@ref)); magma compressibility enters the ODE
  separately as `1/β_m` and must not be folded in here.
- `R_sill` : chamber radius [m], the crack radius sills open against and the extent of the
  2-D/3-D export ([`lateral_profile`](@ref))
- `η_r`    : wall relaxation viscosity [Pa·s]
- `ρ_melt`,`ρ_x`,`ρ_gas`: melt / crystal / gas density laws `ρ(P,T)`. The condensed
  phases are compressible on purpose: with rigid
  condensed magma `1/β_m` drops discontinuously to zero once the melt dissolves all of
  `m_w`. Defaults give bulk moduli of ~10 GPa and ~67 GPa; the default gas law is the
  modified Redlich–Kwong parameterization.
- `solubility`: H₂O saturation law `m_eq(P,T)` per *melt* mass
- `m_w`      : total water mass fraction of the magma [-]. The default is dry (`0.0`), so
  the gas-density law is not evaluated unless water is explicitly supplied.
- `A_visc`,`G_act`: Arrhenius constants of the wall creep law
  `η(T) = A_visc·exp(G_act/(RT))` [Pa·s], [J/mol]
- `crust`: crustal density law `ρ(P,T)` for the lithostatic column
- `g`: gravity [m/s²]

`melt_viscosity` and `solubility` are both fixed by magma composition — see
[`gui_composition`](@ref).
"""
Base.@kwdef mutable struct EruptionParams
    ϕ_erupt::Float64 = 0.5
    h_melt_min::Float64 = 500.0
    ΔP_crit::Float64 = 20.0e6
    ϕ_g_crit::Float64 = 0.5
    ΔP_relax::Float64 = 0.0
    σ_barrier::Float64 = 10.0e6
    w_dike::Float64 = 2.0
    κ_magma::Float64 = 1.0e-6
    melt_viscosity::AbstractCreepLaw = LinearMeltViscosity()
    μ_shear::Float64 = 1.0e10
    R_sill::Float64 = 5.0e3
    η_r::Float64 = 1.0e19
    ρ_melt::AbstractDensity = PT_Density(
        ρ0 = 2400kg / m^3, α = 3.0e-5 / K, β = 1.0e-10 / Pa,
        T0 = 273.15K, P0 = 0Pa
    )
    ρ_x::AbstractDensity = PT_Density(
        ρ0 = 2700kg / m^3, α = 3.0e-5 / K, β = 1.5e-11 / Pa,
        T0 = 273.15K, P0 = 0Pa
    )
    ρ_gas::AbstractDensity = RedlichKwong_Density()
    solubility::AbstractSolubility = Liu2005_Solubility()
    m_w::Float64 = 0.0
    A_visc::Float64 = 4.25e7
    G_act::Float64 = 141.0e3
    crust::AbstractDensity = PT_Density(
        ρ0 = 2700kg / m^3, α = 3.0e-5 / K, β = 1.5e-11 / Pa,
        T0 = 273.15K, P0 = 0Pa
    )
    g::Float64 = 9.81
end

function validate_eruption_params(ep::EruptionParams)
    all(
        isfinite, (
            ep.ϕ_erupt, ep.ΔP_crit, ep.ϕ_g_crit, ep.ΔP_relax, ep.σ_barrier,
            ep.μ_shear, ep.R_sill, ep.η_r, ep.m_w,
            ep.A_visc, ep.G_act, ep.g,
        )
    ) ||
        throw(ArgumentError("eruption parameters must be finite"))
    0 <= ep.ϕ_erupt <= 1 || throw(DomainError(ep.ϕ_erupt, "ϕ_erupt must lie in [0, 1]"))
    isfinite(ep.h_melt_min) && ep.h_melt_min >= 0 ||
        throw(ArgumentError("h_melt_min must be finite and nonnegative"))
    0 <= ep.ϕ_g_crit <= 1 || throw(DomainError(ep.ϕ_g_crit, "ϕ_g_crit must lie in [0, 1]"))
    ep.ΔP_crit > 0 || throw(ArgumentError("ΔP_crit must be positive"))
    0 <= ep.ΔP_relax < ep.ΔP_crit ||
        throw(ArgumentError("ΔP_relax must lie in [0, ΔP_crit)"))
    ep.σ_barrier >= 0 || throw(ArgumentError("σ_barrier must be nonnegative"))
    all(isfinite, (ep.w_dike, ep.κ_magma)) && ep.w_dike > 0 && ep.κ_magma > 0 ||
        throw(ArgumentError("w_dike and κ_magma must be finite and positive"))
    all(>(0), (ep.μ_shear, ep.R_sill, ep.η_r, ep.A_visc, ep.g)) ||
        throw(ArgumentError("mechanical and viscosity parameters must be positive"))
    0 <= ep.m_w <= 1 || throw(DomainError(ep.m_w, "m_w must lie in [0, 1]"))
    # the density laws are evaluated at a mid-crustal reference state rather than inspected:
    # any AbstractDensity is allowed, so what matters is that it returns something usable
    ρ_ref = (melt_density(ep, 2.0e8, 1123.15), crystal_density(ep, 2.0e8, 1123.15))
    all(isfinite, ρ_ref) && all(>(0), ρ_ref) || throw(
        DomainError(
            ρ_ref,
            "ρ_melt and ρ_x must give positive densities at the reference state"
        )
    )
    # Exercise the gas law at a representative reservoir state. `gas_density` already
    # rejects a nonpositive or out-of-range result on every call, so the state only has to
    # be evaluated here, not re-checked.
    iszero(ep.m_w) || gas_density(ep, 2.0e8, 1123.15)
    ep.G_act >= 0 || throw(ArgumentError("G_act must be nonnegative"))
    ρ_c = crust_reference_density(ep)
    isfinite(ρ_c) && ρ_c > 0 ||
        throw(DomainError(ρ_c, "crust must give a positive density at the reference state"))
    return ep
end

"""
    crust_density(ep::EruptionParams, T_K, P) -> ρ [kg/m³]

Crustal density at temperature `T_K` [K] and pressure `P` [Pa] from `ep.crust`.
"""
crust_density(ep::EruptionParams, T_K, P) = compute_density(ep.crust, (; T = T_K, P))

"""
    melt_density(ep::EruptionParams, P, T_K) -> ρ [kg/m³]
    crystal_density(ep::EruptionParams, P, T_K) -> ρ [kg/m³]
    gas_density(ep::EruptionParams, P, T_K) -> ρ [kg/m³]

Phase densities at `P` [Pa], `T_K` [K] from the corresponding density laws in `ep`.
Pressure-dependent melt and crystal laws give the condensed magma a finite compressibility;
see [`mixture_density`](@ref).
"""
melt_density(ep::EruptionParams, P, T_K) = compute_density(ep.ρ_melt, (; P, T = T_K))

@doc (@doc melt_density)
crystal_density(ep::EruptionParams, P, T_K) = compute_density(ep.ρ_x, (; P, T = T_K))

"""
    gas_pressure_floor(law, T_K) -> P [Pa]

Lowest pressure at which a gas-density law still increases with pressure; zero for a law
that has no such limit.

`RedlichKwong_Density` carries the Huber et al. (2010) H₂O parameterization, which is an
empirical power law rather than the Redlich–Kwong equation it is named for. Its `ω^-1.135`
term diverges as `P → 0` instead of approaching the ideal gas, so the fit has a spurious
density minimum. Below that minimum `∂ρ_g/∂P < 0`: decompression makes the gas denser,
which drives the mixture compressibility `1/β_m` negative and leaves the chamber's
mass-constraint solve ([`project_mass_constraint!`](@ref)) without a descent direction.
Solving `∂ρ/∂ω = 0` for `ω = P/Pref` and `τ = (T - T0)/Tref` gives

    ω_min = (1.135 a₂ / (0.033 a₃) · τ^0.411)^(1/1.168)

which is 23–27 MPa between 700 and 1100 °C. Coefficients come from the law rather than
being written out here, so a refit moves the floor with it.
"""
gas_pressure_floor(::AbstractDensity, T_K) = zero(float(T_K))

function gas_pressure_floor(law::RedlichKwong_Density, T_K)
    _, a2, a3 = law.coeffs
    τ = (T_K - NumValue(law.T0)) / NumValue(law.Tref)
    τ > 0 || throw(
        DomainError(T_K, "gas temperature must exceed the law's reference temperature")
    )
    return NumValue(law.Pref) * (1.135 * a2 / (0.033 * a3) * τ^0.411)^(1 / 1.168)
end

"""
    gas_density(ep::EruptionParams, P, T_K) -> ρ_g [kg/m³]

Exsolved-H₂O density from `ep.ρ_gas`, refusing any state its law cannot represent: a
pressure below [`gas_pressure_floor`](@ref), where the density decreases with pressure, and
a nonpositive density, which the Huber fit returns near its minimum above ~1050 °C. Both
are rejected rather than clamped — a clamp would return a plausible number and silently
corrupt `ϕ_g`, the compressibility, and every eruption volume derived from them.
"""
function gas_density(ep::EruptionParams, P, T_K)
    P_floor = gas_pressure_floor(ep.ρ_gas, T_K)
    P >= P_floor || throw(
        DomainError(
            P,
            "pressure is below the $(P_floor) Pa validity floor of " *
                "$(nameof(typeof(ep.ρ_gas))) at $(T_K) K, where ∂ρ_g/∂P < 0"
        )
    )
    ρ_g = compute_density(ep.ρ_gas, (; P, T = T_K))
    ρ_g > 0 || throw(
        DomainError(ρ_g, "gas density law returned a nonpositive density at $(P) Pa, $(T_K) K")
    )
    return ρ_g
end

"""
    condensed_density(ep::EruptionParams, P, T_K, ϕ_melt) -> ρ_c [kg/m³]

Volume-averaged melt + crystal density, `ϕρ_melt + (1-ϕ)ρ_x`. `ϕ_melt` is a volume
fraction — the same ϕ that `integrated_content(ϕ, z)` turns into a melt thickness.
"""
condensed_density(ep::EruptionParams, P, T_K, ϕ_melt) =
    ϕ_melt * melt_density(ep, P, T_K) + (1 - ϕ_melt) * crystal_density(ep, P, T_K)

"""
    crust_reference_density(ep::EruptionParams; T_ref=800.0, P_ref=0.0) -> ρ [kg/m³]

Crustal density at the mid-crustal reference state `T_ref` [°C], `P_ref` [Pa]. The thermal
role of the crustal density — heat storage and conduction — is a single number per model,
and this is where it comes from, so the thermal and lithostatic densities cannot drift
apart. [`check_density_consistency`](@ref) enforces that agreement.
"""
crust_reference_density(ep::EruptionParams; T_ref = 800.0, P_ref = 0.0) =
    crust_density(ep, T_ref + 273.15, P_ref)

"""
    check_density_consistency(MatParam, ep::EruptionParams; T_ref=800.0, P_ref=0.0,
                              rtol=1e-3, phase=0)

Verify that the host-rock thermal density of `MatParam` (evaluated at `T_ref` [°C])
agrees with the crustal law `ep.crust` at the same reference state
([`crust_reference_density`](@ref)). Chamber crystals and melt are distinct materials, so
`ep.ρ_x` and `ep.ρ_melt` are deliberately not compared.

Call once at model setup, where the two parameter objects are built.
"""
function check_density_consistency(
        MatParam, ep::EruptionParams; T_ref = 800.0, P_ref = 0.0,
        rtol = 1.0e-3, phase = 0
    )
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    ρ = zeros(1)
    compute_density!(ρ, MatParam, [phase], (T = [T_ref + 273.15],))
    ρ_ref = crust_reference_density(ep; T_ref, P_ref)
    isapprox(ρ[1], ρ_ref; rtol) || throw(
        ArgumentError(
            "host-rock density mismatch: MatParam gives $(ρ[1]) kg/m³ at $T_ref °C, " *
                "EruptionParams.crust gives $ρ_ref kg/m³ there (rtol=$rtol)"
        )
    )
    return ρ[1]
end

"""
    lithostatic_pressure(ep::EruptionParams, T_C, z, z_target) -> P [Pa]

Integrate the crustal load ``\\int \\rho_{crust}(P,T)\\,g\\,dz`` from the surface down to
`z_target` [m, negative down], over the column `z` (ascending, surface last) carrying
temperatures `T_C` [°C]. The density is evaluated segment by segment with the pressure
already accumulated above it, so the profile is self-consistent to the order of the
column's own resolution.

Thermal expansion makes the crust lighter downward, compression makes it denser, and the
two nearly cancel: the density varies by well under a percent over the upper crust, so the
column is close to — but not exactly — linear in depth.
"""
function lithostatic_pressure(ep::EruptionParams, T_C, z, z_target)
    axes(T_C) == axes(z) ||
        throw(DimensionMismatch("T and z must share axes: $(axes(T_C)) vs $(axes(z))"))
    issorted(z) || throw(ArgumentError("z must be ascending (surface last)"))
    isfinite(z_target) || throw(ArgumentError("z_target must be finite"))
    first(z) <= z_target <= last(z) || throw(
        ArgumentError(
            "z_target=$z_target lies outside the column [$(first(z)), $(last(z))]"
        )
    )
    P = 0.0
    for i in reverse((firstindex(z) + 1):lastindex(z))     # segments from the surface downward
        z_top = z[i]
        z_top <= z_target && continue
        z_bot = max(z[i - 1], z_target)
        Δz = z_top - z_bot
        Δz > 0 || continue
        T_K = 0.5 * (T_C[i] + T_C[i - 1]) + 273.15
        P += crust_density(ep, T_K, P) * ep.g * Δz
    end
    return P
end

"""
    EruptionState

Mutable chamber state carried across timesteps: absolute pressure `P`, lithostatic
reference `P_lith`, chamber volume `V`, areal total mass `M`, areal water mass `M_H2O`, gas
volume fraction `ϕ_g`, and the previous mush (T,ϕ) for the fixed-P density-rate source
term. Areal inventories use kg/m² because the 1-D chamber volume is a thickness; multiplying
them by `lateral_effective_area(R_sill)` gives whole-chamber masses. `h_erupt` is the bulk magma drained by the
pressure ODE during the latest thermal step; `h_pending` is drained magma not yet withdrawn
from the grid because the accumulated thickness is still sub-grid. Both are thicknesses of
the multi-phase mixture, not of its liquid: the chamber's mass and water ledgers debit an
eruption at the bulk composition, so what leaves is magma, crystals included.

`V` is the chamber's own volume [m per unit area], not a re-reading of the grid's mush
extent. The 1-D column has no way to represent walls creeping outward, so a chamber whose
volume were taken from the grid each step would discard exactly the deformation the
relaxation term describes. See [`step_overpressure!`](@ref) for how the two are reconciled:
the ODE owns the mechanical part, the grid owns melting and freezing at the margins.
`V_grid_prev` is the mush extent last seen on the grid, differenced to extract that
thermal part.

`id` names the magma body the state describes and `z_lo`/`z_hi` the grid interval it was
built on. `eruptible_mush` returns whichever lens is largest, so these are what
distinguish a chamber that has grown from one that has been replaced: a lens that does not
overlap `[z_lo, z_hi]` is a different source and gets a fresh state under the next `id`
([`reset_chamber!`](@ref)).
"""
Base.@kwdef mutable struct EruptionState
    P::Float64 = 0.0
    P_lith::Float64 = 0.0
    V::Float64 = 0.0          # chamber volume [m per unit area]
    V_grid_prev::Float64 = 0.0       # mush extent [m] last read off the grid
    M::Float64 = 0.0          # total chamber mass per unit area [kg/m²]
    M_H2O::Float64 = 0.0      # total H₂O mass per unit area [kg/m²]
    M_initial::Float64 = 0.0  # inventory at chamber formation [kg/m²]
    M_in::Float64 = 0.0       # cumulative recharge mass per unit area [kg/m²]
    M_out::Float64 = 0.0      # cumulative erupted mass per unit area [kg/m²]
    M_boundary::Float64 = 0.0 # cumulative mass crossing moving mush margins [kg/m²]
    mass_residual::Float64 = 0.0 # ρV - M [kg/m²]
    ϕ_g::Float64 = 0.0
    T_prev::Float64 = NaN          # mush-mean T [K] at previous step
    ϕ_prev::Float64 = NaN          # mush-mean ϕ at previous step
    inv_βm::Float64 = 0.0          # magma compressibility 1/β_m at the last step [1/Pa]
    h_erupt::Float64 = 0.0          # bulk magma thickness [m] drained by eruptions during the last step
    h_pending::Float64 = 0.0          # drained magma [m] awaiting a grid-resolvable withdrawal
    pending_since::Float64 = NaN      # time [s] of the first trigger represented by h_pending
    m_diss::Float64 = 0.0          # dissolved H₂O mass fraction (per magma mass), Liu 2005
    X_g::Float64 = 0.0          # exsolved H₂O gas mass fraction (per magma mass), D&H eq.18
    ρ_gas::Float64 = 0.0          # exsolved-gas density [kg/m³] at the last step (modified RK)
    ρ_magma::Float64 = 0.0          # 3-phase mixture density [kg/m³] at the last step
    η_magma::Float64 = 0.0          # melt viscosity [Pa·s] at the last step
    η_r::Float64 = 0.0          # shell relaxation viscosity used at the last step [Pa·s]
    ϕ_mush::Float64 = 0.0          # mush-mean melt fraction driving the last step
    id::Int = 0               # chamber generation; incremented whenever a chamber forms
    z_lo::Float64 = NaN       # top of the mush interval this state was built on [m]
    z_hi::Float64 = NaN       # base of that interval [m]
    init::Bool = false
end

"""
    reset_chamber!(state)

Return every field except the generation counter `id` to its fresh value. The chamber is
gone: pressure, volume, inventories, ledgers, and any queued withdrawal belong to a body
that no longer exists, and carrying them into the next chamber would erupt one magma
source's melt out of another. The next chamber to form takes `id + 1`.
"""
function reset_chamber!(state::EruptionState)
    fresh = EruptionState()
    for f in fieldnames(EruptionState)
        f === :id || setfield!(state, f, getfield(fresh, f))
    end
    return state
end

function chamber_water_fraction(state::EruptionState, fallback)
    state.M >= 0 || throw(DomainError(state.M, "chamber mass must be nonnegative"))
    0 <= state.M_H2O <= state.M || throw(
        DomainError(
            state.M_H2O,
            "chamber H₂O mass must lie between zero and total chamber mass $(state.M)"
        )
    )
    m_w = iszero(state.M) ? fallback : state.M_H2O / state.M
    0 <= m_w <= 1 || throw(DomainError(m_w, "chamber H₂O mass fraction must lie in [0, 1]"))
    return m_w
end

"""
    project_mass_constraint!(state, ep, T_K, ϕ_mush, m_w, inv_βr) -> ρ [kg/m³]

Newton-solve chamber pressure and volume for `ρ(P) V(P) == state.M`, writing `P`, `V`, and
the closing `mass_residual` into `state` and returning the mixture density.

The chamber compresses along the host's compliance `inv_βr = 1/βr`, so
`V = V₀ exp((P - P₀)/βr)`, and the magma along its own compressibility from
[`mixture_density`](@ref), differentiated with ForwardDiff. Their sum must be positive for
the step to descend; a gas law below its [`gas_pressure_floor`](@ref) makes it negative and
throws an error. Steps that would drive `P` nonpositive are halved instead.
"""
function project_mass_constraint!(state, ep, T_K, ϕ_mush, m_w, inv_βr)
    P0, V0 = state.P, state.V
    P = P0
    for _ in 1:12
        V = V0 * exp((P - P0) * inv_βr)
        ρ = first(mixture_density(P, T_K, ϕ_mush, ep; m_w))
        residual = ρ * V - state.M
        scale = max(state.M, ρ * V, 1.0)
        if abs(residual) <= 64eps(scale)
            state.P = P
            state.V = V
            state.mass_residual = residual
            return ρ
        end
        inv_βm = ForwardDiff.derivative(
            p -> first(mixture_density(p, T_K, ϕ_mush, ep; m_w)), P
        ) / ρ
        compliance = inv_βr + inv_βm
        compliance > 0 || throw(
            DomainError(compliance, "mass-constraint compliance must be positive")
        )
        ΔP = -residual / (ρ * V * compliance)
        P + ΔP > 0 || (ΔP = -0.5P)
        P += ΔP
    end
    throw(ErrorException("chamber mass projection did not converge in 12 Newton iterations"))
end

"""
    pending_withdrawal!(state, h_requested, h_available, Δz; time) -> h_realizable

Queue `h_requested` bulk magma thickness drained by the D&H pressure model. Return zero
while the queued withdrawal is at most `2Δz`; otherwise return the grid-resolvable
thickness, bounded by `h_available` — the mush extent the closure has to cut the band from,
which is a bulk thickness like the withdrawal itself. The pending amount is not removed
until [`commit_pending_withdrawal!`](@ref) is called after the physical state withdrawal.
`time` is required when starting a new queue so the realized event retains its first
trigger time.
"""
function pending_withdrawal!(state::EruptionState, h_requested, h_available, Δz; time = NaN)
    h_requested >= 0 || throw(ArgumentError("h_requested must be nonnegative"))
    h_available >= 0 || throw(ArgumentError("h_available must be nonnegative"))
    Δz > 0 || throw(ArgumentError("Δz must be positive"))
    if h_requested > 0 && state.h_pending == 0
        isfinite(time) || throw(ArgumentError("time must be finite when starting a pending withdrawal"))
        state.pending_since = time
    end
    state.h_pending += h_requested
    h_realizable = min(state.h_pending, h_available)
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
    water_gas_partition(P, T_K, ϕ_melt, ep; m_w=ep.m_w) -> (m_diss, X_g, ρ_g, m_eq)

H₂O speciation + gas density at `P` [Pa], `T_K` [K], `ϕ_melt`. Dissolved water per *magma*
mass `m_diss` from the Liu et al. (2005) saturation `m_eq` (per *melt* mass, T-dependent,
silicic, H₂O-only so `P_w=P`); exsolved-gas mass fraction `X_g = m_w − m_diss`
(D&H 2014 eq. 18, water conserved); gas density `ρ_g` from `ep.ρ_gas`.
Both [`mixture_density`](@ref) and the chamber diagnostics on [`EruptionState`](@ref) read
this, so the physics lives in one place. CO₂ is not modelled (`X_co2≡0`); there is no CO₂
phase to track.

`m_eq` is per melt *mass*, so converting it to a per-magma-mass fraction weights it by the
melt mass fraction, not by `ϕ_melt`. `ϕ_melt` is volumetric — it is the same ϕ that
`integrated_content(ϕ, z)` turns into a melt thickness — so the two differ by the
melt/crystal density contrast. The distinction is not cosmetic: `X_g` is a small difference
of two larger numbers, so at ρ_melt/ρ_x = 2400/2700 a 5 % error in the weighting becomes
tens of percent in `X_g`, and typical mush sits near the saturation crossover where that
error decides whether a gas phase exists at all.

The gas phase ends where the solubility law says it does: once the melt can dissolve all
of `m_w`, `X_g` reaches zero and `ρ_g` is returned as zero without evaluating the EOS.
"""
function water_gas_partition(P, T_K, ϕ_melt, ep::EruptionParams; m_w = ep.m_w)
    all(isfinite, (P, T_K, ϕ_melt, m_w)) ||
        throw(ArgumentError("P, T_K, ϕ_melt, and m_w must be finite"))
    # the solubility law takes sqrt(P): a negative pressure means the caller's chamber
    # state is unphysical, and it must say so here rather than deep inside GeoParams
    P >= 0 || throw(DomainError(P, "magma pressure must be nonnegative"))
    0 <= ϕ_melt <= 1 || throw(DomainError(ϕ_melt, "ϕ_melt must lie in [0, 1]"))
    0 <= m_w <= 1 || throw(DomainError(m_w, "m_w must lie in [0, 1]"))
    # As the magma crystallizes (ϕ_melt↓) the melt dissolves less, so X_g *rises* — second
    # boiling, D&H's dominant pressurization for small chambers.
    m_eq = first(compute_dissolved(ep.solubility, (; P, T = T_K, X_co2 = 0.0)))
    # m_eq is water per melt mass; m_diss is water per magma mass, so the weight is the
    # melt MASS fraction. ϕ_melt is a volume fraction, and the condensed mixture carries the
    # melt/crystal density contrast between the two.
    ρ_m = melt_density(ep, P, T_K)
    x_melt = ϕ_melt * ρ_m / (ϕ_melt * ρ_m + (1 - ϕ_melt) * crystal_density(ep, P, T_K))
    # Liu 2005 is only calibrated to ~500 MPa; beyond that the -P^1.5 term drives the
    # saturation negative, which would make X_g blow past m_w (and corrupt ρ/ϕ_g). A
    # saturation can't be negative, and dissolved water can't exceed the total m_w — clamp,
    # so m_diss and X_g stay in [0, m_w] (they are mass fractions).
    m_diss = clamp(m_eq * x_melt, zero(m_eq), m_w)
    X_g = m_w - m_diss                  # exsolved-gas mass fraction ∈ [0, m_w]
    ρ_g = X_g > 0 ? gas_density(ep, P, T_K) : zero(P)
    return m_diss, X_g, ρ_g, m_eq
end

"""
    mixture_density(P, T_K, ϕ_melt, ep; m_w=ep.m_w) -> (ρ, ϕ_g)

Three-phase (melt + crystal + exsolved gas) mixture density [kg/m³] and gas volume
fraction at pressure `P` [Pa], temperature `T_K` [K] and melt fraction `ϕ_melt`. The H₂O
speciation and gas density come from [`water_gas_partition`](@ref); water-undersaturated
magma is a two-phase condensed mixture with `ϕ_g = 0`.

!!! note "Compressibility steps down at water saturation"
    `1/β_m = (1/ρ)∂ρ/∂P` is dominated by gas resorption `dX_g/dP`, not by gas compression,
    so it falls sharply where the melt becomes able to dissolve all of `m_w` even though
    `ϕ_g` reaches zero continuously. Gas-saturated and undersaturated magma genuinely differ
    in compressibility by close to an order of magnitude, so the step is physical. It stays
    *finite* because `ρ_melt` and `ρ_x` are compressible density laws: undersaturated magma
    retains the compliance of its own melt and crystals rather than becoming perfectly
    rigid. Constant densities would send `1/β_m` to exactly zero and make the relaxation
    time and drained volume jump discontinuously across a single grid point.
"""
function mixture_density(P, T_K, ϕ_melt, ep::EruptionParams; m_w = ep.m_w)
    ρ_c = condensed_density(ep, P, T_K, ϕ_melt)   # condensed (melt+crystal) density
    _, Xg, ρ_g, _ = water_gas_partition(P, T_K, ϕ_melt, ep; m_w)
    Xg > 0 || return ρ_c, zero(ρ_c)
    Vg = Xg / ρ_g
    Vc = (1 - Xg) / ρ_c
    return 1 / (Vg + Vc), Vg / (Vg + Vc)
end

"""
    host_compliance(ep::EruptionParams, V) -> 1/β_r [1/Pa]

Volumetric compliance of the host rock around a chamber of thickness `V` [m per unit area]
and radius `ep.R_sill`:

    1/β_r = 3/(4 μ ε),    ε = (V/2)/R_sill

`3/(4μ)` is the classical result for a *spherical* chamber in an elastic half-space. A
magma body of this kind is not spherical — the 1-D column is the axis of a sill of lateral
extent `R_sill`, the same radius the 2-D/3-D export tapers over — and an oblate
chamber is more compliant than a sphere of equal volume, with the compliance growing as
`1/ε`. Dividing by the aspect ratio reproduces the sphere exactly at `ε = 1` and carries
the correct scaling for a flattening sill without introducing a constant to calibrate.

Because the chamber's volume is its own state, this is evaluated per step rather than
stored: an inflating chamber flattens against a fixed radius and so grows more compliant.
`ε` is capped at 1, since a body thicker than it is wide is no longer a sill and the
spherical result is the right floor.

This is the compliance of the *rock alone*. The magma's own compressibility enters the
overpressure ODE separately as `1/β_m` ([`mixture_density`](@ref)).
"""
function host_compliance(ep::EruptionParams, V)
    V > 0 || throw(DomainError(V, "chamber volume must be positive"))
    ε = min(0.5 * V / ep.R_sill, 1.0)
    return 3 / (4 * ep.μ_shear * ε)
end

"""
    crustal_viscosity(ep, T_K) -> η [Pa·s]

Arrhenius creep viscosity of the country rock at `T_K` [K] (D&H 2014 eq. A.19),
`η = A_visc·exp(G_act/(RT))`. This is the *local* law that
[`crustal_relaxation_viscosity`](@ref) integrates over the visco-elastic shell; the wall is
the hottest and most compliant rock in that shell, so evaluating this at the wall alone
understates `η_r` by orders of magnitude.
"""
function crustal_viscosity(ep::EruptionParams, T_K)
    isfinite(T_K) && T_K > 0 || throw(ArgumentError("T_K must be finite and positive"))
    return ep.A_visc * exp(ep.G_act / (8.314 * T_K))    # 8.314 J/mol/K
end

"""
    crustal_relaxation_viscosity(ep, T_mush_K, T_far_K; shell_ratio=11, n=64) -> η_r [Pa·s]

Effective viscosity of the visco-elastic shell the chamber deforms against
(D&H 2014 eq. A.18),

    η_r = 3a³ ∫ₐ^S η(T(r)) r⁻⁴ dr,    S = shell_ratio·a,

with `η` the local Arrhenius law ([`crustal_viscosity`](@ref)) and `T(r)` the steady
spherical conduction profile (eq. A.13) running from the chamber wall at `T_mush_K` to the
far field at `T_far_K`. Clamped to `[1e17, 1e24]`: above the ceiling the shell is elastic
on any timescale the model resolves, below the floor it relaxes within a single step.

`T_far_K` is the geotherm at the chamber's own depth, not the surface temperature — the
shell is radial and never reaches the cold shallow crust. It dominates the result: `η_r`
swings ~5 orders of magnitude per 200 K of `T_far_K`, because `η` rises exponentially
outward faster than the `r⁻⁴` kernel decays.

Substituting `r = a·s` makes profile and kernel functions of `s` alone, so the chamber
radius cancels and `η_r` depends only on the two temperatures and `shell_ratio`.
"""
function crustal_relaxation_viscosity(
        ep::EruptionParams, T_mush_K, T_far_K;
        shell_ratio = 11.0, n = 64
    )
    all(isfinite, (T_mush_K, T_far_K)) && T_mush_K > 0 && T_far_K > 0 ||
        throw(ArgumentError("T_mush_K and T_far_K must be finite and positive"))
    T_far_K <= T_mush_K || throw(
        ArgumentError(
            "country rock at $T_far_K K is hotter than the mush it encloses ($T_mush_K K); " *
                "the conduction profile between them is inverted"
        )
    )
    shell_ratio > 1 || throw(ArgumentError("shell_ratio must exceed 1"))
    iseven(n) && n > 0 || throw(ArgumentError("n must be a positive even number of intervals"))
    # A.13 on the normalised radius s = r/a
    T_shell(s) = (T_mush_K * (shell_ratio - s) + shell_ratio * T_far_K * (s - 1)) /
        ((shell_ratio - 1) * s)
    integrand(s) = 3 * crustal_viscosity(ep, T_shell(s)) / s^4
    # Simpson over s; the integrand is smooth and monotonic, and n = 64 holds four
    # significant figures against a reference refinement
    h = (shell_ratio - 1) / n
    total = integrand(1.0) + integrand(shell_ratio)
    for i in 1:(n - 1)
        total += (isodd(i) ? 4 : 2) * integrand(1 + i * h)
    end
    return clamp(total * h / 3, 1.0e17, 1.0e24)
end

"""
    eruptible_mush(ϕ, z; ϕ_erupt=0.5) -> (ind, V_e, z_centroid)

Indices of the largest contiguous mobile-mush chamber (`ϕ ≥ ϕ_erupt`), its bulk
thickness `V_e` [m per unit area], and its melt-weighted centroid depth. Disconnected
melt lenses are independent chambers and are never combined into one pressure reservoir.
"""
function eruptible_mush(ϕ, z; ϕ_erupt = 0.5)
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
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid=nothing, η_r=ep.η_r)

Integrate the master overpressure ODE over one thermal step. Equating the mechanical
(elastic+viscous shell) and thermodynamic volumetric rates:

    (1/β_r + 1/β_m) dP/dt = Ṁ_in/(ρV) - (1/ρ)dρ/dt|_{T,ϕ} - (P-P_lith)/η_r

where `1/β_m = (1/ρ)∂ρ/∂P` (gas-dominated compressibility, differentiated exactly through
the solubility law and gas EOS), `1/β_r` is the host-rock compliance at the chamber's
current aspect ratio ([`host_compliance`](@ref)), the fixed-P density rate is
finite-differenced from the previous mush (T,ϕ), and recharge `Ṁ_in = ȧ·ρ_melt` [kg/m²/s] comes from the accretion
rate `ȧ` [m/s]. `η_r` is the shell relaxation viscosity for *this* chamber; pass it
explicitly ([`crustal_relaxation_viscosity`](@ref)) when several chambers share one `ep`.

This pressure equation is the differential form of `M = ρV`, but `M` is also integrated
explicitly as an independent audit:

    dM/dt = Ṁ_in - Ṁ_out,
    dM_H2O/dt = m_w,in Ṁ_in - m_w Ṁ_out.

Here the inventories are per unit sill area [kg/m²]. Magma entering or leaving through a
moving `ϕ ≥ ϕ_erupt` margin is recorded separately in `M_boundary`; it is a control-volume
boundary flux, not recharge. The state water fraction is always `M_H2O/M`, so phase
partitioning cannot silently create or destroy water. A relative disagreement greater than
0.5 % between the independent inventory and `ρV` aborts the step. Smaller integration
error is removed by an elastic Newton correction of `(P,V)` onto `M = ρV`; `M` itself is
never overwritten by the equation-of-state result.

**Chamber volume.** `V_e` is the mush extent the grid currently shows; `state.V` is the
chamber's own volume, and it is what the ODE uses. The mechanical rate

    (1/V) dV/dt = (1/β_r) dP/dt + (P-P_lith)/η_r

is integrated alongside the pressure — the same elastic and viscous terms that set `dP/dt`,
now also allowed to move the wall. Because the grid cannot represent a chamber creeping
open, `V_e` contributes only its *change* between steps, which is melting and freezing at
the mush margins. Inflation therefore feeds back on the recharge term `Ṁ_in/(ρV)`: a
chamber that accommodates recharge by growing pressurizes more slowly than one that cannot.
`state.V` is seeded from `V_e` the first time the chamber is seen.

**Sub-stepping + internal drain.** Between drains the ODE is linear in ΔP and each sub-step
advances it exactly (relaxation toward `S·η_r` with time constant `η_r·(1/β_r+1/β_m)`), so
the sub-stepping resolves the drain rather than keeping the integration stable. Sub-steps
are sized so ΔP moves by at most `0.1 * ΔP_crit`; a step carrying ΔP far past the threshold
would erupt more melt than the recharge that came in.

A drain fires whenever ΔP crosses `ΔP_crit`, provided the chamber is neither gas-locked nor
unable to drive a dike to the surface ([`dike_ascends`](@ref)). The chamber then relaxes
instantaneously to `ΔP_relax`.

The wall follows the elastic path and the EOS gives the relaxed `ρV`; the difference from
the pre-drain `M` is the erupted mass. Its equivalent pre-drain bulk thickness is returned
in `state.h_erupt`, and is 0 if the chamber only charged. The caller queues it with
[`pending_withdrawal!`](@ref) and books an eruption only once it is physically withdrawn.
"""
function step_overpressure!(
        state::EruptionState, ep::EruptionParams,
        T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid = nothing, η_r = ep.η_r
    )
    validate_eruption_params(ep)
    all(isfinite, (T_mush_K, ϕ_mush, V_e, ȧ, Δt)) ||
        throw(ArgumentError("chamber-step inputs must be finite"))
    isfinite(η_r) && η_r > 0 || throw(ArgumentError("η_r must be finite and positive"))
    T_mush_K > 0 || throw(ArgumentError("T_mush_K must be positive"))
    0 <= ϕ_mush <= 1 || throw(DomainError(ϕ_mush, "ϕ_mush must lie in [0, 1]"))
    V_e >= 0 || throw(ArgumentError("V_e must be nonnegative"))
    ȧ >= 0 || throw(ArgumentError("ȧ must be nonnegative"))
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    z_centroid === nothing && throw(
        ArgumentError(
            "z_centroid is required to weigh magma buoyancy against the lithostatic column"
        )
    )
    isfinite(z_centroid) || throw(ArgumentError("z_centroid must be finite"))
    m_w = chamber_water_fraction(state, ep.m_w)
    ρ, ϕg = mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w)
    state.ϕ_g = ϕg
    state.ρ_magma = ρ
    state.η_magma = magma_viscosity(ep, T_mush_K)
    state.h_erupt = 0.0
    # H₂O speciation diagnostics (dissolved / exsolved mass fractions, gas density) for tracking
    state.m_diss, state.X_g, state.ρ_gas, _ =
        water_gas_partition(state.P, T_mush_K, ϕ_mush, ep; m_w)
    state.η_r = η_r       # the wall viscosity this chamber ran with
    state.ϕ_mush = ϕ_mush # mush-mean melt fraction driving the split (for tracking)
    if !state.init || V_e <= 0
        state.T_prev = T_mush_K
        state.ϕ_prev = ϕ_mush
        state.V = V_e
        state.V_grid_prev = V_e
        state.M = ρ * V_e
        state.M_H2O = ep.m_w * state.M
        state.M_initial = state.M
        state.M_in = 0.0
        state.M_out = 0.0
        state.M_boundary = 0.0
        state.mass_residual = 0.0
        state.init = true
        return state
    end

    M_ledger = state.M_initial + state.M_in - state.M_out + state.M_boundary
    ledger_residual = state.M - M_ledger
    mass_scale = max(state.M, M_ledger, 1.0)
    abs(ledger_residual) <= 1.0e-10 * mass_scale || throw(
        ErrorException(
            "chamber mass closure failed: inventory differs from its flux ledger by " *
                "$ledger_residual kg/m² ($(ledger_residual / mass_scale) relative)"
        )
    )

    # The grid contributes only its change: mush margins that melted or froze since the
    # last step. Everything else about V is the ODE's, so wall deformation is not
    # overwritten by a re-reading of the ϕ ≥ ϕ_erupt band.
    ΔV_grid = V_e - state.V_grid_prev
    ΔM_boundary = ρ * ΔV_grid
    state.V += ΔV_grid
    state.M += ΔM_boundary
    state.M_H2O += m_w * ΔM_boundary
    state.M_boundary += ΔM_boundary
    state.V_grid_prev = V_e
    state.V > 0 || throw(
        DomainError(
            state.V,
            "chamber volume collapsed to zero or below; the mush is no longer a chamber"
        )
    )

    # magma compressibility 1/β_m = (1/ρ)∂ρ/∂P at fixed T,ϕ. This term outruns 1/β_r by an
    # order of magnitude in a gas-bearing chamber, so it is differentiated exactly through
    # the solubility law and the gas EOS rather than finite-differenced.
    inv_βm = ForwardDiff.derivative(
        p -> first(mixture_density(p, T_mush_K, ϕ_mush, ep; m_w)), state.P
    ) / ρ
    state.inv_βm = inv_βm

    # thermodynamic source: -(1/ρ) dρ/dt at fixed P (T & ϕ change only). Held over the
    # sub-steps (T, ϕ only change on the thermal Δt).
    ρ_old, _ = mixture_density(state.P, state.T_prev, state.ϕ_prev, ep; m_w)
    dρdt_TP = (ρ - ρ_old) / Δt
    Ṁ_in = ȧ * melt_density(ep, state.P, T_mush_K)
    S = Ṁ_in / (ρ * state.V) - dρdt_TP / ρ
    inv_βr = host_compliance(ep, state.V)
    invβ = inv_βr + inv_βm
    invβ > 0 || throw(DomainError(invβ, "total compressibility 1/β_r + 1/β_m must be positive"))

    # S and P_lith are constant over the thermal step, so the ODE is linear in ΔP and
    # integrates exactly: ΔP relaxes toward S·η_r with time constant τ = η_r·(1/β_r+1/β_m).
    # τ is routinely far shorter than Δt (a hot wall floors η_r at 1e17 s·Pa), where explicit
    # Euler amplifies ΔP by (1 - Δt/τ) each step and runs away to negative absolute pressure.
    # S is held over the sub-steps: V moves by a small fraction across one thermal step, and
    # holding it is what keeps the ODE linear and the sub-step solution exact.
    τ = η_r * invβ
    ΔP_eq = S * η_r
    ΔP_0 = state.P - state.P_lith
    # sub-step count: keep |ΔP| change per sub-step ≲ 0.1·ΔP_crit so the drain doesn't
    # overshoot and over-count erupted volume. Clamp in float space before the Int cast.
    excursion = abs(ΔP_eq - ΔP_0) * (-expm1(-Δt / τ))
    nsub = round(Int, clamp(ceil(excursion / (0.1 * ep.ΔP_crit)), 1.0, 10_000.0))
    dt_sub = Δt / nsub

    h_out = 0.0
    for _ in 1:nsub
        # A drain can change P, V, density, and compliance by much more than an ordinary
        # sub-step. Re-linearize the mass ODE at every sub-step so later recharge does not
        # keep using the pre-eruption chamber properties.
        m_w = chamber_water_fraction(state, ep.m_w)
        ρ_sub = first(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
        inv_βm_sub = ForwardDiff.derivative(
            p -> first(mixture_density(p, T_mush_K, ϕ_mush, ep; m_w)), state.P
        ) / ρ_sub
        ρ_old_sub = first(mixture_density(state.P, state.T_prev, state.ϕ_prev, ep; m_w))
        dρdt_TP_sub = (ρ_sub - ρ_old_sub) / Δt
        Ṁ_in_sub = ȧ * melt_density(ep, state.P, T_mush_K)
        S_sub = Ṁ_in_sub / (ρ_sub * state.V) - dρdt_TP_sub / ρ_sub
        inv_βr = host_compliance(ep, state.V)
        invβ_sub = inv_βr + inv_βm_sub
        τ_sub = η_r * invβ_sub
        ΔP_eq_sub = S_sub * η_r
        decay_sub = exp(-dt_sub / τ_sub)

        M_in = Ṁ_in_sub * dt_sub
        state.M += M_in
        state.M_H2O += ep.m_w * M_in
        state.M_in += M_in
        ΔP_start = state.P - state.P_lith
        ΔP = ΔP_eq_sub + (ΔP_start - ΔP_eq_sub) * decay_sub
        # viscous creep over the sub-step, integrated in closed form because ΔP(t) is a
        # known exponential
        ∫ΔP_dt = ΔP_eq_sub * dt_sub +
            (ΔP_start - ΔP_eq_sub) * τ_sub * (1 - decay_sub)
        state.P = state.P_lith + ΔP
        # Advance the continuous recharge/creep solution before applying the instantaneous
        # eruption event. This ordering prevents a whole sub-step of creep from being
        # attributed to the pressure snap-back.
        state.V *= exp((ΔP - ΔP_start) * inv_βr + ∫ΔP_dt / η_r)
        ρ_sub = project_mass_constraint!(state, ep, T_mush_K, ϕ_mush, m_w, inv_βr)
        ΔP = state.P - state.P_lith
        ϕg_sub = last(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
        if ϕg_sub < ep.ϕ_g_crit && ΔP >= ep.ΔP_crit &&
                dike_ascends(ΔP, state.P_lith, ρ_sub, state.η_magma, z_centroid, ep)
            M_before = state.M
            state.P = state.P_lith + ep.ΔP_relax
            state.V *= exp((ep.ΔP_relax - ΔP) * inv_βr)
            M_after = first(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w)) * state.V
            M_out = M_before - M_after
            M_out >= 0 || throw(
                DomainError(M_out, "pressure relaxation would add mass during an eruption")
            )
            h_out += M_out / ρ_sub
            state.M = M_after
            state.M_H2O -= m_w * M_out
            state.M_out += M_out
        end
        state.inv_βm = inv_βm_sub
    end

    m_w = chamber_water_fraction(state, ep.m_w)
    ρ, ϕg = mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w)
    state.mass_residual = ρ * state.V - state.M
    all(isfinite, (state.M, state.M_H2O, state.mass_residual)) ||
        throw(ArgumentError("chamber mass balance produced a non-finite inventory"))
    ρ = project_mass_constraint!(state, ep, T_mush_K, ϕ_mush, m_w, inv_βr)
    ϕg = last(mixture_density(state.P, T_mush_K, ϕ_mush, ep; m_w))
    state.ρ_magma = ρ
    state.ϕ_g = ϕg
    state.m_diss, state.X_g, state.ρ_gas, _ =
        water_gas_partition(state.P, T_mush_K, ϕ_mush, ep; m_w)
    state.h_erupt = h_out
    state.T_prev = T_mush_K
    state.ϕ_prev = ϕ_mush
    return state
end

"""
    init_eruption!(state, P_lith)

Initialise the chamber at lithostatic pressure (zero overpressure): `P = P_lith`. The
reference comes from [`lithostatic_pressure`](@ref) at the melt-weighted centroid.
"""
function init_eruption!(state::EruptionState, P_lith)
    P_lith >= 0 || throw(ArgumentError("P_lith must be nonnegative"))
    state.P_lith = P_lith
    state.P = P_lith
    return state
end

"""
    update_lithostatic!(state, P_lith)

Keep the lithostatic reference tracking the (moving) chamber centroid. First call
initialises at lithostatic (as [`init_eruption!`](@ref)); later calls shift `P` by the
change in `P_lith` so `ΔP = P - P_lith` stays continuous while the reference follows the
chamber. Call once per timestep before [`step_overpressure!`](@ref).
"""
function update_lithostatic!(state::EruptionState, P_lith)
    P_lith >= 0 || throw(ArgumentError("P_lith must be nonnegative"))
    if state.P_lith == 0.0 && state.P == 0.0
        state.P = P_lith
    else
        state.P += P_lith - state.P_lith
    end
    state.P_lith = P_lith
    return state
end

"""
    magma_viscosity(ep::EruptionParams, T_K) -> η [Pa·s]

Melt viscosity at `T_K` [K] from `ep.melt_viscosity`. The ascending dike carries extracted
melt, not bulk mush, so this is the melt viscosity rather than a crystal-loaded suspension
viscosity.
"""
magma_viscosity(ep::EruptionParams, T_K) =
    compute_viscosity_εII(ep.melt_viscosity, 1.0, (; T = T_K))

"""
    max_ascent_length(ΔP, Δρg, η, ep) -> L [m]

How far a dike of half-width `ep.w_dike` travels before it freezes. A dike ascends at
`v = Δ w²/(3η)` for driving pressure gradient `Δ` and solidifies once its thermal boundary
layer reaches the wall, after `w²/κ`, so it survives a path `L` only while `L/v < w²/κ`.
Writing `Δ = ΔP/L + Δρg` — the chamber overpressure spread over the path, plus the
buoyancy gradient — the limiting length solves

    L² − (Δρg·c)·L − ΔP·c = 0,    c = w⁴/(3ηκ),

whose positive root this returns. The `w⁴` and the `1/η` are what separate a basalt dike,
which reaches the surface from any crustal depth, from a viscous silicic one that freezes
within a few km unless it is wide or water-rich.
"""
function max_ascent_length(ΔP, Δρg, η, ep::EruptionParams)
    η > 0 || throw(DomainError(η, "melt viscosity must be positive"))
    c = ep.w_dike^4 / (3 * η * ep.κ_magma)
    b = max(Δρg, zero(Δρg)) * c
    return 0.5 * (b + sqrt(b^2 + 4 * max(ΔP, zero(ΔP)) * c))
end

"""
    dike_ascends(ΔP, P_lith, ρ_magma, η_magma, z_centroid, ep) -> Bool

Propagation criterion: a dike leaving a chamber at depth `z_centroid` [m, negative down]
reaches the surface when the magma pressure surviving the ascent exceeds the barrier
stress,

    ΔP + P_lith − ρ_magma·g·|z_centroid| ≥ σ_barrier,

equivalently ``\\Delta P + \\int (\\rho_{crust} - \\rho_{magma}) g\\,dz``. The crustal column
enters through `P_lith` ([`lithostatic_pressure`](@ref)) and the magma column through
`ρ_magma` ([`mixture_density`](@ref)), so exsolved gas lightens the magma and eases ascent.

Buoyancy alone grows with depth and never forbids a deep eruption. The depth limit comes
from the second requirement: the dike must reach the surface before it freezes, so the
chamber can be no deeper than [`max_ascent_length`](@ref).
"""
function dike_ascends(ΔP, P_lith, ρ_magma, η_magma, z_centroid, ep::EruptionParams)
    all(isfinite, (ΔP, P_lith, ρ_magma, η_magma, z_centroid)) ||
        throw(ArgumentError("ΔP, P_lith, ρ_magma, η_magma, and z_centroid must be finite"))
    ρ_magma > 0 || throw(DomainError(ρ_magma, "ρ_magma must be positive"))
    L = abs(z_centroid)
    ΔP + P_lith - ρ_magma * ep.g * L >= ep.σ_barrier || return false
    L > 0 || return true
    return L <= max_ascent_length(ΔP, (P_lith - ρ_magma * ep.g * L) / L, η_magma, ep)
end

"""
    liquidus_temperature(melting; T_max=2273.15, tol=0.05) -> T_liq [K]

Lowest temperature at which the melting parameterisation `melting` is fully molten
(ϕ = 1), found by bisection on `[273.15, T_max]` K. Throws if `melting` never reaches
ϕ = 1 below `T_max`.
"""
function liquidus_temperature(melting; T_max = 2273.15, tol = 0.05)
    ϕ(T) = compute_meltfraction(melting, (; T))
    ϕ(T_max) >= 1 || throw(
        ArgumentError(
            "$(nameof(typeof(melting))) is not fully molten at $T_max K; it has no liquidus " *
                "in the range searched"
        )
    )
    T_lo, T_hi = 273.15, T_max
    while T_hi - T_lo > tol
        T_mid = 0.5 * (T_lo + T_hi)
        ϕ(T_mid) < 1 ? (T_lo = T_mid) : (T_hi = T_mid)
    end
    return T_hi
end

"""
    step_chamber_eruption!(rng, state, ep, T, rocks, tracers, ϕ, z, MatParam, Phases;
                           ȧ, Δt, time, closure, margin, T_background) -> (T, rocks, cargo, event)

Advance one model's lumped chamber over a thermal step, and realize an eruption once the
bulk magma it has drained is resolvable on the grid.

The connected mush `ϕ ≥ ep.ϕ_erupt` supplies the chamber's identity — its mean (T,ϕ) and
melt-weighted centroid — while pressure and volume are the chamber's own state
([`step_overpressure!`](@ref)). The shell relaxation viscosity is derived per call, so
several chambers may share one `ep` without writing through each other.

`T_background` is the *undisturbed* geotherm on the same grid as `T`, sampled at the
chamber centroid for the far-field temperature of the visco-elastic shell
([`crustal_relaxation_viscosity`](@ref)). Passing the evolving `T` instead would feed the
chamber's own aureole back in as its own boundary condition.

A connected region counts as a chamber only once it holds `ep.h_melt_min` of melt; below
that the column evolves thermally with no chamber at all.

Returns `event === nothing` on a step that only charges the chamber, or drains less magma
than the grid can resolve; `T` and `rocks` come back unchanged in that case, and the drained
magma stays queued on `state`. `cargo` is the erupted tracer population.
"""
function step_chamber_eruption!(
        rng, state::EruptionState, ep::EruptionParams,
        T, rocks, tracers, ϕ, z, MatParam, Phases;
        ȧ, Δt, time, closure, margin, T_background
    )
    length(T_background) == length(z) ||
        throw(DimensionMismatch("T_background and z must have equal length"))
    ind_e, V_e, zc = eruptible_mush(ϕ, z; ϕ_erupt = ep.ϕ_erupt)
    if V_e <= 0
        reset_chamber!(state)
        return T, rocks, empty(tracers), nothing
    end

    z_lo, z_hi = z[first(ind_e)], z[last(ind_e)]
    h_melt = melt_thickness(ϕ, z, z_lo, z_hi)
    # A body holding too little mobile liquid is not a chamber: it accumulates heat with no
    # pressure state and cannot erupt. Dropping back below the threshold ends the chamber
    # rather than pausing it — the next engaged step re-seeds (T,ϕ) and V, so the
    # fixed-P density rate is never differenced across a gap it did not integrate.
    if h_melt < ep.h_melt_min
        reset_chamber!(state)
        return T, rocks, empty(tracers), nothing
    end
    # Chamber identity: the mush must still overlap the interval the state was built on.
    # `eruptible_mush` returns whichever lens is currently largest, and a disconnected lens
    # taking that title is a different magma source — it starts its own pressure, mass, and
    # withdrawal history rather than inheriting one.
    state.init && (z_lo > state.z_hi || z_hi < state.z_lo) && reset_chamber!(state)
    state.init || (state.id += 1)
    state.z_lo, state.z_hi = z_lo, z_hi

    update_lithostatic!(state, lithostatic_pressure(ep, T, z, zc))
    T_mush_K = sum(i -> T[i], ind_e) / length(ind_e) + 273.15
    ϕ_mush = sum(i -> ϕ[i], ind_e) / length(ind_e)
    # the shell runs from the mush out to the undisturbed crust at the chamber's own depth;
    # that far-field temperature, not the hot wall, is what decides storing versus erupting
    T_far_K = linear_interpolation(z, T_background)(zc) + 273.15
    η_r = crustal_relaxation_viscosity(ep, T_mush_K, T_far_K)
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid = zc, η_r)

    Δz = z[2] - z[1]
    h_drained = state.h_erupt
    # the drained thickness is bulk magma, so the ceiling is the mush's bulk extent V_e -
    # the band the closure cuts from - not its liquid content
    h_erupt = pending_withdrawal!(state, h_drained, V_e, Δz; time)
    near_boundary = (z_lo - margin <= z[1]) || (z_hi + margin >= z[end])
    eruption_fires(; h_erupt, near_boundary, Δz) ||
        return T, rocks, empty(tracers), nothing

    aggregated = state.h_pending > h_drained + 64eps(max(h_erupt, 1.0))
    T_new, rocks_new, cargo, event = realize_eruption!(
        rng, T, rocks, tracers, ϕ, z,
        MatParam, Phases; realization_time = time, trigger_time = state.pending_since,
        chamber = state.id, h_requested = h_erupt, z_lo, z_hi,
        trigger = "D&H 3-phase", closure, aggregated, eligible_phase = nothing
    )
    # tracers that were not taken as cargo ride the host rock through the closure
    collapse_tracers!(tracers, (z_lo + z_hi) / 2, h_erupt; method = closure)
    commit_pending_withdrawal!(state, event.requested)
    return T_new, rocks_new, cargo, event
end

"""
    check_sill_temperature(melting, T_sill_C)

Verify that injected magma at `T_sill_C` [°C] is not superheated above the liquidus of the
melting parameterisation the model runs it through ([`liquidus_temperature`](@ref)). A
sill hotter than its own liquidus is a composition mismatch — the mush is being described
by a melting law that says it cannot exist as a crystal-bearing magma at that temperature,
and every downstream solubility, density and viscosity call is then an extrapolation.

Call once at model setup, where the melting law and the injection temperature meet.
"""
function check_sill_temperature(melting, T_sill_C)
    T_liq = liquidus_temperature(melting)
    T_sill_C + 273.15 <= T_liq || throw(
        ArgumentError(
            "sill temperature $T_sill_C °C exceeds the $(nameof(typeof(melting))) liquidus " *
                "$(round(T_liq - 273.15, digits = 1)) °C: the injected magma is superheated and the " *
                "melting law does not describe it"
        )
    )
    return T_liq
end
