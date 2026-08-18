# Cumulative enthalpy and mass budgets used to audit the timestep loop.

"""
    column_enthalpy(T, z, MatParam, Phases) -> H  [J/m²]

Total heat content of the column per unit area, `∫ ρ(cₚT + Lϕ) dz`: sensible heat plus
the latent heat stored in the melt fraction. Properties are (re)evaluated from the
passed `T` so a before/after pair is self-consistent.

This is the diagnostic that separates the two eruption closures: `:caldera` translates the
roof rigidly, so `H` drops by exactly the erupted band's content, while `:hybrid` transports
`T` *intensively* (the elastically raised floor re-covers the vent at its own temperature)
and so conserves energy only approximately, exactly to the extent that the near-vent column
is uniform. Measure `H` before and after `erupt_melt!` to quantify that drift before
trusting reservoir Tt-paths / zircon spectra from a `:hybrid` run.
"""
function column_enthalpy(T, z, MatParam, Phases)
    args = (T = T .+ 273.15,)
    ρ = similar(T); Cp = similar(T); Hl = similar(T); ϕ = similar(T)
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
    return (k[end] * (T[end] - T[end - 1]) - k[1] * (T[2] - T[1])) / Δz * Δt
end

"""
    lateral_loss_energy(T, Params, z, Δt) -> E [J/m²]

Net heat the lateral (third-dimension) conduction term exchanges with the far field during
one timestep, over the same interior nodes as the thermal residual.

Negative values cool the column; identically zero for a purely 1-D model (`R_lat = Inf`).
Sign convention matches [`conductive_boundary_energy`](@ref), so the two add.
"""
function lateral_loss_energy(T, Params, z, Δt)
    length(T) == length(z) || throw(DimensionMismatch("T and z must have equal length"))
    Δt >= 0 || throw(ArgumentError("Δt must be nonnegative"))
    I = 2:(length(z) - 1)
    return -sum(Params.Hlat[I] .* (T[I] .- Params.T_bg[I])) * (z[2] - z[1]) * Δt
end

"""
    source_energy(Q, z, Δt) -> E [J/m²]

Heat supplied by the volumetric source during one timestep, using the same interior
nodes as the thermal residual.
"""
function source_energy(Q, z, Δt)
    length(Q) == length(z) || throw(DimensionMismatch("Q and z must have equal length"))
    Δt >= 0 || throw(ArgumentError("Δt must be nonnegative"))
    return sum(@view Q[2:(end - 1)]) * (z[2] - z[1]) * Δt
end

"""
    magma_heat_input(T_host, Tsill, h, MatParam; phase=0) -> E [J/m²]

Sensible plus latent heat supplied by `h` meters of initially liquid magma relative
to host material at `T_host`. This is the discrete-sill counterpart of the integrand
used by [`compute_Q_magma!`](@ref), and shares its density approximation: the magma is
weighted by the host-rock thermal density of `MatParam` at `T_host`, not by
`EruptionParams.ρ_melt`.
"""
function magma_heat_input(T_host, Tsill, h, MatParam; phase = 0)
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
    erupted_bulk_enthalpy(T, ϕ, z, z_lo, z_hi, h, MatParam, Phases) -> E [J/m²]

Enthalpy carried by `h` meters of bulk magma withdrawn from the eruptible interval. The
erupted parcel is the multi-phase mixture — liquid plus the crystals suspended in it — so
the integrand is the one [`column_enthalpy`](@ref) integrates, `ρ(cₚT + Lϕ)`, averaged over
the interval's control volumes by width alone. Latent heat therefore rides on the melt
fraction rather than on the whole parcel, and temperature uses the same 0 °C reference.

Weighting by `ϕΔz` instead would price `h` meters of *liquid*, charging the whole parcel
the latent heat only its melt fraction carries: an overstatement of `1 - ϕ̄` times the
latent share of the total, reaching ~25 % for a crystal-rich mush. Matching
`column_enthalpy` leaves [`EruptionEvent`](@ref)`.enthalpy_residual` reporting the
closure's transport error alone, with no phase-weighting offset underneath it.
"""
function erupted_bulk_enthalpy(T, ϕ, z, z_lo, z_hi, h, MatParam, Phases)
    h >= 0 || throw(ArgumentError("h must be nonnegative"))
    length(T) == length(ϕ) == length(z) == length(Phases) ||
        throw(DimensionMismatch("T, ϕ, z, and Phases must have equal length"))
    ind = findall(z_lo .<= z .<= z_hi)
    isempty(ind) && return 0.0
    edges = grid_cell_edges(z)
    weights = [max(0.0, min(edges[i + 1], z_hi) - max(edges[i], z_lo)) for i in ind]
    total = sum(weights)
    total > 0 || return 0.0
    args = (T = T .+ 273.15,)
    ρ = similar(T); Cp = similar(T); Hl = similar(T)
    compute_density!(ρ, MatParam, Phases, args)
    compute_heatcapacity!(Cp, MatParam, Phases, args)
    compute_latent_heat!(Hl, MatParam, Phases, args)
    return h * sum(weights .* ρ[ind] .* (Cp[ind] .* T[ind] .+ Hl[ind] .* ϕ[ind])) / total
end

"""
    EnthalpyBudget(initial_storage)

Cumulative column-energy diagnostic per unit area. `residual` is

`storage - initial_storage - boundary - injected - source + erupted`.

It deliberately exposes heat created or retained by intensive temperature remapping;
it does not correct transport.
"""
Base.@kwdef mutable struct EnthalpyBudget
    initial_storage::Float64
    storage::Float64
    boundary::Float64 = 0.0
    injected::Float64 = 0.0
    source::Float64 = 0.0
    erupted::Float64 = 0.0
    residual::Float64 = 0.0
end

EnthalpyBudget(initial_storage) = EnthalpyBudget(; initial_storage, storage = initial_storage)

"""
    update_enthalpy_budget!(budget, storage; boundary=0, injected=0, source=0, erupted=0)

Record the current column storage and accumulate the energy exchanged during one timestep,
refreshing `budget.residual` (see [`EnthalpyBudget`](@ref)).
"""
function update_enthalpy_budget!(
        budget::EnthalpyBudget, storage;
        boundary = 0.0, injected = 0.0, source = 0.0, erupted = 0.0
    )
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
    storage = budget.storage,
    storage_change = budget.storage - budget.initial_storage,
    boundary = budget.boundary,
    injected = budget.injected,
    source = budget.source,
    erupted = budget.erupted,
    residual = budget.residual,
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
    initial_magma::Float64
    initial_melt::Float64
    magma::Float64
    melt::Float64
    injected::Float64 = 0.0
    withdrawn::Float64 = 0.0
    erupted::Float64 = 0.0
    residual::Float64 = 0.0
    melt_residual::Float64 = 0.0
end

MassBudget(magma, melt) = MassBudget(; initial_magma = magma, initial_melt = melt, magma, melt)

"""
    update_mass_budget!(budget, magma, melt; injected=0, withdrawn=0, erupted=0)

Record the current intruded-magma content `magma` [m] and stored melt `melt` [m] and
accumulate the thicknesses exchanged during one timestep, refreshing `budget.residual` and
`budget.melt_residual` (see [`MassBudget`](@ref)).
"""
function update_mass_budget!(
        budget::MassBudget, magma, melt;
        injected = 0.0, withdrawn = 0.0, erupted = 0.0
    )
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
    magma = budget.magma,
    melt = budget.melt,
    magma_change = budget.magma - budget.initial_magma,
    melt_change = budget.melt - budget.initial_melt,
    injected = budget.injected,
    withdrawn = budget.withdrawn,
    erupted = budget.erupted,
    residual = budget.residual,
    melt_residual = budget.melt_residual,
)
