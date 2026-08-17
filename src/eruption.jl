# Eruption: finding the eruptible band, removing its melt, and closing the column.

function largest_contiguous_range(mask)
    best_len = 0
    best_lo = 0
    best_hi = 0
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
                best_lo = run_lo
                best_hi = j - 1
            end
            i = j
        else
            i += 1
        end
    end
    return best_len == 0 ? nothing : best_lo:best_hi
end

"""
    collapse_advection(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2)

Move every grid point's *position* toward the eruption zone `[Erupt_z0-half, Erupt_z0+half]`
using the same elastic decay law `insert_sill` uses to open a sill (largest right at the
band edge, decaying away from it with radius `R`), then interpolate the original field `T`
at those moved positions back onto the fixed grid `z`. This is the direct inverse of
`insert_sill`'s opening displacement: instead of pushing host rock apart to make room for
new material, it pulls host rock together to close a gap left by erupted material.

Points below the band move toward `Erupt_z0` by exactly enough to fully close their side
of the gap (not just asymptotically approach it - the naive `insert_sill`-style amplitude
falls fractionally short once evaluated at an actual grid point rather than the idealized
edge, which would otherwise leave a sliver of the band uncollapsed). Points inside the band
are swept toward `Erupt_z0` too, packed into a tiny window around it so the displaced grid
stays strictly monotonic (required for the interpolation) without colliding exactly on top
of each other.
"""
function collapse_advection(T, z; Erupt_z0, Erupt_thick, R = Erupt_thick / 2, method = :hybrid)
    zv = Vector(z)
    n = length(zv)
    half = Erupt_thick / 2

    Displ = collapse_displacement(zv; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, R = R, method = method)
    # the top boundary is unpinned (the surface subsides by Erupt_thick): flat
    # extrapolation fills the vacated cells above the subsided surface with the
    # boundary value (Ttop)
    itp = linear_interpolation(zv .+ Displ, Vector(T); extrapolation_bc = Flat())
    T_new = itp.(zv)

    # the handful of grid points exactly at the collapse center can still backtrace onto
    # their own pre-collapse value due to floating-point collisions in the interpolation;
    # patch them with a clean linear interpolation between the nearest already-correctly
    # -advected neighbors just outside the band
    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    if !isempty(ind) && ind[1] > 1 && ind[end] < n
        lo, hi = ind[1] - 1, ind[end] + 1
        T_new[ind] .= range(T_new[lo], T_new[hi]; length = length(ind) + 2)[2:(end - 1)]
    end

    return T_new
end

"""
    collapse_displacement(zv; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:hybrid)

Build the grid-point displacement field used by `collapse_advection` to close an
erupted band: elastic decay away from the band edges (`crack_perp_displacement`,
radius `R`), rescaled so the nearest exterior point on each side fully closes its
side of the gap, with the interior (band) points packed into a tiny strictly ordered
window around `Erupt_z0`.

The floor side rises elastically toward the vent, while the roof displacement transitions
from `-half` at the wall face to a rigid `-Erupt_thick` subsidence far above (transition
radius `Erupt_thick`, wide enough to keep the warped grid monotonic), so the free surface
sinks by the erupted thickness and the removed volume exits through the unpinned top
boundary. The compression paying for the floor-side stretch is thereby concentrated in the
roof just above the vent.
"""
function collapse_displacement(zv; Erupt_z0, Erupt_thick, R = Erupt_thick / 2, method = :hybrid)
    method === :hybrid ||
        throw(ArgumentError("collapse_displacement supports only :hybrid, got $method"))
    half = Erupt_thick / 2
    z_shift = zv .- Erupt_z0
    Displ = zero(z_shift)

    id_above = findall(z_shift .> half)
    id_below = findall(z_shift .< -half)
    id_inside = findall(abs.(z_shift) .<= half)

    dist_above = z_shift[id_above] .- half   # distance outward from the idealized band edge
    dist_below = -z_shift[id_below] .- half

    # anchor distance: how far the nearest grid point below the band must move to fully
    # reach Erupt_z0. It rescales the decay law so that point's gap closes completely.
    anchor_below = isempty(id_below) ? half : -z_shift[id_below[end]]
    scale_below = anchor_below / crack_perp_displacement(0.0, half; r = R)

    # roof: from -half at the face (walls meet at Erupt_z0) to -Erupt_thick far
    # above (rigid caldera subsidence of the whole overburden)
    Displ[id_above] .= -(Erupt_thick .- crack_perp_displacement(dist_above, half; r = Erupt_thick))
    Displ[id_below] .= scale_below .* crack_perp_displacement(dist_below, half; r = R)

    # every point that ends up inside (or right at the edge of) the old band's footprint -
    # both the exterior anchor points (which land exactly on Erupt_z0) and the interior
    # (melt-zone) points - gets packed into a strictly ordered, tiny window around Erupt_z0,
    # so the displaced grid stays strictly monotonic everywhere (required by the
    # interpolation) without any two points colliding on the exact same position
    id_band = sort(vcat(id_inside, isempty(id_above) ? Int[] : id_above[1], isempty(id_below) ? Int[] : id_below[end]))
    unique!(id_band)
    if !isempty(id_band)
        ε = 1.0e-3 * minimum(diff(zv))
        order = sortperm(z_shift[id_band])   # ascending z_shift == ascending zv on id_band
        targets = length(id_band) == 1 ? [0.0] : collect(range(-ε, ε; length = length(id_band)))
        Displ[id_band[order]] .= targets .- z_shift[id_band[order]]
    end

    # pin the bottom boundary (fixed BC) so the warped grid never contracts past it. The
    # top is left free: the surface subsides and collapse_advection fills the vacated
    # cells by extrapolation.
    Displ[1] = 0.0

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

- `:hybrid` - elastic floor, subsiding roof: the floor rises elastically toward the
  vent while the roof face drops to meet it and the roof displacement transitions to a
  rigid `-Erupt_thick` subsidence away from the vent, so the free surface sinks by the
  erupted thickness and the removed volume exits through the top. Deformation stays
  concentrated near the vent, temperatures are transported as intensive values (no
  dilution cooling), and the heat debit is approximately the erupted band's content —
  exact when the near-vent material is locally uniform, because the floor-side stretch
  duplication is then paid by compression of equally hot roof just above the vent.
  Stacked sills make the near-vent column non-uniform, so the residual is sensitive to
  what sits over the vent in a way `:caldera`'s is not.
"""
function erupt_melt!(T, rocks, z; Erupt_z0, Erupt_thick, method = :caldera)
    method in (:caldera, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    half = Erupt_thick / 2
    zv = Vector(z)
    n = length(zv)
    magma_out(rock_new) = nonnegative_debit(
        integrated_content(rocks, zv),
        integrated_content(rock_new, zv), "injected-magma content"; ncells = n
    )

    if method == :hybrid
        T_new = collapse_advection(Vector(T), zv; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = method)
        # advect the grey phase indicator conservatively with the SAME displacement
        # collapse_advection applies to T, not round(collapse_advection(rocks)): the latter
        # interpolates-then-rounds the 0/1 field under the converging closure, which inflates
        # the grey where the walls stretch (e.g. 700 m -> 1700 m) or annihilates it near the
        # vent (700 m -> 0), and lands it off the co-moving temperature. The grey inside the
        # erupted band leaves with the eruption (zeroed before the remap, else collapse_
        # displacement packs it into one cell as a spike); the surviving grey is remapped
        # conservatively and co-moves with T. Field stays fractional, as after insert_sill.
        Displ = collapse_displacement(zv; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = method)
        src = copy(Vector(rocks))
        src[findall(abs.(zv .- Erupt_z0) .<= half)] .= 0.0   # erupted band grey leaves the system
        rock_new = conservative_advection(src, Displ, zv)
        return T_new, rock_new, magma_out(rock_new)
    end

    T_new = copy(Vector(T))
    rock_new = copy(Vector(rocks))

    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    isempty(ind) && return T_new, rock_new, 0.0

    n_band = length(ind)
    i0 = ind[1]
    T_new[i0:(n - n_band)] .= T_new[(i0 + n_band):n]      # roof block drops onto the floor
    rock_new[i0:(n - n_band)] .= rock_new[(i0 + n_band):n]
    T_new[(n - n_band + 1):n] .= T_new[n]               # subsided surface, filled at Ttop
    rock_new[(n - n_band + 1):n] .= 0.0

    return T_new, rock_new, magma_out(rock_new)
end

"""
    melt_thickness(ϕ, z, z_lo, z_hi)

Melt content `∫ϕ dz` [m] of the depth interval `[z_lo, z_hi]` - the dense-rock
equivalent thickness of the liquid held in that band.

This is a storage diagnostic, not the erupted amount. An eruption withdraws the bulk
mixture, crystals included, so the closure amplitude and the erupted volume
`lateral_effective_area(R_sill) * EruptionEvent.requested` are bulk thicknesses; the liquid fraction of one
withdrawal is `EruptionEvent.melt_requested`.
"""
function melt_thickness(ϕ, z, z_lo, z_hi)
    return integrated_content(ϕ, z, z_lo, z_hi)
end
"""
    erupt_displacement(zm, Erupt_z0, Erupt_thick; method=:caldera, R=Erupt_thick/2)

New position of a material point at depth `zm` under the eruption closure of
`erupt_melt!`, for markers and tracers riding on the host rock. `:caldera`: points
above the erupted band drop rigidly by `Erupt_thick`, points inside land on the
chamber floor, points below stay put. `:hybrid`: points inside the band collapse onto
`Erupt_z0`, points below move toward it with the same elastic decay law as the host
rock in `collapse_advection`, and above it the drop transitions from `Erupt_thick/2`
at the wall face to the full rigid `Erupt_thick` subsidence far above the vent.
"""
function erupt_displacement(zm, Erupt_z0, Erupt_thick; method = :caldera, R = Erupt_thick / 2)
    method in (:caldera, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    half = Erupt_thick / 2
    s = zm - Erupt_z0
    if method == :hybrid
        abs(s) <= half && return Erupt_z0
        s < -half && return zm + crack_perp_displacement(-s - half, half; r = R)
        return zm - (Erupt_thick - crack_perp_displacement(s - half, half; r = Erupt_thick))
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
function collapse_markers!(markers, Erupt_z0, Erupt_thick; method = :caldera)
    for (i, zm) in enumerate(markers)
        markers[i] = erupt_displacement(zm, Erupt_z0, Erupt_thick; method = method)
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
function collapse_tracers!(tracers, Erupt_z0, Erupt_thick; method = :caldera)
    for tracer in tracers
        tracer.z = erupt_displacement(tracer.z, Erupt_z0, Erupt_thick; method = method)
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
cannot be ignored accidentally. The caller owns `rng`; stochastic tests should pass a
seeded generator. When all phases are eligible, the tracer weights are normalized to the
full melt content of the interval so empty tracer cells do not make the sampled cargo
underrepresent the withdrawn state.

Returns `(erupted, h_cargo, quantum)`. `h_cargo` is the melt thickness represented by the
selected tracers. `quantum` is the largest single tracer weight, which bounds how finely
this population can resolve a withdrawal: whole tracers are selected, so `h_cargo` can
miss an arbitrary `h_erupt` by up to `quantum` — half that once two or more tracers are
selected and the nearest-whole-tracer correction has a candidate to drop, but the full
quantum when a single tracer is all there is. A sparsely seeded band has a large
`quantum` — one tracer may stand for many cells' melt — and callers that check cargo
against the withdrawn state must scale their tolerance by it.
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

    isempty(candidate_indices) && return tracers[Int[]], 0.0, 0.0
    widths = interval_cell_widths(z, z_lo, z_hi)
    weights = [ϕ[i] * widths[i] / counts[i] for i in cell_indices]
    if isnothing(eligible_phase)
        weights .*= melt_thickness(ϕ, z, z_lo, z_hi) / sum(weights)
    end
    total = sum(weights)
    quantum = maximum(weights)
    target = min(h_erupt, total)
    target > 0 || return tracers[Int[]], 0.0, quantum

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
    return erupted, h_cargo, quantum
end

"""
    EruptionEvent(; ...)

One realized eruption, its bulk thickness `requested` from the chamber, and the cargo the
tracer population independently represents. `chamber` is the generation counter of the
[`EruptionState`](@ref) that erupted (`EruptionState.id`), so a record spanning the death
of one magma body and the birth of another can be split by source. Separate trigger and realization times preserve
the delay when sub-grid drainage aggregates.

`requested` is bulk magma — liquid and the crystals suspended in it — while the tracers
sample melt, so `melt_requested` is the liquid content of that bulk withdrawal and is what
`cargo_represented` is checked against. It defaults to `requested`, the fully molten limit,
and may never exceed it. Construction fails unless the two match to the
declared tracer-resolution tolerance `cargo_atol`: the tracers are sampled from the mush by
a separate mechanism from the thermal withdrawal, so this is the one account that can
disagree and therefore the one worth checking.

`magma_removed` is the intruded-magma content the closure took out of the column.
`melt_removed` is the measured change in full-column melt storage across the instantaneous
closure. It can differ from `requested` because the fixed-grid closure transports
temperature intensively; this diagnostic exposes that mismatch instead of relabeling the
requested closure amplitude as a measured state withdrawal.
"""
struct EruptionEvent
    chamber::Int
    trigger_time::Float64
    realization_time::Float64
    requested::Float64
    melt_requested::Float64
    cargo_represented::Float64
    z_lo::Float64
    z_hi::Float64
    z_centroid::Float64
    trigger::Symbol
    closure::Symbol
    aggregated::Bool
    cargo_count::Int
    enthalpy_before::Float64
    enthalpy_after::Float64
    erupted_enthalpy::Float64
    enthalpy_residual::Float64
    magma_removed::Float64
    melt_removed::Float64
end

function EruptionEvent(;
        chamber = 0, trigger_time, realization_time, requested,
        melt_requested = requested, cargo_represented,
        z_lo, z_hi, z_centroid, trigger, closure, aggregated = false,
        cargo_count = 0, enthalpy_before, enthalpy_after, erupted_enthalpy,
        magma_removed = 0.0, melt_removed = 0.0, cargo_atol = 0.0
    )
    thicknesses = (requested, melt_requested, cargo_represented)
    all(isfinite, thicknesses) || throw(ArgumentError("eruption thicknesses must be finite"))
    all(x -> x >= 0, thicknesses) ||
        throw(ArgumentError("eruption thicknesses must be nonnegative"))
    melt_requested <= requested + 64eps(max(requested, 1.0)) || throw(
        ArgumentError(
            "erupted melt $melt_requested m exceeds the bulk withdrawal $requested m"
        )
    )
    cargo_atol >= 0 || throw(ArgumentError("cargo_atol must be nonnegative"))
    all(
        isfinite, (
            trigger_time, realization_time, z_lo, z_hi, z_centroid,
            enthalpy_before, enthalpy_after, erupted_enthalpy,
            magma_removed, melt_removed,
        )
    ) ||
        throw(ArgumentError("eruption event values must be finite"))
    magma_removed >= 0 || throw(ArgumentError("magma_removed must be nonnegative"))
    cargo_count >= 0 || throw(ArgumentError("cargo_count must be nonnegative"))
    atol = 64eps(max(maximum(thicknesses), 1.0))
    isapprox(cargo_represented, melt_requested; rtol = 0, atol = cargo_atol + atol) ||
        throw(
        ArgumentError(
            "eruption cargo mismatch: melt-requested=$melt_requested, " *
                "cargo-represented=$cargo_represented (cargo tolerance=$cargo_atol)"
        )
    )
    closure in (:caldera, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $closure"))
    Hres = enthalpy_after - enthalpy_before + erupted_enthalpy
    trigger_time <= realization_time ||
        throw(ArgumentError("trigger_time must not exceed realization_time"))
    return EruptionEvent(
        chamber, trigger_time, realization_time, requested, melt_requested,
        cargo_represented, z_lo, z_hi, z_centroid, Symbol(trigger), closure,
        aggregated, cargo_count, enthalpy_before, enthalpy_after, erupted_enthalpy, Hres,
        magma_removed, melt_removed
    )
end

"""
    realize_eruption!(rng, T, rocks, tracers, ϕ, z, MatParam, Phases; ...) ->
        (T_new, rocks_new, cargo, event)

Withdraw the thermal/rock state and represented tracer cargo as one operation, then
construct the fail-fast [`EruptionEvent`](@ref).

`h_requested` is a *bulk* thickness: the closure removes a band of magma, liquid and
suspended crystals together. The tracers carry melt, so they are sampled to the liquid
content of that band, `h_requested` times the interval's width-averaged melt fraction.

`cargo_atol` is the floor on the cargo-vs-state tolerance, half a grid cell by default.
The tolerance actually applied is the larger of that and the sampler's quantum
([`extract_erupted_tracers!`](@ref)), because whole tracers are selected: a band holding
few eligible tracers resolves a withdrawal only to the melt one tracer stands for, which
is many cells' worth when seeding is sparse. Tightening this to `cargo_atol` alone would
reject sound eruptions from thinly seeded mush.
"""
function realize_eruption!(
        rng, T, rocks, tracers, ϕ, z, MatParam, Phases;
        realization_time, trigger_time = realization_time,
        chamber = 0, h_requested, z_lo, z_hi,
        trigger, closure, aggregated = false, eligible_phase,
        cargo_atol = (z[2] - z[1]) / 2
    )
    h_requested > 0 || throw(ArgumentError("h_requested must be positive"))
    z_hi > z_lo || throw(ArgumentError("z_hi must be greater than z_lo"))
    widths = interval_cell_widths(z, z_lo, z_hi)
    weights = ϕ .* widths
    ind = findall(z_lo .<= z .<= z_hi)
    h_melt = sum(weights[ind])
    h_melt > 0 || throw(ArgumentError("eruptible interval contains no melt"))
    z_centroid = sum(weights[ind] .* z[ind]) / h_melt
    # liquid content of the bulk withdrawal, at the interval's width-averaged melt fraction
    h_melt_requested = h_requested * h_melt / (z_hi - z_lo)
    H_before = column_enthalpy(T, z, MatParam, Phases)
    H_erupted = erupted_bulk_enthalpy(T, ϕ, z, z_lo, z_hi, h_requested, MatParam, Phases)
    T_new, rocks_new, magma_removed = erupt_melt!(
        T, rocks, z;
        Erupt_z0 = (z_lo + z_hi) / 2, Erupt_thick = h_requested, method = closure
    )
    cargo, h_cargo, quantum = extract_erupted_tracers!(
        rng, tracers, ϕ, z, z_lo, z_hi,
        h_melt_requested; eligible_phase
    )
    H_after = column_enthalpy(T_new, z, MatParam, Phases)
    ϕ_after = similar(ϕ)
    compute_meltfraction!(ϕ_after, MatParam, Phases, (T = T_new .+ 273.15,))
    melt_removed = integrated_content(ϕ, z) - integrated_content(ϕ_after, z)
    event = EruptionEvent(;
        chamber, trigger_time, realization_time, requested = h_requested,
        melt_requested = h_melt_requested, cargo_represented = h_cargo,
        z_lo, z_hi, z_centroid, trigger, closure, aggregated,
        cargo_count = length(cargo), enthalpy_before = H_before, enthalpy_after = H_after,
        erupted_enthalpy = H_erupted, magma_removed, melt_removed,
        cargo_atol = max(cargo_atol, quantum)
    )
    return T_new, rocks_new, cargo, event
end
