# Node-centered control volumes on the 1D grid, and the integrated quantities built on them.

const SecYear = 3600 * 24 * 365.25

av(x) = (x[2:end] + x[1:(end - 1)]) / 2

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
    edges[2:(end - 1)] .= av(zv)
    edges[end] = zv[end]
    return edges
end

function interval_cell_widths(z, z_lo = first(z), z_hi = last(z))
    edges = grid_cell_edges(z)
    first(z) <= z_lo <= z_hi <= last(z) || throw(
        ArgumentError(
            "integration interval [$z_lo, $z_hi] must lie inside [$(first(z)), $(last(z))]"
        )
    )
    return [
        max(0.0, min(edges[i + 1], z_hi) - max(edges[i], z_lo))
            for i in eachindex(z)
    ]
end

"""
    integrated_content(field, z[, z_lo, z_hi])

Integrate a node-centered control-volume field over the physical grid domain or a
clipped depth interval. This gives boundary nodes half-cell weight on a uniform grid,
so a constant field integrates to `z_hi - z_lo` rather than one extra grid spacing.
"""
function integrated_content(field, z, z_lo = first(z), z_hi = last(z))
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
    first(z) <= z_lo < z_hi <= last(z) || throw(
        ArgumentError(
            "addition interval [$z_lo, $z_hi] must lie inside [$(first(z)), $(last(z))]"
        )
    )
    amount >= 0 || throw(ArgumentError("amount must be nonnegative"))
    edges = grid_cell_edges(z)
    concentration = amount / (z_hi - z_lo)
    for i in eachindex(field)
        overlap = min(edges[i + 1], z_hi) - max(edges[i], z_lo)
        overlap > 0 && (field[i] += concentration * overlap / (edges[i + 1] - edges[i]))
    end
    return field
end

"""
    nonnegative_debit(before, after, label; ncells=1) -> debit

Content removed by a remap that can only lose material, clamped at zero. A remap that
*gains* more than roundoff is a real conservation failure and throws.

`before` and `after` come from `integrated_content`, a sum over `ncells` control volumes, so
the roundoff floor grows with the number of terms in that sum rather than being a fixed
multiple of `eps`. Pass `ncells`: a 2000-cell column accumulates thousands of times the
per-term error, which a fixed factor rejects as a leak. The resulting tolerance is still
around 1e-12 relative — orders below any transport error worth catching.
"""
function nonnegative_debit(before, after, label; ncells = 1)
    debit = before - after
    tolerance = 8 * max(ncells, 1) * eps(max(abs(before), abs(after), 1.0))
    debit >= 0 && return debit
    debit >= -tolerance && return 0.0
    throw(
        ArgumentError(
            "$label increased by $(-debit) during an outflow-only remap " *
                "(tolerance $tolerance over $ncells cells)"
        )
    )
end
