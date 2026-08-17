# Writing model output to JLD2 and VTK, including 2D/3D expansion of 1D profiles.

"""
    lateral_profile(r, R)

Fraction of the axial thermal anomaly carried at lateral distance `r` from a body of radius
`R`: `√(1 - r²/R²)`, the opening of a penny-shaped crack under uniform internal pressure
(Sneddon 1946).

Compact support: zero at `r ≥ R`, so the body has an edge, not a tail. `R = Inf` is
laterally uniform.

Sets both the lateral heat loss ([`lateral_loss_coefficient`](@ref)) and the 2-D/3-D
expansion ([`lateral_thermal_structure`](@ref)), so the body whose cooling the column pays
for is the body the export draws.

It is also the body [`crack_perp_displacement`](@ref) opens, so the emplacement kinematics
displace `lateral_effective_area(R)` per unit axial aperture.
"""
function lateral_profile(r, R)
    isnan(r) && return oftype(r / R, NaN)   # a NaN must not be absorbed into the zero branch
    return abs(r) < R ? sqrt(1 - (r / R)^2) : zero(r / R)
end

"""
    lateral_effective_area(R) -> A [m²]

Plan-view area `(2/3)πR²` of the body [`lateral_profile`](@ref) describes, `∫w(r) 2πr dr`.

Converts the column's axial thicknesses - erupted magma, chamber volume - into volumes. It
is smaller than `πR²` because a crack of radius `R` is thickest on the axis and thins to
nothing at the tip; the column tracks the axis.
"""
lateral_effective_area(R) = 2 * π * R^2 / 3

"""
    lateral_loss_coefficient(k, R) -> Hlat [W/m³/K]

Coefficient `2k/R²` of the residual's lateral sink `Hlat(T - T_bg)`, for conductivity `k`
and body radius `R`: `-k` times the radial Laplacian of [`lateral_profile`](@ref) on the
axis.

`R = Inf` gives zero, a purely 1-D model.
"""
lateral_loss_coefficient(k, R) = 2k / R^2

"""
    lateral_thermal_structure(profile, background, x; y=nothing, R, center=0)

Turn a vertical temperature `profile` into a 2D or 3D field by tapering its anomaly from
`background` with [`lateral_profile`](@ref) over a body of radius `R`.

In 2D the distance is `x - center`; in 3D it is radial in `x` and `y`, and `center` may be
`(x0, y0)`. Outside `R` the field is `background`, so the body has an edge and the grid may
extend past it without a tail to truncate.
"""
function lateral_thermal_structure(profile, background, x; y = nothing, R, center = 0)
    length(profile) == length(background) ||
        throw(DimensionMismatch("profile and background must have equal length"))
    R > 0 || throw(ArgumentError("R must be positive"))
    anomaly = profile .- background
    if y === nothing
        center isa Number || throw(ArgumentError("2D center must be a number"))
        weight = @. lateral_profile(x - center, R)
        return reshape(background, 1, :) .+ weight .* reshape(anomaly, 1, :)
    end
    center_xy = center isa Number ? (center, center) : center
    length(center_xy) == 2 || throw(ArgumentError("3D center must be a number or (x0, y0)"))
    weight = @. lateral_profile(sqrt((x - center_xy[1])^2 + (y' - center_xy[2])^2), R)
    return reshape(background, 1, 1, :) .+
        reshape(weight, length(x), length(y), 1) .* reshape(anomaly, 1, 1, :)
end

"""
    melt_fraction_from_temperature(T, MatParam)

Evaluate the model's melt law on a temperature field in °C. The result has the same
shape as `T` and assumes the model's sole material phase (phase 0).
"""
function melt_fraction_from_temperature(T, MatParam)
    ϕ = similar(T)
    compute_meltfraction!(ϕ, MatParam, fill(0, size(T)), (T = T .+ 273.15,))
    return ϕ
end

"""
    export_thermal_structure(filename, z; x=nothing, y=nothing, fields,
                             formats=(:jld2, :vtk))

Export point fields on a rectilinear 1D, 2D, or 3D grid. `fields` is a named tuple or
dictionary whose arrays match `(length(z),)`, `(length(x), length(z))`, or
`(length(x), length(y), length(z))`. Use [`lateral_thermal_structure`](@ref) to construct
a 2D or 3D temperature field from a 1D anomaly before export.

JLD2 stores the coordinates and expanded fields as top-level datasets. VTK stores 1D
profiles on a singleton horizontal axis so ParaView and other VTK readers can display
them. The returned vector contains every file written.

```julia
export_thermal_structure("run_1d", z; fields=(temperature=T, melt_fraction=phi))
x = -10e3:1e3:10e3
T2 = lateral_thermal_structure(last_run[:T], last_run[:T_background], x; R=5e3)
T2_Q = lateral_thermal_structure(last_run[:T_Qmagma], last_run[:T_background], x; R=5e3)
export_thermal_structure("run_2d", z; x, fields=(temperature=T2,))
export_thermal_structure("run_Qmagma_2d", z; x, fields=(temperature=T2_Q,))
```
"""
function export_thermal_structure(
        filename, z; x = nothing, y = nothing, fields,
        formats = (:jld2, :vtk)
    )
    y === nothing || x !== nothing || throw(ArgumentError("y requires x"))
    coordinates = x === nothing ? (z = z,) :
        y === nothing ? (x = x, z = z) : (x = x, y = y, z = z)
    for (name, coordinate) in pairs(coordinates)
        isempty(coordinate) && throw(ArgumentError("$name must not be empty"))
        all(diff(coordinate) .> 0) || throw(ArgumentError("$name must be strictly ascending"))
    end

    dimensions = Tuple(length(coordinate) for coordinate in values(coordinates))
    expanded = Dict{Symbol, Any}()
    for (name, field) in pairs(fields)
        field isa AbstractArray || throw(ArgumentError("field $name must be an array"))
        size(field) == dimensions ||
            throw(DimensionMismatch("field $name has size $(size(field)); expected $dimensions"))
        expanded[Symbol(name)] = field
    end
    isempty(expanded) && throw(ArgumentError("fields must not be empty"))

    requested = Tuple(formats)
    all(format -> format in (:jld2, :vtk), requested) ||
        throw(ArgumentError("formats may contain only :jld2 and :vtk"))
    base, extension = splitext(filename)
    basename = extension in (".jld2", ".vtk", ".vti", ".vtr", ".vts") ? base : filename
    written = String[]

    if :jld2 in requested
        payload = Dict{Symbol, Any}(:dimensionality => length(dimensions))
        merge!(payload, Dict(pairs(coordinates)), expanded)
        path = basename * ".jld2"
        jldsave(path; payload...)
        push!(written, path)
    end
    if :vtk in requested
        coordinate_type = promote_type(map(eltype, values(coordinates))...)
        vtk_coordinates = x === nothing ?
            ([zero(coordinate_type)], coordinate_type.(z)) :
            Tuple(coordinate_type.(coordinate) for coordinate in values(coordinates))
        vtk_fields = x === nothing ? Dict(name => reshape(field, 1, length(z)) for (name, field) in expanded) : expanded
        append!(
            written, vtk_grid(basename, vtk_coordinates...) do vtk
                for (name, field) in vtk_fields
                    vtk[string(name)] = field
                end
            end
        )
    end
    return written
end
