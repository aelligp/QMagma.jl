using JLD2
using WriteVTK
using GeoParams

export export_thermal_structure, gaussian_thermal_structure, melt_fraction_from_temperature

"""
    gaussian_thermal_structure(profile, background, x; y=nothing, sigma, center=0)

Turn a vertical temperature `profile` into a 2D or 3D field by applying a Gaussian
horizontal taper to its anomaly from `background`. In 2D the distance is `x - center`;
in 3D it is radial in `x` and `y`, and `center` may be `(x0, y0)`.
"""
function gaussian_thermal_structure(profile, background, x; y=nothing, sigma, center=0)
    length(profile) == length(background) ||
        throw(DimensionMismatch("profile and background must have equal length"))
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    anomaly = profile .- background
    if y === nothing
        center isa Number || throw(ArgumentError("2D center must be a number"))
        weight = @. exp(-(x - center)^2 / (2sigma^2))
        return reshape(background, 1, :) .+ weight .* reshape(anomaly, 1, :)
    end
    center_xy = center isa Number ? (center, center) : center
    length(center_xy) == 2 || throw(ArgumentError("3D center must be a number or (x0, y0)"))
    weight = @. exp(-((x - center_xy[1])^2 + (y' - center_xy[2])^2) / (2sigma^2))
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
`(length(x), length(y), length(z))`. Use [`gaussian_thermal_structure`](@ref) to construct
a 2D or 3D temperature field from a 1D anomaly before export.

JLD2 stores the coordinates and expanded fields as top-level datasets. VTK stores 1D
profiles on a singleton horizontal axis so ParaView and other VTK readers can display
them. The returned vector contains every file written.

```julia
export_thermal_structure("run_1d", z; fields=(temperature=T, melt_fraction=phi))
x = -10e3:1e3:10e3
T2 = gaussian_thermal_structure(last_run[:T], last_run[:T_background], x; sigma=5e3)
T2_Q = gaussian_thermal_structure(last_run[:T_Qmagma], last_run[:T_background], x; sigma=5e3)
export_thermal_structure("run_2d", z; x, fields=(temperature=T2,))
export_thermal_structure("run_Qmagma_2d", z; x, fields=(temperature=T2_Q,))
```
"""
function export_thermal_structure(filename, z; x=nothing, y=nothing, fields,
                                  formats=(:jld2, :vtk))
    y === nothing || x !== nothing || throw(ArgumentError("y requires x"))
    coordinates = x === nothing ? (z=z,) :
                  y === nothing ? (x=x, z=z) : (x=x, y=y, z=z)
    for (name, coordinate) in pairs(coordinates)
        isempty(coordinate) && throw(ArgumentError("$name must not be empty"))
        all(diff(coordinate) .> 0) || throw(ArgumentError("$name must be strictly ascending"))
    end

    dimensions = Tuple(length(coordinate) for coordinate in values(coordinates))
    expanded = Dict{Symbol,Any}()
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
        payload = Dict{Symbol,Any}(:dimensionality => length(dimensions))
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
        append!(written, vtk_grid(basename, vtk_coordinates...) do vtk
            for (name, field) in vtk_fields
                vtk[string(name)] = field
            end
        end)
    end
    return written
end
