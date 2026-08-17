module QMagma

using GeoParams
using ForwardDiff, SparseArrays, SparseDiffTools, LinearAlgebra, Interpolations, Random
using ZirconGrowth
using JLD2, WriteVTK
using Statistics: std, quantile
using SpecialFunctions: erf

export sill_intrusion_1D, compute_zircon_ages, volume_averaged_age
export export_thermal_structure, lateral_thermal_structure, melt_fraction_from_temperature
export lateral_effective_area

# Physics and numerics, in dependency order.
include("grid.jl")
include("solver.jl")
include("flux.jl")
include("injection.jl")
include("eruption.jl")
include("chamber.jl")
include("budgets.jl")
include("tracers.jl")
include("export.jl")
include("controls.jl")

# Populated with the latest run's data at the end of each simulation, so it is reachable
# from the REPL: tracer T-t histories (`QMagma.tracers_out`, for ZirconGrowth.jl), tracers
# removed by eruption (`QMagma.erupted_tracers_out`), and 1D profiles / temporal evolution
# (`QMagma.last_run_out`)
tracers_out = Tracer[]
erupted_tracers_out = Tracer[]
last_run_out = Dict{Symbol, Any}()

"""
    sill_intrusion_1D(; size=nothing)

Interactive GLMakie App for 1D thermal intrusion model. `size` is the size of the window
in pixels; if `nothing` (default), it's chosen automatically to fit within the primary
monitor's available height, since a fixed pixel height can be taller than some screens
(clipping the bottom of the control panel) and there's no scrollable layout to fall back on.

The implementation lives in a package extension, so `GLMakie` must be loaded first.
"""
function sill_intrusion_1D(; kwargs...)
    ext = Base.get_extension(@__MODULE__, :QMagmaGLMakieExt)
    ext === nothing &&
        error("sill_intrusion_1D requires GLMakie; run `using GLMakie` first")
    return ext.sill_intrusion_1D(; kwargs...)
end

#! format: off
# Letter colors run the logo's melt gradient outward from a red core: the two
# central letters glow hottest, the outer pairs repeat as the cooled rim of a
# sill. 256-color codes, not truecolor, for terminals lacking 24-bit support.
function _print_banner(io::IO)
    q = "\e[38;5;221m"  # rim
    m = "\e[38;5;208m"
    a = "\e[38;5;196m"  # core
    g = "\e[38;5;160m"  # core
    m2 = "\e[38;5;208m" # rim, repeating q/m
    a2 = "\e[38;5;221m"
    res = "\e[0m"

    str = """
    $(q) ██████╗ $(m)███╗   ███╗$(a) █████╗ $(g) ██████╗ $(m2)███╗   ███╗$(a2) █████╗ $(res)
    $(q)██╔═══██╗$(m)████╗ ████║$(a)██╔══██╗$(g)██╔════╝ $(m2)████╗ ████║$(a2)██╔══██╗$(res)
    $(q)██║   ██║$(m)██╔████╔██║$(a)███████║$(g)██║  ███╗$(m2)██╔████╔██║$(a2)███████║$(res)
    $(q)██║▄▄ ██║$(m)██║╚██╔╝██║$(a)██╔══██║$(g)██║   ██║$(m2)██║╚██╔╝██║$(a2)██╔══██║$(res)
    $(q)╚██████╔╝$(m)██║ ╚═╝ ██║$(a)██║  ██║$(g)╚██████╔╝$(m2)██║ ╚═╝ ██║$(a2)██║  ██║$(res)
    $(q) ╚══▀▀═╝ $(m)╚═╝     ╚═╝$(a)╚═╝  ╚═╝$(g) ╚═════╝ $(m2)╚═╝     ╚═╝$(a2)╚═╝  ╚═╝$(res)
    """
    printstyled(io, "\n\n", str, "\nVersion: $(pkgversion(@__MODULE__))\n", bold = true, color = :default)
    return nothing
end
#! format: on

function __init__(io::IO = stdout)
    isa(stdout, Base.TTY) || return
    _print_banner(io)
    return nothing
end

end # module QMagma
