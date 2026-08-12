module QMagma

# 1D thermal sill-intrusion solver together with its interactive GLMakie GUI.
# GLMakie is a direct dependency of QMagma, so `sill_intrusion_1D` is available
# as soon as the package is loaded - no need to separately load GLMakie.
#
# `ThermalCode_1D_GLMakie.jl` itself `using`s GLMakie, includes the solver
# (`ThermalCode_1D.jl`) and exports `sill_intrusion_1D`.
include("ThermalCode_1D_GLMakie.jl")

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
    isa(stdout, Base.TTY) || return nothing
    _print_banner(io)
    return nothing
end

end # module QMagma
