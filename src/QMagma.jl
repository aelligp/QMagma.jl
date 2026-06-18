module QMagma

# 1D thermal sill-intrusion solver together with its interactive GLMakie GUI.
# GLMakie is a direct dependency of QMagma, so `sill_intrusion_1D` is available
# as soon as the package is loaded - no need to separately load GLMakie.
#
# `ThermalCode_1D_GLMakie.jl` itself `using`s GLMakie, includes the solver
# (`ThermalCode_1D.jl`) and exports `sill_intrusion_1D`.
include("ThermalCode_1D_GLMakie.jl")

end # module QMagma
