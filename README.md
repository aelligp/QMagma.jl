# QMagma.jl

Interactive 1D sill injection model with source term and zircon growth.

This is a stand-alone package: the interactive [GLMakie](https://github.com/MakieOrg/Makie.jl) GUI is available by default (GLMakie is a regular dependency, not a package extension), so no extra packages need to be loaded.

## Usage

```julia
using QMagma

# Launch the interactive 1D sill-intrusion GUI:
sill_intrusion_1D()
```

Press `RUN SIMULATION` in the window to start the thermal model. Model parameters (grid spacing, number of timesteps, crustal geotherm, sill temperature/thickness injection interval, conductivity and melting parameterisations, ...) can be edited in the GUI before running.

#### TODO:
- [ ] Add source term computation after [Karlstrom et al. (2017)](https://www.nature.com/articles/ngeo2982) and [Mittal et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB021807)
- [ ] Add tracers (potentially reuse MTK.jl tracers here) to track temperature over time
- [ ] Add ZirconGrowth.jl extension 
