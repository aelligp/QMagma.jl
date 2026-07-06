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

## Eruptions

An eruption is triggered whenever a contiguous zone with melt fraction > 0.5 grows thicker than the GUI threshold (default 500 m). The whole molten band then leaves the system (temperature, intruded-rock marker, and the passive tracers inside it, which are collected separately for zircon-age statistics of the erupted material), and the vent is closed by one of three mechanisms, selectable in the `Eruption method` menu:

- **Caldera collapse** — the roof block above the erupted band drops rigidly onto the chamber floor and the surface subsides by the erupted thickness. Heat and intruded-rock budgets debit exactly.
- **Elastic collapse** — host rock on both sides converges on the vent with an elastic decay law (deformation concentrated near the eruption). Temperatures are transported as intensive values, so most of the erupted heat stays in the column - eruptions recur much more frequently.
- **Hybrid collapse** — the chamber floor rises elastically while the roof displacement transitions into a rigid surface subsidence: vent-localized deformation with an (approximately) exact heat debit.

Both the discrete-sill and the Q_magma models erupt independently based on their own melt fraction; cumulative and per-event erupted volumes are plotted vs time and included in the JLD2 data export.

#### TODO:
- [x] Add source term computation after [Karlstrom et al. (2017)](https://www.nature.com/articles/ngeo2982) and [Mittal et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB021807)
- [x] Add tracers (potentially reuse MTK.jl tracers here) to track temperature over time
- [x] Add ZirconGrowth.jl extension 
- [x] Compute zircon ages from tracers and plot age density & cumulative probability spectra
- [x] Add eruption algorithm (caldera / elastic / hybrid vent closure)
- [x] Add plot of eruption volume vs time & GUI input for eruption triggers
- [ ] Add plot of zircon ages of erupted material on plot
