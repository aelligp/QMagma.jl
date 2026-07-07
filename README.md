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

The 1D column is interpreted as the axis of a laterally finite sill/chamber of radius `R_sill` (GUI input). When a contiguous zone with melt fraction > 0.5 erupts, only its **melt content** (∫ϕ dz) leaves the column — the crystal framework stays — and the erupted volume is `π R_sill² × ∫ϕ dz`. The erupted material (temperature, intruded-rock marker, and the passive tracers inside the vent, collected separately for zircon-age statistics) is removed and the vent is closed by one of three mechanisms:

- **Caldera collapse** — the roof block drops rigidly onto the chamber floor and the surface subsides. Heat and intruded-rock budgets debit exactly.
- **Elastic collapse** — host rock on both sides converges on the vent with an elastic decay law (deformation concentrated near the eruption). Temperatures are transported as intensive values, so most of the erupted heat stays in the column - eruptions recur much more frequently.
- **Hybrid collapse** — the chamber floor rises elastically while the roof displacement transitions into a rigid surface subsidence: vent-localized deformation with an (approximately) exact heat debit.

These three trigger on the melt zone exceeding a thickness threshold (default 500 m). A fourth option, **Elastic box model**, replaces the geometric trigger with chamber mechanics: recharge inflates the reservoir against the elastic host rock (storage `β_eff = β_magma + 3/(4μ)`), an eruption fires when the overpressure exceeds `ΔP_crit`, erupts the volume held in elastic storage (capped by the available melt) with the hybrid closure, and relaxes the chamber to lithostatic pressure. With default parameters the elastic storage is much smaller than one sill, so eruptions become injection-paced with volumes ≈ the recharge volume — the classic elastically-limited regime.

Both the discrete-sill and the Q_magma models erupt independently based on their own melt fraction (and their own chamber pressure in box mode); cumulative and per-event erupted volumes are plotted vs time and included in the JLD2 data export.

#### TODO:
- [x] Add source term computation after [Karlstrom et al. (2017)](https://www.nature.com/articles/ngeo2982) and [Mittal et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB021807)
- [x] Add tracers (potentially reuse MTK.jl tracers here) to track temperature over time
- [x] Add ZirconGrowth.jl extension 
- [x] Compute zircon ages from tracers and plot age density & cumulative probability spectra
- [x] Add eruption algorithm (caldera / elastic / hybrid vent closure)
- [x] Add plot of eruption volume vs time & GUI input for eruption triggers
- [ ] Add plot of zircon ages of erupted material on plot
