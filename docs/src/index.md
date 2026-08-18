```@raw html
---
layout: home

hero:
  name: QMagma.jl
  text: One-dimensional sill injection and crustal heating
  tagline: Compare discrete sill emplacement with an equivalent time-averaged magma source.
  image:
    src: /logo.svg
    alt: QMagma.jl
  actions:
    - theme: brand
      text: Installation
      link: /man/installation
    - theme: alt
      text: API reference 📚
      link: /man/listfunctions
    - theme: alt
      text: GitHub
      link: https://github.com/aelligp/QMagma.jl

features:
  - icon: 🎛️
    title: Interactive model
    details: Set model parameters and follow the thermal and eruption histories in a GLMakie window.
    link: /man/gui

  - icon: 🌋
    title: Two emplacement models
    details: Run discrete sills, the Q_magma approximation, or both from the same initial state and mean accretion rate.
    link: /man/model

  - icon: 💥
    title: Eruptions
    details: Choose a melt-thickness or pressure trigger independently from the vent-closure method.
    link: /man/eruptions

  - icon: ⏳
    title: Tracers and zircon
    details: Record tracer temperature-time paths and calculate zircon crystallization ages with ZirconGrowth.jl.
    link: /man/tracers
---
```

## Overview

QMagma.jl solves the one-dimensional heat equation for a crustal column. It includes
latent heat, temperature-dependent material properties, passive tracers, and optional
eruptions. ZirconGrowth.jl converts tracer temperature-time paths into crystallization
ages.

The two emplacement models are:

- **Discrete sills:** finite-thickness sills are injected at random depths within the
  injection window. Each sill opens the host rock elastically.
- **Q\_magma:** the same mean accretion rate is represented by a volumetric heat source
  and the ensemble-mean host-rock velocity of the discrete opening law, following
  [Karlstrom et al. (2017)](https://www.nature.com/articles/ngeo2982) and
  [Mittal et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB021807).

Use `Both (compare)` in the GUI to run the models from the same initial geotherm and mean
accretion rate.

QMagma.jl builds on

* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) - material physics
  (conductivity, heat capacity, density, melting parameterisations, latent heat)
* [ZirconGrowth.jl](https://github.com/aelligp/ZirconGrowth.jl) - zircon crystal growth
  along a cooling path
* [GLMakie.jl](https://github.com/MakieOrg/Makie.jl) - the interactive interface, built on
  a Makie package extension loaded on demand
* [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) and
  [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) - output
## Quick start

```julia
using QMagma
using GLMakie   # loads the GUI extension
sill_intrusion_1D()
```

See [The GUI](man/gui.md) for the controls and
[Headless runs and export](man/scripting.md) for direct use of the solver functions.
