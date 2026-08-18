<img src="docs/src/assets/logo.svg" width="130" align="right" alt="QMagma.jl logo">

# QMagma.jl

[![Version](https://img.shields.io/github/v/release/aelligp/QMagma.jl?label=version&sort=semver)](https://github.com/aelligp/QMagma.jl/releases)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://aelligp.github.io/QMagma.jl/dev/)
[![CI](https://github.com/aelligp/QMagma.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/aelligp/QMagma.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/aelligp/QMagma.jl/graph/badge.svg?token=8TD9XOQI0P)](https://codecov.io/gh/aelligp/QMagma.jl)

QMagma.jl is a one-dimensional thermal model of repeated sill injection. It compares
discrete sill emplacement with an equivalent, time-averaged `Q_magma` source. Both models
track melt fraction, passive tracers, eruptions, and zircon crystallization ages.

## Run the GUI

QMagma.jl requires Julia 1.10 or newer. It is not registered, so install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/aelligp/QMagma.jl")
```

Then launch the GLMakie interface:

```julia
using QMagma
using GLMakie   # loads the GUI extension
sill_intrusion_1D()
```

Set the model parameters and press **RUN SIMULATION**. The default comparison uses the
same initial geotherm and mean accretion rate for both emplacement models,
`ȧ = sill thickness / injection interval`.

## Model output

The GUI plots the thermal state and eruption history during a run. Finished runs are
available from the REPL as:

- `QMagma.tracers_out`
- `QMagma.erupted_tracers_out`
- `QMagma.last_run_out`

The GUI can also save the figure, the full run as JLD2, and 1D, 2D, and 3D VTK output.

See the [documentation](https://aelligp.github.io/QMagma.jl/dev/) for the model equations,
eruption triggers, tracer handling, scripting, and export.
