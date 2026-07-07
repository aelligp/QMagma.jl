# AGENTS.md — QMagma.jl

Guidance for coding agents working in this repository.

## What this is

Interactive 1D thermal model of magmatic sill injection into the crust, with melt-fraction
tracking, passive tracers (for zircon T-t histories via ZirconGrowth.jl), eruption
mechanics, and a GLMakie GUI. Two side-by-side model flavors:

- **Discrete sills**: individual sills injected at random depths every `Sill_int_yr`,
  opening the column elastically (`insert_sill`).
- **Q_magma**: the same magma flux smeared into a steady volumetric source
  (`compute_Q_magma!` + `advect_w!`), after Karlstrom et al. (2017) / Mittal et al. (2021).

## Layout

- `src/ThermalCode_1D.jl` — all physics/numerics: nonlinear heat solver, sill insertion,
  eruption (`erupt_melt!` and friends), tracers, markers, zircon-age computation.
  No plotting.
- `src/ThermalCode_1D_GLMakie.jl` — the GUI (`sill_intrusion_1D()`): widgets, the
  timestep loop (inside an `@async` block), plotting, JLD2/movie export. Includes the
  solver file; `QMagma.jl` just includes this.
- `test/runtests.jl` — single test file, plain `@testset`s.

## Build / test / run

- Tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (~30 s once precompiled;
  first precompile of GLMakie takes minutes).
- **Known flake**: the test/aux process sometimes segfaults *after* "Testing QMagma
  tests passed" — that's GLMakie/GPU teardown at process exit, not a test failure.
- The GUI cannot be driven headlessly. To verify physics changes, write a headless
  script that replicates the timestep loop with internal functions
  (`init_model`, `nonlinear_solution`, `insert_sill`, `find_eruptible_region`,
  `erupt_melt!`, `compute_meltfraction!`); see the loop in `runtests.jl`
  ("discrete sills vs Q_magma agree...") as a template. GUI defaults worth
  replicating: 40 km crust, Δz=20 m, Δt=100 yr, 20 °C/km geotherm, 100 m sills every
  1000 yr into 10–20 km depth, Tsill=1200 °C, `MeltingParam_Smooth3rdOrder()` (the
  "Basalt" menu default), eruption threshold 500 m.
- After a GUI run, results are exported to the REPL: `QMagma.tracers_out`,
  `QMagma.erupted_tracers_out`, `QMagma.last_run_out`.

## Model conventions

- Grid: `z` ascending, `z[1] = -L` (bottom, Dirichlet `Tbot`), `z[end] = 0` (surface,
  Dirichlet `Ttop`), uniform `Δz`. Temperatures in °C (converted to K only when calling
  GeoParams melting laws).
- `rocks` is a binary (0/1) grid field marking intruded sill material — the grey band
  in the GUI's melt-fraction axis. It is diagnostic, not part of the physics.
- Melt fraction `ϕ` is a pure function of `T` (no composition/depletion state), so any
  eruption scheme can only suppress re-eruption by lowering `T`.
- `Tracer` is a mutable struct; tracers are moved by mutating `tracer.z`.
- Fields are moved by warping grid-point positions and re-interpolating onto the fixed
  grid (`semilagrangian_advection`). **This transports intensive values**: where material
  stretches, integrated content (heat, grey) is duplicated; where it compresses, content
  is discarded. Every conservation statement below follows from this.

## Eruption mechanics (hard-won semantics — read before touching)

The 1D column represents the axis of a laterally finite sill/chamber of area
`A_sill = π R_sill²` (GUI input, default radius 5 km — matching `insert_sill`'s elastic
decay radius). Only the melt content of the eruptible band leaves the column
(`melt_thickness` = ∫ϕ dz; the crystal framework stays), so the closure amplitude is
`h_melt` (or the elastic-storage release, below) and the erupted volume is
`A_sill * h_erupt` — not the old `thickness³` cube.

Two trigger types (GUI menu), each model (discrete / Q_magma) evaluated independently
on its own ϕ:

- **Melt-fraction threshold** (Caldera/Elastic/Hybrid menu items): largest contiguous
  run of `ϕ > 0.5` (`find_eruptible_region`) erupts its melt content when its bulk
  thickness ≥ threshold and it is ≥ `5Δz` from the domain edges.
- **Elastic box model** ("Elastic box model" menu item): the chamber is an elastic
  reservoir with storage `β_eff = β_magma + 3/(4μ)`. Recharge into an existing mush
  raises overpressure (`P_over += ΔV_in/(V_ch β_eff)` with `V_ch = A_sill·h_band`;
  discrete sills on injection, Q_magma continuously); eruption triggers at
  `P_over ≥ ΔP_c`, erupts `h_erupt = min(β_eff·h_band·P_over, h_melt)` with the hybrid
  closure, and resets `P_over` to lithostatic. With default parameters, elastic storage
  (~0.35% of chamber volume) is far smaller than one sill, so eruptions are
  injection-paced with volume ≈ recharge volume — the known elastically-limited regime,
  not a bug.

Erupted tracers are extracted (`extract_erupted_tracers!`) for zircon statistics;
remaining tracers and the Q_magma zone markers are moved with the same mechanism as the
fields (`collapse_tracers!`, `collapse_markers!`, both via `erupt_displacement`).

Three closure mechanisms in `erupt_melt!(...; method=...)`:

| method | host rock motion | removes heat/grey | T preserved pointwise |
|---|---|---|---|
| `:caldera` | roof block drops rigidly by the band thickness; surface subsides | exact | yes |
| `:elastic` | both walls converge with elastic decay (R = thick/2); ends pinned | ~5% only (documented) | yes |
| `:hybrid` | floor rises elastically; roof transitions from −half at the face to rigid −thick subsidence; surface sinks | ≈96% (exact when near-vent material is uniform) | yes |

Constraint that shapes all of this (do not try to "fix" it): in a fixed-length column,
removing the erupted `∫ρcpT` must be paid either in **volume** (a boundary moves →
caldera/hybrid subsidence) or in **temperature** (stretched walls dilute — a
conservative remap variant exists in `collapse_conservative`, currently unused because
the local cooling was judged unphysical). Pure elastic closure with pinned ends
necessarily conserves what it should remove; that is `:elastic`'s documented behavior,
not a bug.

Numerical invariants that must survive any change (property-test them): the warped grid
`z .+ Displ` stays strictly monotonic (else interpolation silently corrupts); linear
interpolation must not overshoot field bounds; `rocks` stays binary with no white hole
inside the grey; eruption near domain edges is guarded by the GUI margin.

`insert_sill` opens elastically (`crack_perp_displacement`, R = 5 km) and re-binarizes
`rocks` with `round` — **not `ceil`**, which systematically inflated the grey band by a
cell per boundary per injection (old bug, do not reintroduce).

## Gotchas

- The Julia LS flags many valid keyword-argument calls in these files as
  "Possible method call error" (Information severity) — false positives, ignore.
- The GUI simulation loop runs in `@async` with a `try/catch` that logs
  "Simulation loop failed"; errors there do not crash the window.
- `mod(time/SecYear, Sill_int_yr) == 0` gates injection — changing Δt so it doesn't
  divide the sill interval silently disables injection.
- Grid-alignment: a band `abs.(z .- z0) .<= half` covers `2*half/Δz` **or one more**
  cell depending on alignment; account for `length(ind)*Δz ≠ Erupt_thick` in tests.
