# The GUI

```julia
using QMagma
using GLMakie

sill_intrusion_1D()
```

[`sill_intrusion_1D`](@ref) opens the interactive window. The window height is limited to
the available height of the primary monitor. Pass `size = (width, height)` in pixels to
override it. Press **RUN SIMULATION** to start and **STOP** to interrupt a run. The
simulation loop is asynchronous; failures are reported as `Simulation loop failed` in the
Julia log.

## Panels

The left side shows temperature and melt fraction against depth, with intruded material in
gray. The right side shows time series:
maximum temperature and maximum melt fraction, cumulative and per-event erupted volume,
and, for a D&H three-phase run, water speciation and mean mush melt fraction.

## Controls

All controls take effect when the run is started, so change them before pressing **RUN
SIMULATION**.

### Numerics

| Control | Default | Meaning |
|---|---|---|
| `Δz [m]` | 20 | Grid spacing |
| `# steps nt` | 3000 | Number of timesteps |
| `Δt [yrs]` | 100 | Timestep |

### Crust

| Control | Default | Meaning |
|---|---|---|
| `Crust [km]` | 40 | Model depth `L` |
| `Ttop [ᵒC]` | 0 | Surface temperature (Dirichlet) |
| `Geotherm [ᵒC/km]` | 20 | Initial linear geotherm; sets the basal temperature |

### Injection

| Control | Default | Meaning |
|---|---|---|
| `Sill T [ᵒC]` | 1200 | Injection temperature |
| `Sill thick [m]` | 100 | Full aperture of each discrete sill; variable flux changes event frequency |
| Flux-history menu | `Constant` | `Constant`, `Linear ramp`, `Pulse`, or `CSV table` |
| `Base flux [m/yr]` | 0.1 | Constant flux or ramp start; unused by a pulse |
| `Peak/end flux [m/yr]` | 0.2 | Ramp endpoint or pulse flux |
| `Flux start/end [kyr]` | 50 / 100 | Injection episode; flux is zero outside it |
| `Flux CSV path` | `examples/flux_history.csv` | Example table used in `CSV table` mode |
| `Top inj. [km]` | 10 | Top of the injection window |
| `Bottom inj. [km]` | 20 | Bottom of the injection window |

Both emplacement styles use the same history. Ramp and pulse are *episodes*: `Flux
start/end` bound the whole injection, and the flux is zero outside them, so sills stop
arriving at `Flux end`. Ramp flux climbs linearly from the base to the peak across that
window; pulse flux equals the peak throughout it and has no base, so its box is disabled.
Use a CSV table for a steady background flux with a surge on top. CSV tables contain
`time_kyr,flux_m_per_yr,depth_km` and may include one header row. Depth is positive downward;
when present it overrides the GUI injection interval. The discrete branch centers each sill
at the interpolated depth; the Q_magma branch centers a source band one sill aperture wide at
the same depth. A two-column CSV falls back to `Top inj.` / `Bottom inj.`. Flux and depth use
linear interpolation with flat extrapolation, and flux is integrated exactly over each
thermal step, so a pulse is not lost when a timestep crosses both pulse edges.
The realized step-mean history is exported as `last_run_out[:flux_m_per_yr]` alongside
`time_vec`; the selected mode and its parameters are exported under `flux_*` keys, including
`flux_table_depth_km`.

### Material physics

| Control | Default | Meaning |
|---|---|---|
| `Latent heat [kJ/kg]` | 255 | Latent heat of fusion |
| Conductivity menu | Constant 3 W/m/K | Constant properties, or Whittington temperature-dependent conductivity and heat capacity |
| Melting menu | `MeltingParam_Basalt` | Assimilation, basalt (`MeltingParam_Smooth3rdOrder`), or rhyolite melt law |

### Method

`Discrete sills`, `Q_magma`, or `Both (compare)` (the default), which runs the two side by
side from identical initial conditions and identical `ȧ(t)`. See
[Model formulation](model.md).

### Eruption

The trigger menu selects `None` or `D&H 3-phase`. The collapse menu independently selects
`Hybrid` or `Caldera`. With `None` selected the eruption controls are disabled; they all
apply to the 3-phase chamber together.

| Control | Default | Role |
|---|---|---|
| `Sill radius [km]` | 5 | Chamber area `πR²`, and so the reported erupted volume |
| `ΔP crit [MPa]` | 20 | Roof-failure overpressure |
| `μ shear [GPa]` | 10 | Host-rock shear modulus; sets `1/β_r = 3/(4με)`, `ε = min(V/2R_sill, 1)` |
| `H₂O [wt%]` | 0 | Total magmatic water; nonzero values activate volatile partitioning and gas density |
| `Min. chamber melt [m]` | 500 | Melt content `∫ϕ dz` a connected region must hold to count as a chamber |

The melting menu also selects the H₂O saturation law — basalt takes the mafic law, rhyolite
and assimilation the silicic Liu et al. (2005) law — so the two cannot be mismatched.

The shell relaxation viscosity has no widget: it is computed each step from the mush
temperature and the geotherm at the chamber's depth
([`QMagma.crustal_relaxation_viscosity`](@ref)). See [Eruptions](eruptions.md).

### Output

`filename` is the stem used by all output. **SAVE SCREENSHOT** writes `<filename>.png`,
**SAVE JLD2 + VTK** writes the full run to `<filename>.jld2` plus VTK files for each active
model - the 1D profiles and Gaussian 2D/3D temperature fields with σ equal to the sill
radius and a lateral extent of ±3σ (see [Headless runs and export](scripting.md)). The
`Record movie` toggle records the run to a video file. **COMPUTE ZIRCON AGES** runs
[`compute_zircon_ages`](@ref) on the tracers of the finished run and plots the age
spectra of the reservoir and of the erupted cargo (see
[Tracers and zircon ages](tracers.md)).

## Results in the REPL

When a run finishes, its results are assigned to module-level variables:

| Variable | Contents |
|---|---|
| `QMagma.tracers_out` | Tracers of the primary model, with their T-t histories |
| `QMagma.erupted_tracers_out` | Tracers removed by eruptions |
| `QMagma.last_run_out` | `Dict{Symbol,Any}` with the full run |

In a comparison run the discrete-sill model is the primary one; the Q\_magma counterparts
are stored in `last_run_out` under keys suffixed `_Qmagma`.

Common keys in `last_run_out` include:

- `:z`, `:T_background`, `:gaussian_sigma`, and `:time_vec`
- `:tracers`, `:erupted_tracers`, `:eruption_events`, and `:enthalpy_budget` for the
  primary branch
- `:T`, `:phi`, `:rocks`, `:Tmax_vec`, and `:phimax_vec` when discrete sills run
- `:erupted_volume_vec`, `:eruption_event_time_vec`, `:eruption_event_volume_vec`,
  `:collapse_event_time_vec`, `:collapse_event_thickness_vec`, and
  `:surface_subsidence_vec` for the discrete branch
- for a D&H run, `:chamber_dP_vec`, `:chamber_mdiss_vec`, `:chamber_Xg_vec`,
  `:chamber_phig_vec`, `:chamber_rhogas_vec`, `:chamber_etar_vec`, `:chamber_phimush_vec`,
  `:chamber_M_vec`, `:chamber_MH2O_vec`, and `:chamber_mass_residual_vec` against
  `:time_vec`. Chamber masses and residuals are areal quantities in kg/m².

Q\_magma profiles and time series use the same names with `_Qmagma` appended, for example
`:T_Qmagma`, `:eruption_events_Qmagma`, and `:enthalpy_budget_Qmagma`.

!!! info "Headless runs"
    The widgets do not provide a scripting interface. For automated runs, call the solver
    functions directly as shown in [Headless runs and export](scripting.md).
