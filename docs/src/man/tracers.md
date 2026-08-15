# Tracers and zircon ages

## Passive tracers

A [`QMagma.Tracer`](@ref) stores its current depth, temperature, melt fraction, material
phase, and temperature-time history. `time_vec` is in Myr and `T_vec` is in °C. Phase `0`
denotes host rock and phase `1` denotes injected material.

Tracer creation depends on the emplacement model:

- [`QMagma.init_tracers`](@ref) seeds the injection window at the start of a run.
- [`QMagma.add_sill_tracers!`](@ref) adds tracers inside each freshly emplaced sill.
- [`QMagma.add_zone_tracers!`](@ref) adds Q\_magma tracers at the center of the injection
  window at the discrete injection cadence.

Each tracer moves with its model branch:

- [`QMagma.advect_tracers!`](@ref) uses the Q\_magma velocity `Params.w`.
- [`QMagma.advect_tracers_sill!`](@ref) applies the discrete sill-opening displacement.
- [`QMagma.collapse_tracers!`](@ref) applies the selected eruption closure.

[`QMagma.update_tracers_T!`](@ref) interpolates the temperature field (and optionally the
melt fraction) onto each tracer position once per step and appends the sample to its
history.

## Zircon ages

[`compute_zircon_ages`](@ref) runs ZirconGrowth.jl on each tracer path and returns the
volume-averaged crystallization age and final crystal radius per tracer:

```julia
result = compute_zircon_ages(QMagma.tracers_out)
result.age_years          # volume-averaged ages [yr]
result.zircon_radius_um   # final radii [µm]
```

Pass `return_results = true` to also get the full `ZirconGrowth.SimulationResult` objects.

[`volume_averaged_age`](@ref) weights each concentric growth shell by
``r_{i+1}^3 - r_i^3`` and assigns the age of its midpoint, matching
MagmaThermoKinematics.jl.

Tracers are omitted from the result if they have fewer than two samples, never exceed
`T_zr_min` (650 °C by default), or show no growth beyond the seed radius.

## Comparing the reservoir and erupted cargo

Use the same `t_ref_Myr` for both populations. Calling `compute_zircon_ages` twice without
this keyword gives each population its own default reference time.

```julia
reservoir = QMagma.tracers_out
cargo = QMagma.erupted_tracers_out
t_ref = maximum((tr.time_vec[end] for tr in reservoir if length(tr.time_vec) >= 2);
                init=0.0)

reservoir_result = compute_zircon_ages(reservoir; t_ref_Myr=t_ref)
cargo_result = compute_zircon_ages(cargo; t_ref_Myr=t_ref)
```

For a tracer removed by an eruption, the function adds the interval between its last
sample and `t_ref_Myr` to its crystallization age. The GUI uses this shared-clock method.

## Comparing the two injection models

Discrete sills and Q\_magma carry separate tracer populations, so they have separate age
spectra. `QMagma.tracers_out` holds only the primary (discrete) branch of a comparison
run; the Q\_magma population is in `last_run_out` under `:tracers_Qmagma` and
`:erupted_tracers_Qmagma`. Put all four populations on one clock:

```julia
run = QMagma.last_run_out
t_ref = maximum((tr.time_vec[end] for tr in run[:tracers] if length(tr.time_vec) >= 2);
                init=0.0)

sill_result = compute_zircon_ages(run[:tracers]; t_ref_Myr=t_ref)
Qmagma_result = compute_zircon_ages(run[:tracers_Qmagma]; t_ref_Myr=t_ref)
```

**COMPUTE ZIRCON AGES** in the GUI does exactly this for every model the run activated.

The tracer loop uses the available Julia threads. Start Julia with `--threads auto` to use
more than one thread.
