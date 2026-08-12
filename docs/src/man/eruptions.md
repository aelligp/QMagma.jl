# Eruptions

The one-dimensional column represents the axis of a sill or chamber with radius `R_sill`
and area ``A = \pi R_{\mathrm{sill}}^2``. Only melt leaves during an eruption; crystals
remain in the column. For an eruptible interval, the available melt thickness is

```math
h_{\mathrm{melt}} = \int \phi\,dz,
```

computed by [`QMagma.melt_thickness`](@ref). An event that withdraws thickness
``h_{\mathrm{erupt}}`` has volume ``A h_{\mathrm{erupt}}``.

The GUI treats the trigger and vent closure as separate choices. In comparison runs, the
discrete and Q\_magma branches evaluate their own melt and pressure states.

## Shared eruption checks

[`QMagma.find_eruptible_region`](@ref) returns the largest contiguous region with
``\phi > 0.5``. Separate lenses are not combined. Every active trigger also requires:

- the region midpoint to be no deeper than `Max erupt depth`;
- at least `5Δz` between the region and either model boundary; and
- a withdrawal greater than `2Δz`.

The last condition prevents a collapse smaller than the grid can resolve.

## Triggers

### Melt thickness

The region erupts when its bulk thickness, `z_hi - z_lo`, reaches `Threshold`. The model
then withdraws all of the melt in that region, ``h_{\mathrm{erupt}} = h_{\mathrm{melt}}``.

### Elastic box model

The effective chamber compliance is

```math
\beta_{\mathrm{eff}} = \beta_{\mathrm{magma}} + \frac{3}{4\mu}.
```

Recharge increases overpressure only after a region with ``\phi > 0.5`` exists. The GUI
uses

```math
\Delta P = \frac{\Delta h_{\mathrm{in}}}
                  {\max(h_{\mathrm{band}}, d)\,\beta_{\mathrm{eff}}},
```

where `d` is sill thickness and ``\Delta h_{\mathrm{in}}`` is either one discrete sill or
the Q\_magma accretion during the timestep. At ``P_{\mathrm{over}} \geq \Delta P_c``, the
withdrawal is

```math
h_{\mathrm{erupt}} = \min(\beta_{\mathrm{eff}} h_{\mathrm{band}}
P_{\mathrm{over}}, h_{\mathrm{melt}}).
```

The overpressure is reset to zero after the event. With the default parameters, elastic
storage is much smaller than one sill, so discrete eruptions are generally tied to
injection events.

### D&H three-phase model

This trigger evolves a lumped melt-crystal-gas chamber based on Degruyter and Huber
(2014):

```math
\left(\frac{1}{\beta_r} + \frac{1}{\beta_m}\right)\frac{dP}{dt}
= \frac{\dot M_{\mathrm{in}}}{\rho V}
- \frac{1}{\rho}\left.\frac{d\rho}{dt}\right|_{T,\phi}
- \frac{P-P_{\mathrm{lith}}}{\eta_r}.
```

[`QMagma.step_overpressure!`](@ref) integrates this equation. It calculates magma
compressibility by finite difference from [`QMagma.mixture_density`](@ref), and estimates
the fixed-pressure density change from the current and previous mean mush states. Both the
pressure reservoir and the eventual withdrawal use the largest contiguous region with
``\phi \geq 0.5``; disconnected melt lenses are not combined into one chamber.

In the GUI, `μ shear` and `β magma` set ``\beta_r = 1/\beta_{\mathrm{eff}}`` for this
trigger. The D&H mixture compressibility ``1/\beta_m`` is calculated from mixture density;
it is not the value entered in the `β magma` control.

The free-gas phase is a shallow-crust mechanism. By default,
[`QMagma.water_gas_partition`](@ref) applies the silicic Liu et al. (2005) water-solubility
law and [`QMagma.rho_gas_RK`](@ref) only for chamber centroids in the upper 10 km
(`EruptionParams.z_gas_max`). Deeper chambers retain water in the condensed phase, report
zero free gas, and never evaluate RK. RK additionally fails outside its 30–400 MPa and
873.15–1173.15 K calibration box. The GUI therefore permits this trigger only with the
rhyolite melting preset. CO₂ is not included. Wall viscosity is recalculated each step
from the temperatures immediately outside the mush using
[`QMagma.wall_relaxation_viscosity`](@ref).

The pressure calculation is substepped, up to 10,000 substeps per thermal step. Whenever
the chamber reaches ``\Delta P_c``, is shallower than the depth limit, and has gas volume
fraction below `ϕ_g_crit`, the stored volume is drained and pressure returns to
`ΔP_relax`. [`QMagma.pending_withdrawal!`](@ref) accumulates drained melt until more than
`2Δz` can be withdrawn from the grid; [`QMagma.commit_pending_withdrawal!`](@ref) debits
the queue after a successful event.

The parameters and state are stored in [`QMagma.EruptionParams`](@ref) and
[`QMagma.EruptionState`](@ref).

## Vent closure

[`QMagma.erupt_melt!`](@ref) accepts three closure methods:

| `method` | Motion | Surface subsidence | Thermal effect |
|---|---|---:|---|
| `:caldera` | The roof drops rigidly onto the chamber floor. | ``h_{\mathrm{erupt}}`` | Removes the vent profile with only grid and surface-fill error. |
| `:elastic` | Both walls converge with elastic decay; both boundaries remain fixed. | 0 | Intensive interpolation retains much of the heat assigned to the erupted melt. |
| `:hybrid` | The floor rises elastically and the roof approaches rigid subsidence away from the vent. | ``h_{\mathrm{erupt}}`` | Approximately removes the erupted heat without dilution cooling. |

The intruded-rock field is zeroed inside the erupted interval and conservatively remapped
for the elastic and hybrid closures. [`QMagma.collapse_displacement`](@ref) builds the
grid displacement and [`QMagma.collapse_advection`](@ref) applies it to temperature.

The elastic method cannot both keep the model boundaries fixed and remove heat while
preserving temperature as an intensive field. Use the `enthalpy_budget` output to measure
the resulting energy residual.

## Events and tracers

[`QMagma.realize_eruption!`](@ref) updates temperature and the intruded-rock field, samples
erupted tracers, and constructs one [`QMagma.EruptionEvent`](@ref). Event construction
throws an error unless the requested, state-withdrawn, cargo-represented, and booked melt
thicknesses agree. `state_withdrawn` is the prescribed closure amplitude; the separate
`melt_removed` field measures the actual full-column melt-storage change after remapping.
For a pinned elastic closure these are intentionally very different. The default
tracer-cargo tolerance is half a grid cell.

Each event records its trigger and realization times, closure, depth, tracer count, column
enthalpy before and after the event, erupted enthalpy, the enthalpy residual, intruded-magma
removal, and measured melt-storage removal. Surviving
tracers and Q\_magma zone markers move with the same closure through
[`QMagma.collapse_tracers!`](@ref) and [`QMagma.collapse_markers!`](@ref).
