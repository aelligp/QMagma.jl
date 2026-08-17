# Eruptions

The one-dimensional column represents the axis of a sill or chamber with radius `R_sill`.
It thins away from the axis as a penny-shaped crack, so its plan-view area is
``A = \tfrac{2}{3}\pi R_{\mathrm{sill}}^2`` ([`lateral_effective_area`](@ref)), not
``\pi R_{\mathrm{sill}}^2``. A connected region is considered eruptible where
the melt fraction is ``\phi \geq 0.5`` — equivalently, where the crystal fraction
``1-\phi \leq 0.5``. The eruption withdraws bulk magma, including its liquid and entrained
crystals. A withdrawn bulk-equivalent thickness ``h_{\mathrm{erupt}}`` therefore has volume
``A h_{\mathrm{erupt}}``.

The liquid content of an interval is the separate diagnostic

```math
h_{\mathrm{melt}} = \int \phi\,dz,
```

computed by [`QMagma.melt_thickness`](@ref). It is not the bulk erupted thickness. The
liquid carried by one withdrawal is recorded separately as `EruptionEvent.melt_requested`,
and it — not the bulk `requested` — is what the erupted tracer cargo is sampled to, since
tracers ride the melt. The enthalpy debit, by contrast, is charged on the bulk parcel:
crystals carry sensible heat but not the latent heat of fusion.

The GUI treats the trigger and vent closure as separate choices. In comparison runs, the
discrete and Q\_magma branches each carry their own chamber, and neither reads the other's
state.

## Shared eruption checks

[`QMagma.eruptible_mush`](@ref) returns the largest contiguous region with
``\phi \geq`` `ϕ_erupt`; separate lenses are distinct chambers and are never combined.
Drainage requires a dike that reaches the surface, ``\Delta P + P_{\mathrm{lith}} -
\rho_{\mathrm{magma}} g |z_c| \geq \sigma_{\mathrm{barrier}}``
([`QMagma.dike_ascends`](@ref)), tested at the moment of drainage against the overpressure
that drives the dike. Realizing that drainage on the grid additionally requires:

- at least `5Δz` between the region and either model boundary; and
- a withdrawal greater than `2Δz`.

These two are grid constraints, not physics: the first leaves the closure host rock to draw
on, the second prevents a collapse smaller than the grid can resolve. Melt drained but not
yet resolvable stays queued rather than being discarded.

The propagation check replaces a depth cutoff: the magma density carries the gas content, so
a gas-rich chamber ascends against a barrier that stalls a degassed one at the same
overpressure.

``P_{\mathrm{lith}}`` is the integrated crustal load
([`QMagma.lithostatic_pressure`](@ref)), with ``\rho_{\mathrm{crust}}(P,T)`` from
`EruptionParams.crust`. Thermal expansion and compression nearly cancel down a geotherm, so
that column varies by well under a percent — small next to the several hundred kg/m³ of
magma-crust contrast. The pressure criterion therefore grows with depth and never limits it.

The depth limit comes from the second requirement: the dike must reach the surface before
it freezes. A dike of half-width ``w`` ascends at ``v = G w^2/(3\eta)`` under the driving
gradient ``G = \Delta P/L + \Delta\rho g`` and solidifies once its thermal boundary layer
reaches the wall, after ``w^2/\kappa``. It survives a path ``L`` only while ``L/v <
w^2/\kappa``, giving the reachable length ([`QMagma.max_ascent_length`](@ref))

```math
L^2 - (\Delta\rho g\, c) L - \Delta P\, c = 0, \qquad c = \frac{w^4}{3\eta\kappa}.
```

The ``w^4`` and the ``1/\eta`` do the work. With `w_dike` = 1 m a basalt dike clears any
crustal depth, while a silicic one freezes within a few km unless it is wider or hotter —
which is why silicic eruptions tap shallow storage and basalts do not have to. `EruptionParams.melt_viscosity` selects
the composition; the default is the basalt parameterisation.

## Trigger

The eruption trigger is a lumped melt-crystal-gas chamber based on Degruyter and Huber
(2014):

```math
\left(\frac{1}{\beta_r} + \frac{1}{\beta_m}\right)\frac{dP}{dt}
= \frac{\dot M_{\mathrm{in}}}{\rho V}
- \frac{1}{\rho}\left.\frac{d\rho}{dt}\right|_{T,\phi}
- \frac{P-P_{\mathrm{lith}}}{\eta_r}.
```

[`QMagma.step_overpressure!`](@ref) integrates this equation. It differentiates
[`QMagma.mixture_density`](@ref) exactly with respect to pressure for the magma
compressibility, and estimates
the fixed-pressure density change from the current and previous mean mush states.

The pressure equation is total-mass conservation with ``M = \rho V`` differentiated, but
that identity is no longer the only mass account. Each chamber also integrates independent
areal inventories

```math
\frac{dM}{dt}=\dot M_{\mathrm{in}}-\dot M_{\mathrm{out}}, \qquad
\frac{dM_{\mathrm{H_2O}}}{dt}=m_{w,\mathrm{in}}\dot M_{\mathrm{in}}
-m_w\dot M_{\mathrm{out}}.
```

Mass crossing a moving mush margin is booked separately as a control-volume boundary flux.
The volatile partition uses ``m_w=M_{\mathrm{H_2O}}/M`` from these inventories rather than
re-reading the input parameter. `EruptionState.mass_residual` reports ``\rho V-M``; a
relative mismatch above 0.5 % stops the simulation instead of allowing the pressure and
mass solutions to drift apart silently. Smaller time-integration error is removed by an
elastic Newton correction of ``(P,V)`` onto ``M=\rho V`` after each step. The correction
moves the thermodynamic/mechanical state; it never replaces the independently integrated
mass with the EOS result.

The chamber volume ``V`` is the chamber's own state, not a re-reading of the grid's mush
extent. It evolves under the mechanical rate

```math
\frac{1}{V}\frac{dV}{dt} = \frac{1}{\beta_r}\frac{dP}{dt}
+ \frac{P-P_{\mathrm{lith}}}{\eta_r},
```

the same elastic and viscous terms that set ``dP/dt``, now also permitted to move the wall.
A fixed 1-D grid cannot represent walls creeping outward, so the mush extent contributes
only its *change* between steps — melting and freezing at the margins — and ``V`` is seeded
from it when the chamber first appears. Inflation therefore feeds back on the recharge term
``\dot M_{\mathrm{in}}/(\rho V)``: a chamber that accommodates recharge by growing
pressurizes more slowly than one that cannot, which is the physical relief path a
fixed-volume chamber lacks.

In the GUI, `G` sets ``1/\beta_r = 3/(4G\varepsilon)``
([`QMagma.host_compliance`](@ref)), the compliance of the host rock alone. The aspect ratio
``\varepsilon = \min(V/2R_{\mathrm{sill}}, 1)`` flattens the chamber against a fixed radius:
an oblate sill is more compliant than the sphere the classical ``3/(4G)`` describes, and
recovers it at ``\varepsilon = 1``. Because the chamber's volume is its own state, this is
re-evaluated each step, so an inflating chamber grows more compliant.

The magma's own compressibility ``1/\beta_m`` is computed from the mixture density — the gas
phase plus the melt and crystal density laws — and is added separately by the ODE, so it
must not also be folded into ``\beta_r``.

[`QMagma.water_gas_partition`](@ref) takes the silicic Liu et al. (2005) water-solubility
law or the mafic law from GeoParams. Melt, crystal, and gas density are three independent
density parameterizations in `EruptionParams`; the default gas law is the modified
Redlich–Kwong parameterization of Huber et al. (2010). The default magma is dry
(`m_w = 0`), so no gas-density law is evaluated unless water is explicitly supplied. The
free-gas phase also ends where the solubility law says it does: a melt that dissolves all
of `m_w` has ``X_g = 0``, and no gas-density law is evaluated.

Which saturation law applies is fixed by composition, not chosen separately: Liu is
calibrated on silicic melts and `Mafic_Solubility` on basaltic ones, so
[`QMagma.gui_composition`](@ref) returns the melting parameterisation and the solubility law
as one selection and there is no control that can set one without the other. CO₂ is not
included.

`m_eq` is water per melt *mass*, so converting it to a per-magma-mass fraction weights it by
the melt mass fraction rather than by ϕ, which is volumetric. At ρ_melt/ρ_x = 2400/2700 that
distinction moves ``X_g`` by 20 % at ϕ = 0.5 and 64 % at ϕ = 0.6, because ``X_g`` is a small
difference of two larger numbers.

`ρ_melt` and `ρ_x` are compressible density laws rather than constants. This matters for
``1/\beta_m``, which is dominated by gas resorption and would otherwise fall to exactly zero
the moment the gas phase vanishes — undersaturated magma would be perfectly rigid, and the
relaxation time and drained volume would jump discontinuously across one grid point.

The relaxation viscosity ``\eta_r`` is the effective viscosity of the whole visco-elastic
shell, integrated each step by [`QMagma.crustal_relaxation_viscosity`](@ref) (D&H eq. A.18)
over the conduction profile running from the mush out to the undisturbed geotherm at the
chamber's depth, and passed per chamber into [`QMagma.step_overpressure!`](@ref) so the two
emplacement models cannot write through each other's shell state. The integral is an
arithmetic mean of viscosity weighted as ``r^{-4}``, so although it favours the near wall
geometrically, the exponential Arrhenius rise outward means the cold outer shell sets the
result: ``\eta_r`` moves about five orders of magnitude per 200 K of far-field temperature
and barely responds to the mush temperature. Chamber depth is therefore the dominant control
on whether a chamber stores or erupts — shallow chambers sit in crust too cold to creep and
so pressurize to failure, deep ones relax and accumulate.

The pressure calculation is substepped, up to 10,000 substeps per thermal step. Whenever
the chamber reaches ``\Delta P_c``, has gas volume fraction below `ϕ_g_crit`, and can drive
a dike to the surface ([`QMagma.dike_ascends`](@ref)), pressure returns to `ΔP_relax` along
an instantaneous elastic path. The erupted mass is the difference between the pre-drain
inventory and ``\rho V`` in that relaxed state; dividing it by the pre-drain bulk density
gives the equivalent thickness passed to the 1-D withdrawal.
[`QMagma.pending_withdrawal!`](@ref) accumulates drained melt until more than `2Δz` can be
withdrawn from the grid;
[`QMagma.commit_pending_withdrawal!`](@ref) debits the queue after a successful event.

The parameters and state are stored in [`QMagma.EruptionParams`](@ref) and
[`QMagma.EruptionState`](@ref).

## Vent closure

[`QMagma.erupt_melt!`](@ref) accepts two closure methods. Both conserve volume and subside
the free surface by ``h_{\mathrm{erupt}}``; they differ in whether the deformation is rigid
or elastic, and that choice is what sets the energy residual.

| `method` | Motion | Thermal effect |
|---|---|---|
| `:caldera` | The roof drops rigidly onto the chamber floor. | No parcel changes temperature, so the debit is the band's own content plus grid and surface-fill error. The residual is a stable offset, independent of what sits over the vent. |
| `:hybrid` | The floor rises elastically and the roof approaches rigid subsidence away from the vent. | Temperature is transported intensively, so the raised floor re-covers part of the vent at its own temperature. Exact when the near-vent column is uniform; stacked sills make it non-uniform, so the residual varies with what sits over the vent. |

The intruded-rock field is zeroed inside the erupted interval and conservatively remapped
for the hybrid closure. [`QMagma.collapse_displacement`](@ref) builds the grid displacement
and [`QMagma.collapse_advection`](@ref) applies it to temperature. Use the `enthalpy_budget`
output to measure the residual of whichever closure a run selects.

## Events and tracers

[`QMagma.realize_eruption!`](@ref) updates temperature and the intruded-rock field, samples
erupted tracers, and constructs one [`QMagma.EruptionEvent`](@ref). Event construction
throws an error unless the melt thickness the tracer cargo represents matches the requested
withdrawal to the declared tolerance. The tracers are sampled from the mush by a separate
mechanism from the thermal withdrawal, so this is the one account that can disagree. The
default tolerance is half a grid cell, widened to the sampler's own quantum when the band is
thinly seeded. The separate `melt_removed` field measures the actual full-column
melt-storage change after remapping, which the intensive `:hybrid` transport can make differ
from the requested amplitude.

Each event records its trigger and realization times, closure, depth, tracer count, column
enthalpy before and after the event, erupted enthalpy, the enthalpy residual, intruded-magma
removal, and measured melt-storage removal. Surviving
tracers and Q\_magma zone markers move with the same closure through
[`QMagma.collapse_tracers!`](@ref) and [`QMagma.collapse_markers!`](@ref).
