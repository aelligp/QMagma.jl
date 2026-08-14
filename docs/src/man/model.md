# Model formulation

## Governing equation

QMagma.jl solves the one-dimensional heat equation with latent heat of crystallization,
a volumetric source term and a lateral heat-loss term,

```math
\rho \left( c_p + L \frac{\partial \phi}{\partial T} \right) \frac{\partial T}{\partial t}
= \frac{\partial}{\partial z} \left( k \frac{\partial T}{\partial z} \right)
- \frac{2k}{R_\mathrm{lat}^2}\left( T - T_\mathrm{bg} \right) + Q ,
```

where ``\phi(T)`` is the melt fraction, ``L`` the latent heat of fusion and ``Q`` the
magmatic source term. The effective heat capacity ``c_p + L\,\partial\phi/\partial T``
accounts for latent heat released as magma crystallizes.

## Lateral heat loss

The column is the axis of an axisymmetric body of lateral extent ``R_\mathrm{lat}``, so
heat also escapes sideways into the crust. The anomaly ``T - T_\mathrm{bg}`` relative to
the initial geotherm ``T_\mathrm{bg}`` is taken to taper radially as a Gaussian of width
``R_\mathrm{lat}`` — the same shape [`gaussian_thermal_structure`](@ref) uses to expand a
profile into 2-D or 3-D — and the radial part of the Laplacian of that Gaussian on the axis
is ``-2(T - T_\mathrm{bg})/R_\mathrm{lat}^2``. Undisturbed host rock sits at
``T_\mathrm{bg}``, where the term vanishes; a hot chamber loses heat at a rate set by its
own width, and the correction dominates vertical conduction once ``R_\mathrm{lat}`` falls
below the chamber thickness.

`R_lat = Inf` recovers a purely one-dimensional column. The GUI ties it to the chamber
radius `R_sill`, the same radius that fixes the chamber's compliance and the export
Gaussian's ``\sigma``. [`QMagma.lateral_loss_energy`](@ref) books the loss in the enthalpy
budget alongside the top and bottom boundary fluxes.

Material properties ``k``, ``c_p``, ``\rho``, ``\phi`` and ``\partial\phi/\partial T`` come
from [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) and are functions of
temperature only. Melt fraction carries no compositional or depletion state: ``\phi`` is a
pure function of ``T``, so the only way any process in the model can suppress remelting or
re-eruption is by lowering the temperature.

## Grid and boundary conditions

The column is discretized on a uniform grid `z` of `nz` points with

- `z[1] = -L` at the base of the model, held at the Dirichlet value `Tbot`,
- `z[end] = 0` at the free surface, held at the Dirichlet value `Ttop`,
- ascending `z` (negative downward), spacing `Δz = L/(nz-1)`.

Temperatures are in °C everywhere; they are converted to Kelvin only where GeoParams
requires it. Conductivity lives on the `nz-1` cell centers; all other fields are on grid
points. [`QMagma.init_model`](@ref) builds this state and returns the `Params` named tuple
that the solver reads and writes.

## Time integration

Each timestep is one implicit (backward Euler) solve. The residual
[`QMagma.Res!`](@ref) is assembled with the material properties **frozen at the old
temperature**: [`QMagma.update_properties!`](@ref) evaluates them once per step from
`Params.Told`, so the residual is linear in the unknown `T` and the expensive GeoParams
evaluations are not repeated for every residual and Jacobian evaluation.

[`QMagma.nonlinear_solution`](@ref) drives a Newton iteration on that residual. The
Jacobian is built by [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl)
with forward-mode automatic differentiation and matrix coloring on the tridiagonal
sparsity pattern, and each update is scaled by the step size chosen by
[`QMagma.LineSearch`](@ref).

## Two ways to supply magma

Both models use the same accretion history ``\dot a(t)``. For constant flux the default is
0.1 m/yr. The GUI also supplies linear ramps, pulses, and piecewise-linear CSV tables.
[`QMagma.injected_thickness`](@ref) integrates built-in histories exactly across every
thermal timestep, including steps that cross a pulse or ramp boundary. A constant history
is equivalent to

```math
\dot a = \frac{d}{\Delta t_{\text{inj}}},
```

with `d` the full sill aperture and ``\Delta t_{\text{inj}}`` an equivalent injection
interval. Under variable flux, `d` remains fixed and the discrete event frequency changes.
An optional CSV `depth_km` column supplies a positive-downward, piecewise-linear emplacement
depth. It centers both the discrete sill and the aperture-wide Q_magma source at the same
location; without it, the GUI injection interval is used.

### Discrete sills

[`QMagma.insert_sill`](@ref) emplaces a sill of full thickness `Sill_thick` at depth
`Sill_z0`: the host rock is pushed apart, `Sill_T` is mixed over the overlapped control
volumes, and the intruded-rock content is added exactly. Emplacement is
symmetric, so each wall moves by at most half the aperture.

The opening displacement follows [`QMagma.crack_perp_displacement`](@ref),

```math
u(z) = \frac{d}{2}\left( 1 - \frac{|z|}{\sqrt{r^2 + z^2}} \right),
```

which decays away from the sill over the crack radius `r` (5 km by default), or is
constant with `SillType = :constant`.

Emplacement is keyed to the injected thickness rather than to elapsed time. Each step,
[`QMagma.injected_thickness`](@ref) integrates the accretion-rate history ``\dot a`` over
the step and [`QMagma.sills_due`](@ref) counts the whole apertures that thickness completes,
so the injection interval need not be an integer multiple of `Δt` and a single step may
contain several injections. ``\dot a`` may be a callable ``\dot a(t)``: the aperture is
fixed, so a varying flux changes the event *frequency*, and the same integrated thickness
also sets the `Q_magma` branch's rate — the two emplacement models cannot drift apart.

!!! note "Two advection schemes, on purpose"
    Temperature is moved with [`QMagma.semilagrangian_advection`](@ref) - the field is
    interpolated at back-traced positions, which transports the *intensive* value. The
    intruded-rock marker is moved with [`QMagma.conservative_advection`](@ref), a
    finite-volume remap that preserves the marker's node-centered control-volume integral
    instead of its value, so thin bands are not eroded by repeated advection. The marker can
    contain fractional values and values above one where material is compressed.

### Q\_magma

[`QMagma.compute_Q_magma!`](@ref) replaces individual sills by an equivalent continuous
source over the injection window ``z \in [-z_{\text{bot}}, -z_{\text{top}}]`` of thickness
``H``:

```math
Q_{\text{magma}}(z,t) = \rho \frac{\dot a}{H}
\Big[ c_p \big( T_m - T(z,t) \big) + L \big( 1 - \phi(T) \big) \Big] ,
```

zero outside the window. The first term is the sensible heat the magma gives up cooling
from the injection temperature ``T_m`` to the local temperature; the second is the latent
heat released as it crystallizes from ``\phi = 1`` down to the local melt fraction.

Injection also displaces host rock, and the smeared model has to do the same or the two
branches would not be comparable. [`QMagma.mean_sill_velocity`](@ref) gives the
ensemble-mean velocity produced by sills whose centers are uniformly distributed across the
injection window, using the same elastic opening law; [`QMagma.advect_w!`](@ref) applies it
to the temperature field once per step with the same semi-Lagrangian scheme used for
discrete injection.

## Energy diagnostics

Because fields are transported by warping grid positions and re-interpolating, integrated
quantities are not conserved to machine precision, and eruption closures remove energy.
The following functions expose the energy balance:

- [`QMagma.column_enthalpy`](@ref) - ``\int \rho (c_p T + L\phi)\,dz`` for the whole column,
- [`QMagma.conductive_boundary_energy`](@ref) - net conductive flux through the boundaries
  over one step,
- [`QMagma.source_energy`](@ref) - heat supplied by ``Q`` over one step,
- [`QMagma.magma_heat_input`](@ref) - heat carried by one discrete sill,
- [`QMagma.EnthalpyBudget`](@ref) and [`QMagma.update_enthalpy_budget!`](@ref) - the
  cumulative balance, including its diagnostic `residual` field.

## Mass diagnostics

[`QMagma.MassBudget`](@ref) and [`QMagma.update_mass_budget!`](@ref) track the same
injection history as thickness rather than energy, in two accounts:

- the **magma-volume budget** (`residual`) compares the injected thickness against the
  intruded-rock control-volume content and the magma the transport removed
  (the erupted band, reported by [`QMagma.erupt_melt!`](@ref), plus material displaced past
  a domain boundary, reported by [`QMagma.insert_sill`](@ref)). Sill content is added rather
  than assigned, so this balance closes up to remapping roundoff even when a new sill is
  emplaced into previously intruded material;
- the **melt-content budget** (`melt_residual`) compares it against the stored melt but
  deliberately does not label the remainder as crystallization: it also contains host-rock
  melting, boundary transport, withdrawal discretization, and the realized eruptions.

Both are diagnostics: like the enthalpy budget they measure transport and never correct it.
