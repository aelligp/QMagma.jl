# Headless runs and export

## Driving the solver from a script

The GUI has no scripting interface. A headless run calls the solver functions directly.
The example below follows the two branches of the GUI loop and uses its default material
models.

```julia
using QMagma, GeoParams, LinearAlgebra, SparseArrays, SparseDiffTools, Random

SecYear  = 3600*24*365.25
H, γ     = 40.0, 20.0                      # crustal thickness [km], geotherm [ᵒC/km]
Ttop     = 0.0
Tbot     = Ttop + H*γ
Δt       = 100SecYear
nt       = 3000
Tsill       = 1200.0                       # injection temperature [ᵒC]
Sillthick   = 100.0                        # full aperture [m]
Silltop, Sillbot = 10.0, 20.0              # injection window [km]

nz = round(Int, H*1e3/20.0) + 1            # Δz = 20 m
Params, BC, N, Δ, T, z = QMagma.init_model(;
    nz, L=H*1e3, Geotherm=γ, Ttop, Tbot, Δt,
    ρ=2700.0, Q_L=255e3,
    Conductivity=ConstantConductivity(k=3.0),
    HeatCapacity=ConstantHeatCapacity(),
    Melting=MeltingParam_Smooth3rdOrder(),
)
MatParam = Params.MatParam
Params.Told .= T

# Both branches share one accretion history. This example ramps from 0.05 to 0.15 m/yr
# between 50 and 150 kyr; use FluxHistory(:constant; base=0.1/SecYear) for a constant rate.
Params_Q = deepcopy(Params)
T_Q      = deepcopy(T)
ȧ = QMagma.FluxHistory(:ramp;
    base=0.05/SecYear, peak=0.15/SecYear,
    t_start=50e3SecYear, t_end=150e3SecYear)

# tridiagonal sparsity pattern of the 1D residual, with the Dirichlet rows decoupled
J1 = Tridiagonal(ones(N[1]-1), ones(N[1]), ones(N[1]-1))
J1[1, 2] = 0; J1[2, 1] = 0; J1[N[1]-1, N[1]] = 0; J1[N[1], N[1]-1] = 0
Jac    = sparse(Float64.(abs.(J1) .> 0))
colors = matrix_colors(Jac)

F, F_Q = zero(T), zero(T_Q)
rocks  = zero(T)
rng    = MersenneTwister(42)
time   = 0.0
A_inj  = 0.0                               # cumulative injected thickness [m]

for _ in 1:nt
    # discrete sills
    T, converged, iterations = QMagma.nonlinear_solution(F, T, Jac, colors;
        Δ, N, BC, Params, MatParam, verbose=false)
    converged || error("discrete solve failed after $iterations iterations")
    Δh = QMagma.injected_thickness(ȧ, time, Δt)
    for _ in 1:QMagma.sills_due(A_inj, Δh, Sillthick)
        Sill_z0 = rand(rng, -Sillbot*1e3:1.0:-Silltop*1e3)
        T, rocks = QMagma.insert_sill(T, rocks, z;
            Sill_thick=Sillthick, Sill_z0, Sill_T=Tsill)
    end
    global A_inj += Δh
    Params.Told .= T

    # Q_magma
    QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill, ȧ=Δh/Δt, Silltop, Sillbot)
    QMagma.advect_w!(Params_Q)
    # Recompute the source from the advected temperature.
    QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill, ȧ=Δh/Δt, Silltop, Sillbot)
    T_Q, converged, iterations = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors;
        Δ, N, BC, Params=Params_Q, MatParam, verbose=false)
    converged || error("Q_magma solve failed after $iterations iterations")
    Params_Q.Told .= T_Q

    global time += Δt
end
```

Melt fraction is recomputed from the temperature whenever it is needed:

```julia
ϕ = melt_fraction_from_temperature(T, MatParam)
```

To add eruptions, evaluate a trigger on `ϕ` at the end of each step and call
[`QMagma.realize_eruption!`](@ref). The GUI also applies depth, boundary-margin, and
minimum-withdrawal checks described in [Eruptions](eruptions.md).

!!! warning "Δt and the injection interval"
    The injection interval need not be a multiple of `Δt`.
    [`QMagma.sills_due`](@ref) counts the whole apertures completed inside each step and may
    return more than one. Both branches read the same
    [`QMagma.injected_thickness`](@ref), so replacing the constant `ȧ` with a callable
    `ȧ(t)` changes the discrete branch's event frequency and the `Q_magma` branch's rate
    together.

## Export

[`export_thermal_structure`](@ref) writes point fields on a rectilinear 1D, 2D or 3D grid
to JLD2, VTK or both. Field arrays must match the grid dimensions; 1D profiles are written
on a singleton horizontal axis so VTK readers such as ParaView can display them.

```julia
using QMagma

ϕ = melt_fraction_from_temperature(T, MatParam)
export_thermal_structure("run_1d", z; fields=(temperature=T, melt_fraction=ϕ, rocks=rocks))
```

A 1D profile becomes a 2D or 3D field by tapering its anomaly from a background profile
with a Gaussian, [`gaussian_thermal_structure`](@ref). This is what the GUI's
**SAVE JLD2 + VTK** button does, with σ equal to the sill radius and a lateral extent
of ±3σ:

```julia
x  = -15e3:250.0:15e3
T2 = gaussian_thermal_structure(T, T_background, x; sigma=5e3)
export_thermal_structure("run_2d", z; x, fields=(temperature=T2,))

T3 = gaussian_thermal_structure(T, T_background, x; y=x, sigma=5e3)
export_thermal_structure("run_3d", z; x, y=x, fields=(temperature=T3,))
```

Use `formats=(:vtk,)` or `formats=(:jld2,)` to write only one of the two. The returned
vector lists every file written.
