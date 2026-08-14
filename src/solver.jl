# Model setup and the nonlinear (Newton + line search) heat-diffusion solver.


"""
    init_model(; nz=101, L=40e3, Geotherm=0, Ttop=400.0, Tbot=0.0,
               Δt=1e3*SecYear, R_lat=Inf, MatParam=nothing, ρ=nothing, Q_L=nothing,
               Conductivity=nothing, HeatCapacity=nothing, Melting=nothing)

Create initial model setup.

The material parameters are assembled here from the individual constitutive laws, so every
entry point (GUI, scripts, tests) shares one definition of the host-rock properties. `ρ`
[kg/m³] is the host-rock *thermal* density, the one that enters heat storage and
conduction; it is a separate role from the melt, crystal and lithostatic densities on
[`EruptionParams`](@ref), which [`check_density_consistency`](@ref) keeps in step. Pass a
ready-made `MatParam` tuple only to reuse one built elsewhere; combining it with any
component keyword is an error because those components would otherwise be ignored.

`R_lat` [m] is the lateral radius of the body whose axis the column represents; it closes
the radial part of the 3-D Laplacian, so heat is also lost sideways (see [`Res!`](@ref)).
`Inf` leaves the model purely 1-D. The initial geotherm is kept as `Params.T_bg`, the
far-field temperature that lateral loss relaxes towards.
"""
function init_model(;
        nz = 101, L = 40.0e3, Geotherm = 0, Ttop = 400.0, Tbot = 0.0, Δt = 1.0e3 * SecYear,
        R_lat = Inf, MatParam = nothing,
        ρ = nothing, Q_L = nothing, Conductivity = nothing,
        HeatCapacity = nothing, Melting = nothing
    )
    R_lat > 0 || throw(ArgumentError("R_lat must be positive"))
    if isnothing(MatParam)
        ρ = something(ρ, 2700.0)
        Q_L = something(Q_L, 2.55e5)
        Conductivity = something(Conductivity, T_Conductivity_Whittington())
        HeatCapacity = something(HeatCapacity, T_HeatCapacity_Whittington())
        Melting = something(Melting, MeltingParam_Assimilation())
        MatParam = (
            SetMaterialParams(;
                Name = "RockMelt", Phase = 0,
                Density = ConstantDensity(; ρ = ρ * kg / m^3),
                LatentHeat = ConstantLatentHeat(Q_L = Q_L * J / kg),
                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0 = 0.0e-7Watt / m^3),
                Conductivity, HeatCapacity, Melting
            ),
        )
    elseif any(x -> !isnothing(x), (ρ, Q_L, Conductivity, HeatCapacity, Melting))
        throw(ArgumentError("MatParam cannot be combined with component material keywords"))
    end

    # Numerics
    Told = zeros(nz)
    T = zeros(nz)
    ρ = zeros(nz)
    Cp = zeros(nz)
    dϕdT = zeros(nz)
    ϕ = zeros(nz)
    Hl = zeros(nz)
    Q = zeros(nz)     # volumetric heat source term [W/m^3]
    Hlat = zeros(nz)  # lateral heat-loss coefficient [W/m^3/K]
    w = zeros(nz)     # vertical advection velocity [m/s]
    k = zeros(nz - 1)
    dz = L / (nz - 1)
    z = -L:dz:0
    T = -Geotherm / 1.0e3 .* Vector(z) .+ Ttop
    Told = -Geotherm / 1.0e3 .* Vector(z) .+ Ttop

    T_bg = copy(T)

    Phases = fill(0, nz)
    Phases_c = fill(0, nz - 1)

    Params = (; Δt, k, ρ, Cp, dϕdT, ϕ, Hl, Q, Hlat, R_lat, T_bg, w, Told, Phases, Phases_c, MatParam, z)
    N = (nz,)
    BC = (; Ttop, Tbot)
    Δ = (dz,)

    return Params, BC, N, Δ, T, z
end

"""
    update_properties!(Params, MatParam)

Evaluate all temperature-dependent material properties (k, Cp, ρ, dϕ/dT, ϕ, latent
heat) from `Params.Told` and cache them in `Params`. These depend only on the *old*
temperature, not on the Newton unknown `T`, so the residual is linear in `T` and the
properties stay constant across the solve. Call this once per timestep (done in
`nonlinear_solution`) instead of recomputing the (expensive) GeoParams evaluations on
every residual / Jacobian / line-search evaluation.
"""
function update_properties!(Params, MatParam)
    args = (T = Params.Told .+ 273.15,)
    args_c = (T = av(Params.Told) .+ 273.15,)
    compute_conductivity!(Params.k, MatParam, Params.Phases_c, args_c)
    compute_heatcapacity!(Params.Cp, MatParam, Params.Phases, args)
    compute_density!(Params.ρ, MatParam, Params.Phases, args)
    compute_dϕdT!(Params.dϕdT, MatParam, Params.Phases, args)
    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, args)
    compute_latent_heat!(Params.Hl, Params.MatParam, Params.Phases, args)
    # 2k/R² at the interior nodes: the radial Laplacian of the lateral Gaussian on the axis
    # (see Res!). k lives on cell centres, so 2*av(k) is its node-centred value times two.
    Params.Hlat[2:(end - 1)] .= 2 .* av(Params.k) ./ Params.R_lat^2
    return Params
end

"""
    Res!(F::AbstractArray, T::AbstractArray, Δ, N, BC, Params, MatParam)

Residual of the backward-Euler heat equation on the column,

`ρ(Cp + Hₗ∂ϕ/∂T) ∂T/∂t = ∂/∂z(k ∂T/∂z) - Hlat(T - T_bg) + Q`.

`Hlat(T - T_bg)` is the lateral loss the third dimension carries away. The column is the
axis of an axisymmetric body whose anomaly `T - T_bg` tapers laterally as a Gaussian of
width `Params.R_lat` — the same shape [`gaussian_thermal_structure`](@ref) exports — and
that Gaussian's radial Laplacian on the axis is `-2(T - T_bg)/R_lat²`. With `R_lat = Inf`
the coefficient is zero and the model is purely 1-D.
"""
function Res!(F::AbstractVector{_T}, T::AbstractVector{_T}, Δ::NTuple, N::NTuple, BC::NamedTuple, Params::NamedTuple, MatParam) where {_T <: Number}

    dz = Δ[1]       # grid spacing
    nz = N[1]       # grid size

    # Material properties (k, Cp, ρ, dϕ/dT, ϕ, Hl) are frozen at Params.Told and cached
    # once per step by update_properties!; they don't depend on the unknown T (the
    # residual is linear in T), so they are read here, not recomputed on every eval.
    I = 2:(nz - 1)
    # (host-rock advection, if any, is applied to Params.Told beforehand via a
    # semi-Lagrangian remap - see `advect_w!` - rather than as a term here)
    F[2:(end - 1)] = Params.ρ[I] .* (Params.Cp[I] + Params.Hl[I] .* Params.dϕdT[I]) .* (T[I] - Params.Told[I]) / Params.Δt - diff(Params.k .* diff(T) / dz) / dz .+ Params.Hlat[I] .* (T[I] .- Params.T_bg[I]) .- Params.Q[I]

    F[1] = T[1] - BC.Tbot
    F[nz] = T[nz] - BC.Ttop

    return F
end

"""
    LineSearch(func::Function, F, x, δx; α=[0.01 0.05 0.1 0.25 0.5 0.75 1.0])

Pick the step size out of `α` that minimizes the residual norm of `func` at `x + α*δx`,
and return it together with that norm.
"""
function LineSearch(func::Function, F, x, δx; α = [0.01 0.05 0.1 0.25 0.5 0.75 1.0])
    Fnorm = zero(α)
    N = length(x)
    for i in eachindex(α)
        func(F, x .+ α[i] .* δx)
        Fnorm[i] = norm(F) / N
    end
    _, i_opt = findmin(Fnorm)
    return α[i_opt], Fnorm[i_opt]
end


"""
    nonlinear_solution(Fup, T, J, colors; tol=1e-8, maxit=100, verbose=true,
                       Δ, N, BC, Params, MatParam) -> (T, converged, it)

Solve one backward-Euler heat-diffusion step by Newton iteration with line search, and
report the temperature, whether the residual norm reached `tol`, and the iteration count.

`Fup` is a work vector shaped like the residual, `T` the initial guess, `J` the sparse
Jacobian, and `colors` its coloring vector from `matrix_colors(J)`. Material properties are
frozen at `Params.Told` by [`update_properties!`](@ref) once per call, so [`Res!`](@ref) is
linear in `T` and one iteration converges.
"""
function nonlinear_solution(
        Fup::Vector, T::Vector, J, colors; tol = 1.0e-8, maxit = 100, verbose = true,
        Δ, N, BC, Params, MatParam
    )

    Res_closed! = (F, T) -> Res!(F, T, Δ, N, BC, Params, MatParam)

    update_properties!(Params, MatParam)   # frozen-coefficient props: once per step, not per eval

    r = zero(Fup)
    err = 1.0e3; it = 0
    while err > tol && it < maxit
        Res_closed!(r, T)     # compute residual

        forwarddiff_color_jacobian!(J, Res_closed!, T, colorvec = colors) # compute jacobian in an in-place manner

        dT = J \ -r    # solve linear system:
        α, err = LineSearch(Res_closed!, r, T, dT)  # optimal step size
        T += α * dT   # update solution
        it += 1
        if verbose
            println("   Nonlinear iteration $it: error = $err, α=$α")
        end
    end

    converged = err <= tol

    return T, converged, it
end
