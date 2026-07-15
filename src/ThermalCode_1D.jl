using GeoParams
using ForwardDiff, SparseArrays, SparseDiffTools, LinearAlgebra, Interpolations
using ZirconGrowth

av(x) = (x[2:end]+x[1:end-1])/2


"""
    init_model(;nz=101, L=40e3, Geotherm=0, Ttop=400.0, Tbot=0.0, Δt=1e3*SecYear, MatParam=nothing)

Create initial model setup
"""
function init_model(;nz=101, L=40e3, Geotherm=0, Ttop=400.0, Tbot=0.0, Δt=1e3*SecYear, MatParam=nothing)
    if isnothing(MatParam)
        MatParam     = (SetMaterialParams(Name="RockMelt", Phase=0,
                            Density         = ConstantDensity(ρ=2700kg/m^3),                            # used in the parameterisation of Whittington
                            LatentHeat      = ConstantLatentHeat(Q_L=2.55e5J/kg),
                            RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                            Conductivity    = T_Conductivity_Whittington(),                             #  T-dependent k
                            HeatCapacity    = T_HeatCapacity_Whittington(),                             # T-dependent cp
                            Melting         = MeltingParam_Assimilation()                               # Quadratic parameterization as in Tierney et al.
                            ),
                    )
    end

    # Numerics
    Told        =   zeros(nz)
    T           =   zeros(nz)
    ρ           =   zeros(nz)
    Cp          =   zeros(nz)
    dϕdT        =   zeros(nz)
    ϕ           =   zeros(nz)
    Hl          =   zeros(nz)
    Q           =   zeros(nz)     # volumetric heat source term [W/m^3]
    w           =   zeros(nz)     # vertical advection velocity [m/s]
    k           =   zeros(nz-1)
    dz          =   L/(nz-1)
    z           =   -L:dz:0
    T           =   -Geotherm/1e3.*Vector(z) .+ Ttop

    Phases      =   fill(0,nz)
    Phases_c    =   fill(0,nz-1)

    Params      =   (; Δt, k, ρ, Cp, dϕdT, ϕ, Hl, Q, w, Told, Phases, Phases_c, MatParam, z)
    N           =   (nz,)
    BC          =   (; Ttop, Tbot)
    Δ           =   (dz,)

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
    args   = (T = Params.Told .+ 273.15,)
    args_c = (T = av(Params.Told) .+ 273.15,)
    compute_conductivity!(Params.k, MatParam, Params.Phases_c, args_c)
    compute_heatcapacity!(Params.Cp, MatParam, Params.Phases, args)
    compute_density!(Params.ρ, MatParam, Params.Phases, args)
    compute_dϕdT!(Params.dϕdT, MatParam, Params.Phases, args)
    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, args)
    compute_latent_heat!(Params.Hl, Params.MatParam, Params.Phases, args)
    return Params
end

"""
    Res!(F::AbstractArray, T::AbstractArray, Δ, N, BC)
"""
function Res!(F::AbstractVector{_T}, T::AbstractVector{_T}, Δ::NTuple, N::NTuple, BC::NamedTuple, Params::NamedTuple, MatParam) where _T<:Number

    dz     = Δ[1]       # grid spacing
    nz     = N[1]       # grid size

    # Material properties (k, Cp, ρ, dϕ/dT, ϕ, Hl) are frozen at Params.Told and cached
    # once per step by update_properties!; they don't depend on the unknown T (the
    # residual is linear in T), so they are read here, not recomputed on every eval.
    I          = 2:nz-1
    #  ρ(Cp + Hₗ∂ϕ/∂T) ∂T/∂t = ∂/∂z(k ∂T/∂z) + Q
    # (host-rock advection, if any, is applied to Params.Told beforehand via a
    # semi-Lagrangian remap - see `advect_w!` - rather than as a term here)
    F[2:end-1] = Params.ρ[I].*(Params.Cp[I]  + Params.Hl[I].*Params.dϕdT[I]).*(T[I]-Params.Told[I])/Params.Δt  -   diff(Params.k .* diff(T)/dz)/dz   .-   Params.Q[I];

    F[1]  = T[1]  - BC.Tbot
    F[nz] = T[nz] - BC.Ttop

    return F
end
Res_closed! = (F,T) -> Res!(F, T, Δ, N, BC, Params, MatParam)

function LineSearch(func::Function, F, x, δx;  α = [0.01 0.05 0.1 0.25 0.5 0.75 1.0])
    Fnorm = zero(α)
    N     = length(x)
    for i in eachindex(α)
        func(F, x .+ α[i].*δx)
        Fnorm[i] = norm(F)/N
    end
    _, i_opt = findmin(Fnorm)
    return α[i_opt], Fnorm[i_opt]
end


"""
    Usol = nonlinear_solution(Fup::Vector, U::Vector{<:AbstractArray}, J, colors; tol=1e-8, maxit=100)

Computes a nonlinear solution using a Newton method with line search.
`U` needs to be a vector of abstract arrays, which contains the initial guess of every field
`J` is the sparse jacobian matrix, and `colors` the coloring matrix, usually computed with `matrix_colors(J)`
"""
function nonlinear_solution(Fup::Vector, T::Vector, J, colors; tol=1e-8, maxit=100, verbose=true,
                            Δ, N, BC, Params, MatParam)

    Res_closed! = (F,T) -> Res!(F, T, Δ, N, BC, Params, MatParam)

    update_properties!(Params, MatParam)   # frozen-coefficient props: once per step, not per eval

    r   = zero(Fup)
    err = 1e3; it=0;
    while err>tol && it<maxit
        Res_closed!(r,T)     # compute residual

        forwarddiff_color_jacobian!(J, Res_closed!, T, colorvec = colors) # compute jacobian in an in-place manner

        dT      =   J\-r    # solve linear system:
        α, err  =   LineSearch(Res_closed!, r, T, dT); # optimal step size
        T       +=   α*dT   # update solution
        it      +=1;
        if verbose; println("   Nonlinear iteration $it: error = $err, α=$α"); end
    end

    converged = err <= tol

    return T, converged, it
end

"""
    compute_Q_magma!(Params, MatParam, z; Tsill, ȧ, Silltop, Sillbot)

Smears repeated sill injection into a steady volumetric heat source `Params.Q` [W/m^3],
following

    Q_magma(z,t) = ρₘ ȧ/H [ cp(Tₘ - T(z,t)) + L(1-ϕ(T)) ]

over the injection zone `z ∈ [-Sillbot, -Silltop]` (in m) of thickness `H = Sillbot-Silltop`,
and zero elsewhere. `ȧ = Sillthick/Sill_interval` is the time-averaged accretion rate [m/s].
The first term is the sensible heat magma surrenders cooling from `Tsill` to the local
temperature; the second is the latent heat released crystallizing from ϕ=1 (injected liquid)
down to the local melt fraction ϕ(T). ρ, cp and ϕ are (re-)evaluated here from `Params.Told`,
consistent with how the rest of the residual is linearized.

Also sets `Params.w`, the vertical advection velocity that mimics the host-rock
displacement caused by discrete sill injection. Discrete sills use an elastic
displacement profile (`crack_perp_displacement`, decay scale `r`) that decays away
from the sill rather than persisting at constant amplitude far into the host rock; a
spatially constant `w` outside the injection zone would over-advect heat into distal
regions relative to discrete sills and bias Q_magma high. Here `w` is zero inside the
injection zone, and outside it uses the same `crack_perp_displacement` decay law
measured from the nearest zone edge: it peaks at `ȧ/2` right at the edge (matching the
net accretion rate) and decays toward zero with the same length scale `r` used by
discrete sill injection.
"""
function compute_Q_magma!(Params, MatParam, z; Tsill, ȧ, Silltop, Sillbot, r=5e3)
    args = (T = Params.Told .+ 273.15,)
    compute_heatcapacity!(Params.Cp, MatParam, Params.Phases, args)
    compute_density!(Params.ρ, MatParam, Params.Phases, args)
    compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, args)

    Q_L = NumValue(MatParam[1].LatentHeat[1].Q_L)
    H   = (Sillbot - Silltop)*1e3

    Params.Q .= 0.0
    ind = findall( z .>= -Sillbot*1e3 .&& z .<= -Silltop*1e3 )
    Params.Q[ind] .= Params.ρ[ind].*(ȧ/H).*( Params.Cp[ind].*(Tsill .- Params.Told[ind]) .+ Q_L.*(1.0 .- Params.ϕ[ind]) )

    # host-rock advection: elastic-style decay away from the injection zone, matching
    # the decay law used by discrete sill injection (crack_perp_displacement). Zero
    # inside the zone; peaks at ȧ/2 at each edge and decays outward with scale r.
    zbot, ztop  = -Sillbot*1e3, -Silltop*1e3
    ind_below   = findall(z .< zbot)
    ind_above   = findall(z .> ztop)
    dist_below  = zbot .- z          # distance below the zone (positive below zbot)
    dist_above  = z .- ztop          # distance above the zone (positive above ztop)

    Params.w .= 0.0
    Params.w[ind_below] .= -(ȧ/2).*(1.0 .- dist_below[ind_below]./sqrt.(r^2 .+ dist_below[ind_below].^2))
    Params.w[ind_above] .=  (ȧ/2).*(1.0 .- dist_above[ind_above]./sqrt.(r^2 .+ dist_above[ind_above].^2))

    return Params.Q
end

crack_perp_displacement(z, d; r=5e3) = d.*(1.0 .- abs.(z)./(sqrt.(r^2 .+ z.^2)))

"""
    Tadv = insert_sill!(T,z; Sill_thick=400, Sill_z0=-20e3, Sill_T=1200, SillType=:constant)

Adds a sill to the setup, using a 1D WENO5 advection scheme for a given temperature field `T` on a grid `z`.
Optional parameters are the sill thickness `Sill_thick`, the sill center `Sill_z0`, the sill temperature `Sill_T`.
Advection is done by `SillType`, which can be `:constant` (where rocks above/below are moved with constant displacement
or `:elastic`, where the displacement decreases with distance from the sill.
"""
function insert_sill(T,rocks, z; Sill_thick=400, Sill_z0=-20e3, Sill_T=1200, Sill_phase=1.0, SillType=:elastic)

    # find points above & below sill emplacement level
    z_shift = Vector(z) .-  Sill_z0;
    Displ   = zero(z_shift)

    # shift points above
    id_above = findall(z_shift.>0)
    id_below = findall(z_shift.<0)

    # Assume constant displacement - in elastic case this should decrease with distance from sill
    if SillType==:constant
        Displ[id_above]  .= Sill_thick
        Displ[id_below]  .= -Sill_thick
    elseif SillType==:elastic
        R = 5e3;
        Displ[id_above]  .=  crack_perp_displacement(z_shift[id_above], Sill_thick; r=R)
        Displ[id_below]  .= -crack_perp_displacement(z_shift[id_below], Sill_thick; r=R)
    end

    # use WENO5 to advect the temperature field
    T_adv = semilagrangian_advection(T, Displ, z)

    # set sill temperature
    ind = findall( abs.(z .- Sill_z0) .<= Sill_thick/2)
    T_adv[ind]  .= Sill_T

    # move host rock and previously injected material with the same (elastic or constant)
    # displacement as T. Use the conservative (finite-volume) remap, not semi-Lagrangian
    # interpolation + round: the latter is non-conservative under the diverging injection
    # displacement and ratchets grey away (~2-3% of the injected crust lost per sill, since
    # a boundary cell that interpolates just under 0.5 rounds to 0 and never comes back).
    # The remap keeps Σ rocks·Δz == injected crust thickness; the field stays fractional.
    rock_adv = conservative_advection(rocks, Displ, z)
    rock_adv[ind]  .= Sill_phase

    return T_adv, rock_adv
end

"""
    Tadv = semilagrangian_advection(T, Displ, z)
Do semilagrangian_advection
"""
function semilagrangian_advection(T, Displ, z)

    z_new = z + Displ # advect grid
    interp_linear = linear_interpolation(z_new, T);
    T_adv = interp_linear.(z)

    return T_adv
end

"""
    conservative_advection(field, Displ, z)

Finite-volume (mass-conserving) advection of a per-cell `field` by the displacement
`Displ` [m] on the uniform grid `z`. Unlike [`semilagrangian_advection`](@ref) — which
interpolates the *value* and so lets `Σ field·Δz` drift wherever the displacement
stretches or compresses the grid — this moves each cell's *content* `field·Δz`: cell
edges are displaced (Displ interpolated to the edges) and every parcel deposits its
conserved content onto the fixed cells it overlaps. `Σ field·Δz` is preserved exactly
(up to parcels pushed past the domain boundary).

Use this for the injected-rock phase indicator, whose integral is the injected crust
thickness: repeated non-conservative advection erodes thin grey bands (a one-way loss
of ~2-3% per injection). Temperature keeps using `semilagrangian_advection` — an
intensive field where the stretch-induced drift is small and expected.
"""
function conservative_advection(field, Displ, z)
    n  = length(z)
    Δz = abs(z[2] - z[1])
    ze = vcat(z .- Δz/2, z[end] + Δz/2)                       # n+1 cell edges
    de = linear_interpolation(z, Displ; extrapolation_bc=Line()).(ze)
    xe = ze .+ de                                             # displaced edges
    out = zeros(n)
    for j in 1:n
        L, R = xe[j], xe[j+1]
        w = R - L
        w <= 0 && continue                                   # folded/collapsed parcel
        ρ  = field[j]*Δz/w                                    # conserved content spread over new width
        i1 = clamp(floor(Int, (L - ze[1])/Δz) + 1, 1, n)
        i2 = clamp(ceil(Int,  (R - ze[1])/Δz),     1, n)
        for i in i1:i2
            ov = min(R, ze[i+1]) - max(L, ze[i])
            ov > 0 && (out[i] += ρ*ov/Δz)
        end
    end
    return out
end

"""
    find_eruptible_region(ϕ, z; ϕ_threshold=0.5)

Find the envelope `[z_bot, z_top]` (in the grid's units, e.g. m) of the largest
contiguous run of grid points where the melt fraction `ϕ` exceeds `ϕ_threshold`,
anywhere in the column. Separate melt lenses are not combined: only the single
longest contiguous run is returned, so an isolated near-threshold point far from the
real melt zone can't inflate the reported thickness. Returns `nothing` if no point
exceeds the threshold.
"""
function find_eruptible_region(ϕ, z; ϕ_threshold=0.5)
    above = ϕ .> ϕ_threshold
    any(above) || return nothing

    best_len = 0
    best_lo  = 0
    best_hi  = 0
    run_lo   = 0
    i = 1
    n = length(above)
    while i <= n
        if above[i]
            run_lo = i
            j = i
            while j <= n && above[j]
                j += 1
            end
            run_len = j - run_lo
            if run_len > best_len
                best_len = run_len
                best_lo  = run_lo
                best_hi  = j - 1
            end
            i = j
        else
            i += 1
        end
    end

    return z[best_lo], z[best_hi]
end

"""
    collapse_advection(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2)

Move every grid point's *position* toward the eruption zone `[Erupt_z0-half, Erupt_z0+half]`
using the same elastic decay law `insert_sill` uses to open a sill (largest right at the
band edge, decaying away from it with radius `R`), then interpolate the original field `T`
at those moved positions back onto the fixed grid `z`. This is the direct inverse of
`insert_sill`'s opening displacement: instead of pushing host rock apart to make room for
new material, it pulls host rock together to close a gap left by erupted material.

Points outside the band move toward `Erupt_z0` by exactly enough to fully close their side
of the gap (not just asymptotically approach it - the naive `insert_sill`-style amplitude
falls fractionally short once evaluated at an actual grid point rather than the idealized
edge, which would otherwise leave a sliver of the band uncollapsed). Points inside the band
are swept toward `Erupt_z0` too, packed into a tiny window around it so the displaced grid
stays strictly monotonic (required for the interpolation) without colliding exactly on top
of each other.
"""
function collapse_advection(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)
    zv = Vector(z)
    n = length(zv)
    half = Erupt_thick/2

    Displ = collapse_displacement(zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, R=R, method=method)
    # for the hybrid the top boundary is unpinned (the surface subsides by
    # Erupt_thick): flat extrapolation fills the vacated cells above the subsided
    # surface with the boundary value (Ttop)
    itp = linear_interpolation(zv .+ Displ, Vector(T); extrapolation_bc=Flat())
    T_new = itp.(zv)

    # the handful of grid points exactly at the collapse center can still backtrace onto
    # their own pre-collapse value due to floating-point collisions in the interpolation;
    # patch them with a clean linear interpolation between the nearest already-correctly
    # -advected neighbors just outside the band
    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    if !isempty(ind) && ind[1] > 1 && ind[end] < n
        lo, hi = ind[1]-1, ind[end]+1
        T_new[ind] .= range(T_new[lo], T_new[hi]; length=length(ind)+2)[2:end-1]
    end

    return T_new
end

"""
    collapse_displacement(zv; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)

Build the grid-point displacement field used by `collapse_advection` to close an
erupted band: elastic decay away from the band edges (`crack_perp_displacement`,
radius `R`), rescaled so the nearest exterior point on each side fully closes its
side of the gap, with the interior (band) points packed into a tiny strictly ordered
window around `Erupt_z0`.

`method=:elastic`: both domain boundaries are pinned - the closure is fully absorbed
by stretching the walls. `method=:hybrid`: the floor side is unchanged (elastic rise
toward the vent), but the roof displacement transitions from `-half` at the wall face
to a rigid `-Erupt_thick` subsidence far above (transition radius `Erupt_thick`, wide
enough to keep the warped grid monotonic), so the free surface sinks by the erupted
thickness and the removed volume exits through the unpinned top boundary. The
compression paying for the floor-side stretch is thereby concentrated in the roof
just above the vent.
"""
function collapse_displacement(zv; Erupt_z0, Erupt_thick, R=Erupt_thick/2, method=:elastic)
    half = Erupt_thick/2
    z_shift = zv .- Erupt_z0
    Displ   = zero(z_shift)

    id_above  = findall(z_shift .> half)
    id_below  = findall(z_shift .< -half)
    id_inside = findall(abs.(z_shift) .<= half)

    dist_above = z_shift[id_above]  .- half   # distance outward from the idealized band edge
    dist_below = -z_shift[id_below] .- half

    # anchor distance: how far the nearest grid point on each side actually needs to move
    # to fully reach Erupt_z0, used to rescale the decay law so that point's gap fully closes
    anchor_above = isempty(id_above) ? half : z_shift[id_above[1]]
    anchor_below = isempty(id_below) ? half : -z_shift[id_below[end]]
    scale_above = anchor_above / crack_perp_displacement(0.0, half; r=R)
    scale_below = anchor_below / crack_perp_displacement(0.0, half; r=R)

    if method == :hybrid
        # roof: from -half at the face (walls meet at Erupt_z0) to -Erupt_thick far
        # above (rigid caldera subsidence of the whole overburden)
        Displ[id_above] .= -(Erupt_thick .- crack_perp_displacement(dist_above, half; r=Erupt_thick))
    else
        Displ[id_above] .= -scale_above .* crack_perp_displacement(dist_above, half; r=R)
    end
    Displ[id_below] .=  scale_below .* crack_perp_displacement(dist_below, half; r=R)

    # every point that ends up inside (or right at the edge of) the old band's footprint -
    # both the exterior anchor points (which land exactly on Erupt_z0) and the interior
    # (melt-zone) points - gets packed into a strictly ordered, tiny window around Erupt_z0,
    # so the displaced grid stays strictly monotonic everywhere (required by the
    # interpolation) without any two points colliding on the exact same position
    id_band = sort(vcat(id_inside, isempty(id_above) ? Int[] : id_above[1], isempty(id_below) ? Int[] : id_below[end]))
    unique!(id_band)
    if !isempty(id_band)
        ε = 1e-3*minimum(diff(zv))
        order = sortperm(z_shift[id_band])   # ascending z_shift == ascending zv on id_band
        targets = length(id_band) == 1 ? [0.0] : collect(range(-ε, ε; length=length(id_band)))
        Displ[id_band[order]] .= targets .- z_shift[id_band[order]]
    end

    # pin the bottom boundary (fixed BC) so the warped grid never contracts past it;
    # the top is pinned only for :elastic - for :hybrid the surface subsides and the
    # vacated cells are filled by extrapolation in collapse_advection
    Displ[1] = 0.0
    if method != :hybrid
        Displ[end] = 0.0
    end

    return Displ
end

"""
    erupt_melt!(T, rocks, z; Erupt_z0, Erupt_thick)

Erupt a melt region of thickness `Erupt_thick` [m] centered at `Erupt_z0` [m], the
inverse of `insert_sill`. Two closure mechanisms are available via `method`:

- `:caldera` (default) - roof subsidence: the erupted band's cells are deleted (the
  magma and its heat leave the system) and the entire roof block above the band drops
  rigidly by the band thickness onto the chamber floor, so the free surface subsides
  by the erupted thickness. The vacated cells at the top are filled with the surface
  temperature (the Dirichlet boundary value) - the caldera floor. No rock parcel
  changes temperature: the roof block's profile is translated, not deformed, so the
  column's total heat drops by exactly the erupted band's content (minus the
  negligible cold surface fill) without any artificial cooling around the vent.
  `rocks` subsides the same way, so `sum(rocks)` drops by exactly the erupted band's
  grey content.

- `:elastic` - elastic collapse: host rock on both sides moves toward the vent with
  `collapse_advection`'s elastic decay law (largest at the band edges, decaying with
  distance) and is re-interpolated onto the regular grid; `rocks` follows the same
  displacement (re-binarized by rounding). Temperatures are transported as intensive
  values, so the stretched walls re-cover the vent at their own temperature - the
  column's total heat drops by much less than the erupted band's content, and
  `sum(rocks)` drops by less than the band's grey content.

- `:hybrid` - elastic + caldera: the floor rises elastically toward the vent as in
  `:elastic`, while the roof face drops to meet it and the roof displacement
  transitions to a rigid `-Erupt_thick` subsidence away from the vent, so the free
  surface sinks by the erupted thickness and the removed volume exits through the
  top. Deformation stays concentrated near the vent, temperatures are transported as
  intensive values (no dilution cooling), and the heat debit is approximately the
  erupted band's content (exact when the near-vent material is locally uniform: the
  floor-side stretch duplication is paid by compression of the equally hot roof just
  above the vent).
"""
function erupt_melt!(T, rocks, z; Erupt_z0, Erupt_thick, method=:caldera)
    half = Erupt_thick/2
    zv = Vector(z)
    n  = length(zv)

    if method == :elastic || method == :hybrid
        T_new    = collapse_advection(Vector(T), zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=method)
        # advect the grey phase indicator conservatively with the SAME displacement
        # collapse_advection applies to T, not round(collapse_advection(rocks)): the latter
        # interpolates-then-rounds the 0/1 field under the converging closure, which inflates
        # the grey where the walls stretch (e.g. 700 m -> 1700 m) or annihilates it near the
        # vent (700 m -> 0), and lands it off the co-moving temperature. The grey inside the
        # erupted band leaves with the eruption (zeroed before the remap, else collapse_
        # displacement packs it into one cell as a spike); the surviving grey is remapped
        # conservatively and co-moves with T. Field stays fractional, as after insert_sill.
        Displ    = collapse_displacement(zv; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=method)
        src      = copy(Vector(rocks))
        src[findall(abs.(zv .- Erupt_z0) .<= half)] .= 0.0   # erupted band grey leaves the system
        rock_new = conservative_advection(src, Displ, zv)
        return T_new, rock_new
    end

    T_new    = copy(Vector(T))
    rock_new = copy(Vector(rocks))

    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    isempty(ind) && return T_new, rock_new

    n_band = length(ind)
    i0 = ind[1]
    T_new[i0:n-n_band]    .= T_new[i0+n_band:n]      # roof block drops onto the floor
    rock_new[i0:n-n_band] .= rock_new[i0+n_band:n]
    T_new[n-n_band+1:n]    .= T_new[n]               # subsided surface, filled at Ttop
    rock_new[n-n_band+1:n] .= 0.0

    return T_new, rock_new
end

"""
    collapse_conservative(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2)

Conservative (finite-volume) version of `collapse_advection` for the temperature
field: instead of interpolating temperatures (which preserves the intensive value but
lets the integrated heat grow wherever the walls stretch), the *heat content* of each
material parcel is transported and deposited.

Each grid cell is treated as a material parcel of width `Δz` carrying content
`T*Δz`. The parcels inside the erupted band are deleted - erupted material leaves the
system together with its heat. The surviving parcels' edges move with the same
elastic decay law as `collapse_advection` (rescaled so the two wall faces meet
exactly at `Erupt_z0`, with the domain-boundary edges pinned), and each parcel then
deposits its conserved content onto the cells it overlaps. A parcel stretched to
width `w` therefore reads a diluted temperature `T*Δz/w`: the column's total
`Σ T Δz` drops by exactly the erupted band's content, paid by the stretched walls.
"""
function collapse_conservative(T, z; Erupt_z0, Erupt_thick, R=Erupt_thick/2)
    half = Erupt_thick/2
    zv = Vector(z)
    n  = length(zv)
    Δz = zv[2] - zv[1]

    ind = findall(abs.(zv .- Erupt_z0) .<= half)
    isempty(ind) && return copy(Vector(T))

    # material-cell edges on the (uniform) grid
    ze = vcat(zv .- Δz/2, zv[end] + Δz/2)

    # edge displacements: elastic decay away from the wall faces (the outer edges of
    # the deleted band), rescaled so each face lands exactly on Erupt_z0
    De  = zeros(n+1)
    ref = crack_perp_displacement(0.0, half; r=R)
    j_lo = ind[1]          # edge index of the lower wall face
    j_hi = ind[end] + 1    # edge index of the upper wall face
    d_lo = ze[j_lo] .- ze[1:j_lo]
    d_hi = ze[j_hi:n+1] .- ze[j_hi]
    De[1:j_lo]   .=  (Erupt_z0 - ze[j_lo]) .* crack_perp_displacement(d_lo, half; r=R) ./ ref
    De[j_hi:n+1] .= -(ze[j_hi] - Erupt_z0) .* crack_perp_displacement(d_hi, half; r=R) ./ ref
    De[1]   = 0.0   # pinned domain boundaries, as in collapse_displacement
    De[n+1] = 0.0
    xe = ze .+ De

    # deposit each surviving parcel's conserved content T*Δz onto the fixed grid
    T_new = zeros(n)
    for j in vcat(1:ind[1]-1, ind[end]+1:n)
        L, Rj = xe[j], xe[j+1]
        w = Rj - L
        w <= 0 && continue
        ρT = T[j]*Δz/w
        i1 = clamp(floor(Int, (L  - ze[1])/Δz) + 1, 1, n)
        i2 = clamp(ceil(Int,  (Rj - ze[1])/Δz),     1, n)
        for i in i1:i2
            ov = min(Rj, ze[i+1]) - max(L, ze[i])
            ov > 0 && (T_new[i] += ρT*ov/Δz)
        end
    end

    return T_new
end

"""
    melt_thickness(ϕ, z, z_lo, z_hi)

Melt content `∫ϕ dz` [m] of the depth interval `[z_lo, z_hi]` - the dense-rock
equivalent thickness of the melt held in that band. This is what actually leaves the
column when the band erupts: the crystal framework stays, so the vent closure
amplitude and the erupted volume (`A_sill * melt_thickness`) are based on the melt
content rather than on the bulk band thickness.
"""
function melt_thickness(ϕ, z, z_lo, z_hi)
    zv = Vector(z)
    Δz = zv[2] - zv[1]
    ind = findall(z_lo .<= zv .<= z_hi)
    return sum(ϕ[ind]) * Δz
end

"""
    column_enthalpy(T, z, MatParam, Phases) -> H  [J/m²]

Total heat content of the column per unit area, `∫ ρ(cₚT + Lϕ) dz`: sensible heat plus
the latent heat stored in the melt fraction. Properties are (re)evaluated from the
passed `T` so a before/after pair is self-consistent.

This is the diagnostic for the elastic/hybrid eruption closures: `:caldera` translates
the roof rigidly, so `H` drops by exactly the erupted band's content, but `:elastic`
transports `T` *intensively* (stretched walls re-cover the vent at their own temperature)
so it conserves energy only approximately. Measure `H` before and after `erupt_melt!` to
quantify that drift before trusting reservoir Tt-paths / zircon spectra from those methods.
"""
function column_enthalpy(T, z, MatParam, Phases)
    Δz   = abs(z[2] - z[1])
    args = (T = T .+ 273.15,)
    ρ    = similar(T); Cp = similar(T); Hl = similar(T); ϕ = similar(T)
    compute_density!(ρ, MatParam, Phases, args)
    compute_heatcapacity!(Cp, MatParam, Phases, args)
    compute_latent_heat!(Hl, MatParam, Phases, args)
    compute_meltfraction!(ϕ, MatParam, Phases, args)
    return sum(ρ .* (Cp .* T .+ Hl .* ϕ)) * Δz
end

"""
    erupt_displacement(zm, Erupt_z0, Erupt_thick; method=:caldera, R=Erupt_thick/2)

New position of a material point at depth `zm` under the eruption closure of
`erupt_melt!`, for markers and tracers riding on the host rock. `:caldera`: points
above the erupted band drop rigidly by `Erupt_thick`, points inside land on the
chamber floor, points below stay put. `:elastic`: points inside the band collapse
onto `Erupt_z0`, points outside move toward it with the same elastic decay law as
the host rock in `collapse_advection`. `:hybrid`: like `:elastic` below the band,
while above it the drop transitions from `Erupt_thick/2` at the wall face to the
full rigid `Erupt_thick` subsidence far above the vent.
"""
function erupt_displacement(zm, Erupt_z0, Erupt_thick; method=:caldera, R=Erupt_thick/2)
    half = Erupt_thick/2
    s = zm - Erupt_z0
    if method == :elastic
        abs(s) <= half && return Erupt_z0
        return zm - sign(s)*crack_perp_displacement(abs(s) - half, half; r=R)
    elseif method == :hybrid
        abs(s) <= half && return Erupt_z0
        s < -half && return zm + crack_perp_displacement(-s - half, half; r=R)
        return zm - (Erupt_thick - crack_perp_displacement(s - half, half; r=Erupt_thick))
    end
    zm > Erupt_z0 + half && return zm - Erupt_thick
    zm >= Erupt_z0 - half && return Erupt_z0 - half
    return zm
end

"""
    collapse_markers!(markers, Erupt_z0, Erupt_thick; method=:caldera)

Move marker positions (e.g. the Q_magma injection-zone boundary markers drawn in the
melt-fraction plot) with the eruption closure, consistent with `erupt_melt!` (see
`erupt_displacement` for the two `method`s).
"""
function collapse_markers!(markers, Erupt_z0, Erupt_thick; method=:caldera)
    for (i, zm) in enumerate(markers)
        markers[i] = erupt_displacement(zm, Erupt_z0, Erupt_thick; method=method)
    end
    return markers
end

"""
    collapse_tracers!(tracers, Erupt_z0, Erupt_thick; method=:caldera)

Move the passive tracers with the eruption closure, consistent with `erupt_melt!`
(see `erupt_displacement` for the two `method`s). Tracers inside the erupted band
should already have been removed with `extract_erupted_tracers!` - any remaining
there are placed on the closed vent.
"""
function collapse_tracers!(tracers, Erupt_z0, Erupt_thick; method=:caldera)
    for tracer in tracers
        tracer.z = erupt_displacement(tracer.z, Erupt_z0, Erupt_thick; method=method)
    end
    return tracers
end

"""
    extract_erupted_tracers!(tracers, Erupt_z0, Erupt_thick)

Remove all tracers within the erupted band `[Erupt_z0 - Erupt_thick/2, Erupt_z0 + Erupt_thick/2]`
from `tracers` (in place) and return them as a separate `Vector{Tracer}`, so zircon ages can
be computed specifically from the population of tracers that has just erupted.
"""
function extract_erupted_tracers!(tracers, Erupt_z0, Erupt_thick)
    zlo, zhi = Erupt_z0 - Erupt_thick/2, Erupt_z0 + Erupt_thick/2
    is_erupted = [zlo <= tracer.z <= zhi for tracer in tracers]
    erupted = tracers[is_erupted]
    deleteat!(tracers, is_erupted)
    return erupted
end

# =====================================================================================
#  OVERPRESSURE TRIGGER  (Degruyter & Huber 2014 / Townsend 2021, 3-phase)
# =====================================================================================
# A physically richer alternative to the GUI's single-β_eff elastic box model: a lumped
# 3-phase (melt/crystal/gas) chamber whose overpressure ΔP = P - P_lith evolves from a
# master ODE driven by the column's own Ṫ and ϕ̇. Gas exsolution/water solubility set the
# compressibility, and a depth cutoff stalls dikes from chambers deeper than z_erupt_max.
# This is a *trigger*: the actual erupted band is removed with erupt_melt! as elsewhere.

"""
    EruptionParams

Tunable parameters for the 3-phase overpressure trigger. Thermodynamic / solubility
constants follow Degruyter & Huber (2014); treat the defaults as order-of-magnitude
values and check the paper's Table 1 for a specific system.

- `ϕ_erupt`  : mobile-melt threshold; the eruptible chamber is the mush ϕ ≥ ϕ_erupt (0.5)
- `ΔP_crit`  : roof-failure overpressure [Pa] (D&H: ~10-40 MPa)
- `ϕ_g_crit` : gas-volume-fraction lock-up (shut-off) [-]
- `ΔP_relax` : overpressure left after an eruption relaxes the chamber [Pa]
- `z_erupt_max`: max chamber-centroid depth [m] at which eruptions can still occur
- `β_r`,`η_r`: host-rock stiffness [Pa] and wall relaxation viscosity [Pa·s]
- `ρ_melt`,`ρ_x`: melt / crystal densities [kg/m³]
- `m_w`      : total water mass fraction of the magma [-]
- `A_visc`,`B_gas`,`G_act`: Arrhenius wall-relaxation-viscosity constants (D&H 2014 Table 1),
  `η(T) = A_visc·exp(G_act/(B_gas·T))`; used to compute `η_r` from the crustal wall T.
- `ρ_crust`,`g`: crustal density [kg/m³] and gravity [m/s²] for the lithostatic reference
"""
Base.@kwdef mutable struct EruptionParams
    ϕ_erupt  :: Float64 = 0.5
    ΔP_crit  :: Float64 = 20e6
    ϕ_g_crit :: Float64 = 0.5
    ΔP_relax :: Float64 = 0.0
    z_erupt_max :: Float64 = 10e3
    β_r      :: Float64 = 1e10
    η_r      :: Float64 = 1e19
    ρ_melt   :: Float64 = 2400.0
    ρ_x      :: Float64 = 2700.0
    m_w      :: Float64 = 0.05
    A_visc   :: Float64 = 4.25e7        # D&H Table 1: viscosity-law prefactor [Pa·s]
    B_gas    :: Float64 = 8.314         # molar gas constant [J/mol/K]
    G_act    :: Float64 = 141e3         # activation energy for creep [J/mol]
    ρ_crust  :: Float64 = 2700.0
    g        :: Float64 = 9.81
end

"""
    EruptionState

Mutable chamber state carried across timesteps: absolute pressure `P`, lithostatic
reference `P_lith`, gas volume fraction `ϕ_g`, and the previous mush (T,ϕ) for the
fixed-P density-rate source term.
"""
Base.@kwdef mutable struct EruptionState
    P        :: Float64 = 0.0
    P_lith   :: Float64 = 0.0
    ϕ_g      :: Float64 = 0.0
    T_prev   :: Float64 = NaN          # mush-mean T [K] at previous step
    ϕ_prev   :: Float64 = NaN          # mush-mean ϕ at previous step
    inv_βm   :: Float64 = 0.0          # magma compressibility 1/β_m at the last step [1/Pa]
    h_erupt  :: Float64 = 0.0          # melt thickness [m] drained by eruptions during the last step
    m_diss   :: Float64 = 0.0          # dissolved H₂O mass fraction (per magma mass), Liu 2005
    X_g      :: Float64 = 0.0          # exsolved H₂O gas mass fraction (per magma mass), D&H eq.18
    ρ_gas    :: Float64 = 0.0          # exsolved-gas density [kg/m³] at the last step (modified RK)
    η_r      :: Float64 = 0.0          # wall relaxation viscosity used at the last step [Pa·s]
    ϕ_mush   :: Float64 = 0.0          # mush-mean melt fraction driving the last step
    init     :: Bool    = false
end

"""
    rho_gas_RK(P, T_K) -> ρ_g [kg/m³]

Modified Redlich–Kwong parameterisation for H₂O gas density (Huber et al. 2010, D&H eq.
A.1), valid ~873<T<1173 K and 30<P<400 MPa (chambers sit in this box). `P` in Pa (used as
bar internally), `T_K` in K (used as °C). Ported from the reference `eos_g.m`.
"""
function rho_gas_RK(P, T_K)
    Tc = T_K - 273.15
    Pb = max(P, 0.0)*1e-5                      # Pa -> bar
    return 1e3*(-112.528*Tc^-0.381 + 127.811*Pb^-1.135 + 112.04*Tc^-0.411*Pb^0.033)
end

"""
    water_gas_partition(P, T_K, ϕ_melt, ep) -> (m_diss, X_g, ρ_g, m_eq)

H₂O speciation + gas density at `P` [Pa], `T_K` [K], `ϕ_melt`. Dissolved water per *magma*
mass `m_diss = m_eq·ϕ_melt` from the Liu et al. (2005) saturation `m_eq` (per *melt* mass,
T-dependent, silicic, H₂O-only so P_w=P); exsolved-gas mass fraction `X_g = max(0, m_w −
m_diss)` (D&H 2014 eq. 18, water conserved); gas density `ρ_g` from the modified
Redlich–Kwong EOS ([`rho_gas_RK`](@ref)). Both [`mixture_density`](@ref) and the chamber
diagnostics on [`EruptionState`](@ref) read this, so the physics lives in one place.
CO₂ is not modelled (X_co2≡0); there is no CO₂ phase to track.
"""
function water_gas_partition(P, T_K, ϕ_melt, ep::EruptionParams)
    Pp     = max(P, 0.0)
    # Liu et al. (2005): coefficients give wt%, the trailing 1e-2 → mass fraction (matches
    # reference exsolve_silicic.m). As the magma crystallizes (ϕ_melt↓) the melt dissolves
    # less, so X_g *rises* — second boiling, D&H's dominant pressurization for small chambers.
    Pm     = Pp*1e-6                          # Pa -> MPa
    # Liu 2005 is only calibrated to ~500 MPa; beyond that the -Pm^1.5 term drives the
    # saturation negative, which would make X_g blow past m_w (and corrupt ρ/ϕ_g). A
    # saturation can't be negative, and dissolved water can't exceed the total m_w — clamp
    # both, so m_diss and X_g stay in [0, m_w] (they are mass fractions).
    m_eq   = max(0.0, 1e-2*((354.94*sqrt(Pm) + 9.623*Pm - 1.5223*Pm^1.5)/T_K + 1.2439e-3*Pm^1.5))
    ρ_g    = max(rho_gas_RK(Pp, T_K), 1e-6)   # modified Redlich–Kwong (was ideal gas)
    m_diss = min(m_eq*ϕ_melt, ep.m_w)         # dissolved H₂O per magma mass, ≤ total water
    X_g    = ep.m_w - m_diss                  # exsolved-gas mass fraction ∈ [0, m_w]
    return m_diss, X_g, ρ_g, m_eq
end

"""
    mixture_density(P, T_K, ϕ_melt, ep) -> (ρ, ϕ_g)

Three-phase (melt + crystal + exsolved gas) mixture density [kg/m³] and gas volume
fraction at pressure `P` [Pa], temperature `T_K` [K] and melt fraction `ϕ_melt`. The H₂O
speciation and gas density come from [`water_gas_partition`](@ref).
"""
function mixture_density(P, T_K, ϕ_melt, ep::EruptionParams)
    _, Xg, ρ_g, _ = water_gas_partition(P, T_K, ϕ_melt, ep)
    ρ_c   = ϕ_melt*ep.ρ_melt + (1-ϕ_melt)*ep.ρ_x   # condensed (melt+crystal) density
    Vg    = Xg/ρ_g
    Vc    = (1-Xg)/ρ_c
    ρ     = 1.0/(Vg + Vc)
    ϕ_g   = Vg/(Vg + Vc)
    return ρ, ϕ_g
end

"""
    wall_relaxation_viscosity(ep, T_wall_K) -> η_r [Pa·s]

Arrhenius wall-relaxation viscosity at the chamber-wall temperature `T_wall_K` [K]
(D&H 2014 Appendix A.5, minimal wall-T tier): `η_r = A_visc·exp(G_act/(B_gas·T_wall))`,
clamped to `[1e17, 1e24]` so a cold wall can't freeze the overpressure ODE. As the crust
matures (wall heats) η_r drops and the chamber crosses from *storing* into *erupting* on
its own, removing the free GUI knob. This is the geometry-free tier; the full spherical
radial integral (D&H A.18–A.21) is deferred (1-D-Cartesian makes it only *effective*).
"""
function wall_relaxation_viscosity(ep::EruptionParams, T_wall_K)
    return clamp(ep.A_visc*exp(ep.G_act/(ep.B_gas*T_wall_K)), 1e17, 1e24)
end

"""
    eruptible_mush(ϕ, z; ϕ_erupt=0.5) -> (ind, V_e, z_centroid)

Indices of the eruptible (mobile-mush) cells `ϕ ≥ ϕ_erupt`, the eruptible thickness
`V_e = Σ ϕ·dz` [m] (melt-weighted, per unit area), and the melt-weighted centroid depth.
"""
function eruptible_mush(ϕ, z; ϕ_erupt=0.5)
    dz  = abs(z[2] - z[1])
    ind = findall(ϕ .>= ϕ_erupt)
    isempty(ind) && return ind, 0.0, 0.0
    V_e  = sum(ϕ[ind])*dz
    zc   = sum(ϕ[ind].*z[ind]) / sum(ϕ[ind])
    return ind, V_e, zc
end

"""
    step_overpressure!(state, ep, T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid=nothing)

Integrate the master overpressure ODE over one thermal step. Equating the mechanical
(elastic+viscous shell) and thermodynamic volumetric rates:

    (1/β_r + 1/β_m) dP/dt = Ṁ_in/(ρV) - (1/ρ)dρ/dt|_{T,ϕ} - (P-P_lith)/η_r

where `1/β_m = (1/ρ)∂ρ/∂P` (gas-dominated compressibility, by finite difference), the
fixed-P density rate is finite-differenced from the previous mush (T,ϕ), and recharge
`Ṁ_in = ȧ·ρ_melt` [kg/m²/s] comes from the accretion rate `ȧ` [m/s].

**Sub-stepping + internal drain (mass-conserving eruptions).** The overpressure builds
much faster than the thermal Δt (recharge can add ≫ΔP_crit per thermal step — a stiff
ODE, gap §1.8), so a single Euler step overshoots and, cycled every step, erupts more
melt than was recharged. Instead we sub-step so ΔP moves ≲¼·ΔP_crit per sub-step; and
whenever ΔP crosses `ΔP_crit` (and not gas-locked, and — if `z_centroid` is given —
shallow enough), the chamber **drains** the stored volume `V_e·(ΔP-ΔP_relax)·(1/β_r+1/β_m)`
back to `ΔP_relax`. Each drain releases exactly the volume its overpressure represents, so
summed over the step the erupted melt equals the recharge that came in (mass conserved,
gap §1.2). The total drained melt thickness for the step is returned in `state.h_erupt`
(0 if the chamber only charged); the caller does the physical band removal with it.
"""
function step_overpressure!(state::EruptionState, ep::EruptionParams,
                            T_mush_K, ϕ_mush, V_e, ȧ, Δt; z_centroid=nothing)
    ρ, ϕg   = mixture_density(state.P, T_mush_K, ϕ_mush, ep)
    state.ϕ_g   = ϕg
    state.h_erupt = 0.0
    # H₂O speciation diagnostics (dissolved / exsolved mass fractions, gas density) for tracking
    state.m_diss, state.X_g, state.ρ_gas, _ = water_gas_partition(state.P, T_mush_K, ϕ_mush, ep)
    state.η_r = ep.η_r    # record the wall viscosity the caller set (per-model, ep is shared)
    state.ϕ_mush = ϕ_mush # mush-mean melt fraction driving the split (for tracking)
    if !state.init || V_e <= 0
        state.T_prev = T_mush_K
        state.ϕ_prev = ϕ_mush
        state.init   = true
        return state
    end

    # magma compressibility 1/β_m = (1/ρ)∂ρ/∂P  (finite difference at fixed T,ϕ)
    dP       = max(1e3, 1e-4*max(state.P, 1e5))
    ρp, _    = mixture_density(state.P + dP, T_mush_K, ϕ_mush, ep)
    inv_βm   = (ρp - ρ)/(ρ*dP)
    state.inv_βm = inv_βm

    # thermodynamic source: -(1/ρ) dρ/dt at fixed P (T & ϕ change only). Held over the
    # sub-steps (T, ϕ only change on the thermal Δt).
    ρ_old, _ = mixture_density(state.P, state.T_prev, state.ϕ_prev, ep)
    dρdt_TP  = (ρ - ρ_old)/Δt
    Ṁ_in     = ȧ*ep.ρ_melt
    S        = Ṁ_in/(ρ*V_e) - dρdt_TP/ρ
    inv_βr   = 1.0/ep.β_r
    invβ     = inv_βr + inv_βm

    # sub-step count: keep |ΔP| change per sub-step ≲ ¼·ΔP_crit so the drain doesn't
    # overshoot (which is what over-counted erupted volume before)
    dPdt0    = (S - (state.P - state.P_lith)/ep.η_r) / invβ
    # clamp in float space BEFORE the Int cast: a soft (floored) η_r can make |dPdt0| huge,
    # and ceil(Int, >9.2e18) overflows Int64 (InexactError) before the clamp can cap it.
    nsub     = round(Int, clamp(ceil(abs(dPdt0)*Δt / (0.25*ep.ΔP_crit)), 1.0, 10_000.0))
    dt_sub   = Δt/nsub

    can_drain = ϕg < ep.ϕ_g_crit &&
                (z_centroid === nothing || abs(z_centroid) <= ep.z_erupt_max)
    h_out = 0.0
    for _ in 1:nsub
        dPdt      = (S - (state.P - state.P_lith)/ep.η_r) / invβ
        state.P  += dPdt*dt_sub
        if can_drain && (state.P - state.P_lith) >= ep.ΔP_crit
            h_out  += V_e*(state.P - state.P_lith - ep.ΔP_relax)*invβ
            state.P = state.P_lith + ep.ΔP_relax
        end
    end

    state.h_erupt = h_out
    state.T_prev  = T_mush_K
    state.ϕ_prev  = ϕ_mush
    return state
end

"""
    init_eruption!(state, ep, z_centroid)

Initialise the chamber at lithostatic pressure (zero overpressure) for a chamber whose
melt-weighted centroid sits at depth `z_centroid` [m, negative down]:
`P_lith = ρ_crust·g·|z_centroid|`, `P = P_lith`.
"""
function init_eruption!(state::EruptionState, ep::EruptionParams, z_centroid)
    state.P_lith = ep.ρ_crust*ep.g*abs(z_centroid)
    state.P      = state.P_lith
    return state
end

"""
    update_lithostatic!(state, ep, z_centroid)

Keep the lithostatic reference tracking the (moving) chamber centroid. First call
initialises at lithostatic (as [`init_eruption!`](@ref)); later calls shift `P` by the
change in `P_lith` so `ΔP = P - P_lith` stays continuous while the reference follows the
chamber. Call once per timestep before [`step_overpressure!`](@ref).
"""
function update_lithostatic!(state::EruptionState, ep::EruptionParams, z_centroid)
    P_lith = ep.ρ_crust*ep.g*abs(z_centroid)
    if state.P_lith == 0.0 && state.P == 0.0
        state.P = P_lith
    else
        state.P += P_lith - state.P_lith
    end
    state.P_lith = P_lith
    return state
end

"""
    overpressure_erupts(state, ep, z_centroid) -> Bool

Eruption criterion for the 3-phase trigger: roof failure `ΔP ≥ ΔP_crit`, not gas-locked
`ϕ_g < ϕ_g_crit`, and the chamber shallower than `z_erupt_max`. The caller removes the
band with `erupt_melt!` and then relaxes `state.P = state.P_lith + ep.ΔP_relax`.
"""
function overpressure_erupts(state::EruptionState, ep::EruptionParams, z_centroid)
    (state.P - state.P_lith) >= ep.ΔP_crit &&
        state.ϕ_g < ep.ϕ_g_crit &&
        abs(z_centroid) <= ep.z_erupt_max
end

"""
    _overpressure_selfcheck()

Runnable assert-based sanity check for the 3-phase overpressure trigger (no framework).
"""
function _overpressure_selfcheck()
    ep = EruptionParams(ΔP_crit=20e6, ϕ_erupt=0.5, z_erupt_max=15e3)

    # RK gas EOS (item 1): hotter gas lighter, higher-P gas denser, positive over the box
    @assert rho_gas_RK(1e8, 1123.15) > rho_gas_RK(1e8, 1173.15)   # hotter -> lighter
    @assert rho_gas_RK(3e8, 1123.15) > rho_gas_RK(1e8, 1123.15)   # higher P -> denser
    @assert rho_gas_RK(3e8, 1123.15) > 0 && rho_gas_RK(3e7, 1173.15) > 0

    # Liu 2005 solubility (item 4): ~5 wt% at 200 MPa/1200 K, more soluble at higher P
    meq(Pmpa) = 1e-2*((354.94*sqrt(Pmpa)+9.623*Pmpa-1.5223*Pmpa^1.5)/1200.0 + 1.2439e-3*Pmpa^1.5)
    @assert 0.03 < meq(200.0) < 0.07
    @assert meq(400.0) > meq(200.0)

    # wall-T Arrhenius η_r (item 2a): colder wall -> stiffer. Test at realistic country-rock
    # temps (500-650 K, in-range); at magmatic T the raw value falls below the 1e17 floor.
    @assert wall_relaxation_viscosity(ep, 500.0) > wall_relaxation_viscosity(ep, 650.0)
    @assert 1e17 <= wall_relaxation_viscosity(ep, 500.0) <= 1e24
    @assert wall_relaxation_viscosity(ep, 1200.0) == 1e17    # hot wall clamps to the floor

    # mixture density: higher P dissolves more water -> less gas -> denser, ϕ_g ∈ [0,1]
    ρlo, glo = mixture_density(5e6,  1200.0, 0.7, ep)
    ρhi, ghi = mixture_density(3e8,  1200.0, 0.7, ep)
    @assert 0 <= glo <= 1 && 0 <= ghi <= 1
    @assert ρhi > ρlo && ghi < glo

    # second boiling (D&H's dominant trigger): at fixed P,T, crystallizing the magma
    # (ϕ_melt↓) must EXSOLVE gas (ϕ_g↑), because the shrinking melt dissolves less water
    _, g_wet = mixture_density(1e8, 1200.0, 0.8, ep)   # crystal-poor
    _, g_dry = mixture_density(1e8, 1200.0, 0.4, ep)   # crystal-rich
    @assert g_dry > g_wet

    z = collect(-30e3:100.0:0.0)
    ϕ = [(-15e3 <= zi <= -12e3) ? 0.8 : 0.1 for zi in z]   # a mobile mush at ~13.5 km
    ind, V_e, zc = eruptible_mush(ϕ, z; ϕ_erupt=ep.ϕ_erupt)
    @assert !isempty(ind) && V_e > 0 && zc < 0

    st = EruptionState(); init_eruption!(st, ep, zc)
    @assert st.P ≈ st.P_lith
    @assert !overpressure_erupts(st, ep, zc)          # at lithostatic: no eruption

    # pressurize with recharge and near-rigid, slowly-relaxing walls; disable draining
    # (unreachable ΔP_crit) to test that recharge alone builds overpressure
    ep.η_r = 1e30
    ep.ΔP_crit = 1e15
    for _ in 1:200
        step_overpressure!(st, ep, 1200.0+273.15, 0.8, V_e, 1e-9, 1e10)
    end
    @assert st.P - st.P_lith > 0 && st.h_erupt == 0.0  # recharge builds ΔP, no eruption

    # with a reachable threshold the chamber drains, and the erupted melt over a run is
    # mass-conserving: it can't exceed the melt recharged (this over-counted before)
    ep.ΔP_crit = 20e6
    st2 = EruptionState(); init_eruption!(st2, ep, zc)
    ȧ_t, Δt_t, nstep = 1e-9, 1e11, 60
    step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t)   # init call
    erupted = 0.0
    for _ in 1:nstep
        step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t; z_centroid=zc)
        erupted += st2.h_erupt
    end
    recharge = ȧ_t*Δt_t*nstep                          # total melt recharged [m]
    @assert erupted > 0                                # it erupts
    @assert erupted <= 1.05*recharge                   # never more than came in (conserved)

    # depth cutoff: too-deep chamber never drains even when over-pressured
    st2.P = st2.P_lith + 5*ep.ΔP_crit
    step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t; z_centroid=-30e3)
    @assert st2.h_erupt == 0.0                          # 30 km > z_erupt_max (15 km): no drain

    # depth cutoff: same overpressure, chamber too deep -> no eruption
    ep_deep = EruptionParams(ΔP_crit=20e6, z_erupt_max=10e3)   # zc ≈ -13.5 km > 10 km
    st.P = st.P_lith + 10*ep.ΔP_crit
    @assert overpressure_erupts(st, ep, zc)            # shallow enough (15 km max)
    @assert !overpressure_erupts(st, ep_deep, zc)      # too deep (10 km max)

    # lithostatic reference tracks a migrating chamber without jumping ΔP
    st_m = EruptionState()
    update_lithostatic!(st_m, ep, -13e3)
    @assert st_m.P ≈ st_m.P_lith                       # first call initialises at lithostatic
    st_m.P += 5e6                                       # build some overpressure
    update_lithostatic!(st_m, ep, -14e3)               # chamber centroid deepens
    @assert st_m.P_lith ≈ ep.ρ_crust*ep.g*14e3
    @assert st_m.P - st_m.P_lith ≈ 5e6                 # ΔP continuous across the shift

    println("overpressure trigger self-check passed")
    return true
end

"""
    advect_w!(Params)

Semi-Lagrangian advection of `Params.Told` by the vertical velocity field `Params.w`
[m/s] over one timestep `Params.Δt`, using the same `semilagrangian_advection` scheme
as discrete sill injection (`insert_sill`). Mimics the host-rock displacement caused by
continuous magma accretion (e.g. as set by `compute_Q_magma!`) in a way consistent with
how discrete sills displace the column, rather than via an upwind advection term in the
residual.
"""
function advect_w!(Params)
    Displ = Params.w .* Params.Δt
    Params.Told .= semilagrangian_advection(Params.Told, Displ, Params.z)
    return Params.Told
end

"""
    advect_markers!(markers, Params)

Advance a vector of passive marker depths `markers` [m] by one timestep using the
host-rock advection velocity `Params.w` [m/s], interpolated onto the (possibly
off-grid) marker positions. Used to visualize how far host rock outside the
injection zone has moved under `compute_Q_magma!`/`advect_w!`, analogous to the
`rocks` field tracked for discrete sill injection.
"""
function advect_markers!(markers, Params)
    w_interp = linear_interpolation(Params.z, Params.w; extrapolation_bc=Line())
    markers .+= w_interp.(markers) .* Params.Δt
    return markers
end

"""
    Tracer

Passive tracer carrying its own temperature-time history, in the same ragged
per-tracer-vector convention used by MagmaThermoKinematics.jl (and consumed
directly by ZirconGrowth.jl's `simulate_from_cooling_path(time_Myr, T_C)`):
`time_vec` is in Myr, `T_vec` and `T` are in °C. Melt fraction `phi` (0-1) is
tracked alongside temperature so eruptibility (whether a tracer is currently
within the eruptible melt-fraction window) can be evaluated later.

# Fields
- `z::Float64`: current depth [m]
- `T::Float64`: current temperature [°C]
- `phi::Float64`: current melt fraction [-]
- `phase::Int`: 0 = host rock, 1 = injected sill/magma material
- `time_vec::Vector{Float64}`: recorded times [Myr], growing over the run
- `T_vec::Vector{Float64}`: recorded temperatures [°C], growing over the run
"""
mutable struct Tracer
    z        :: Float64
    T        :: Float64
    phi      :: Float64
    phase    :: Int
    time_vec :: Vector{Float64}
    T_vec    :: Vector{Float64}
end

"""
    init_tracers(Silltop, Sillbot; n=20)

Seed `n` passive tracers uniformly across the injection zone `z ∈ [-Sillbot, -Silltop]`
(in km, as elsewhere in the GUI), with `phase=0` (host rock) and an as-yet-empty
temperature-time history.
"""
function init_tracers(Silltop, Sillbot; n=20)
    zs = range(-Sillbot*1e3, -Silltop*1e3, length=n)
    return [Tracer(z, 0.0, 0.0, 0, Float64[], Float64[]) for z in zs]
end

"""
    add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=5)

Seed `n` new passive tracers within a freshly-inserted sill centered at `Sill_z0` [m]
with thickness `Sill_thick` [m], at temperature `Sill_T` [°C] and `phase=1` (injected
material), and append them to `tracers`.
"""
function add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=5)
    zs = range(Sill_z0 - Sill_thick/2, Sill_z0 + Sill_thick/2, length=n)
    append!(tracers, [Tracer(z, Sill_T, 1.0, 1, Float64[], Float64[]) for z in zs])
    return tracers
end

"""
    add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=2)

Seed `n` new passive tracers at the center of the injection zone `z ∈ [-Sillbot, -Silltop]`
(in km), at temperature `Tsill` [°C] and `phase=1` (injected material), and append them to
`tracers`. Used with `compute_Q_magma!`/`advect_w!` to keep replenishing tracers at the
zone as the host rock is continuously advected away from its center, analogous to
`add_sill_tracers!` for discrete sill injection.
"""
function add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=2)
    z0 = -(Silltop + Sillbot)/2*1e3
    append!(tracers, [Tracer(z0, Tsill, 1.0, 1, Float64[], Float64[]) for _ in 1:n])
    return tracers
end

"""
    advect_tracers!(tracers, Params)

Advance each tracer's depth `tracer.z` by one timestep using the host-rock advection
velocity `Params.w` [m/s], exactly as `advect_markers!` does for the injection-zone
boundary markers. Use this for the `compute_Q_magma!`/`advect_w!` path; discrete sill
injection does not populate `Params.w` and should use `advect_tracers_sill!` instead.
"""
function advect_tracers!(tracers, Params)
    w_interp = linear_interpolation(Params.z, Params.w; extrapolation_bc=Line())
    for tracer in tracers
        tracer.z += w_interp(tracer.z) * Params.Δt
    end
    return tracers
end

"""
    advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic)

Displace each tracer's depth `tracer.z` by the same per-point displacement law that
`insert_sill` applies to the temperature/rock fields when a sill of thickness
`Sill_thick` [m] is emplaced at `Sill_z0` [m]: zero inside the sill, and
`±crack_perp_displacement` outside it (or `±Sill_thick` for `SillType=:constant`).
Use this for the discrete-sill path; `compute_Q_magma!`/`advect_w!` does not displace
the column this way and should use `advect_tracers!` instead.
"""
function advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic, r=5e3)
    for tracer in tracers
        z_shift = tracer.z - Sill_z0
        if abs(z_shift) <= Sill_thick/2
            continue   # inside the sill: no host-rock displacement to apply
        elseif SillType == :constant
            tracer.z += z_shift > 0 ? Sill_thick : -Sill_thick
        elseif SillType == :elastic
            d = crack_perp_displacement(z_shift, Sill_thick; r=r)
            tracer.z += z_shift > 0 ? d : -d
        end
    end
    return tracers
end

"""
    update_tracers_T!(tracers, T, z, time_Myr, phi=nothing)

Interpolate the temperature field `T` (defined on grid `z` [m]) onto each tracer's
current depth, update `tracer.T`, and append `(time_Myr, tracer.T)` to the tracer's
`time_vec`/`T_vec` history. If the melt fraction field `phi` (on the same grid `z`)
is given, also interpolate it onto each tracer's depth and update `tracer.phi` —
used later to check whether a tracer is within the eruptible melt-fraction window.
"""
function update_tracers_T!(tracers, T, z, time_Myr, phi=nothing)
    T_interp = linear_interpolation(z, T; extrapolation_bc=Line())
    phi_interp = phi === nothing ? nothing : linear_interpolation(z, phi; extrapolation_bc=Line())
    for tracer in tracers
        tracer.T = T_interp(tracer.z)
        push!(tracer.time_vec, time_Myr)
        push!(tracer.T_vec, tracer.T)
        if phi_interp !== nothing
            tracer.phi = phi_interp(tracer.z)
        end
    end
    return tracers
end

"""
    volume_averaged_age(result::ZirconGrowth.SimulationResult) -> Float64

Compute the volume-averaged crystallisation age (in years, before the end of the
cooling path) of a single zircon crystal, exactly as
`MagmaThermoKinematics.volume_averaged_age` does: each concentric growth shell is
weighted by its volume (∝ `r[i+1]^3 - r[i]^3`) and assigned the age of its midpoint;
shells with zero or negative growth are excluded.
"""
function volume_averaged_age(result::ZirconGrowth.SimulationResult)
    t = result.time_years
    r = result.zircon_radius_um

    age_sum = 0.0
    vol_sum = 0.0
    t_end   = t[end]

    for i in 1:(length(r) - 1)
        dV = r[i+1]^3 - r[i]^3
        dV <= 0 && continue
        age_mid  = t_end - 0.5*(t[i] + t[i+1])
        age_sum += age_mid * dV
        vol_sum += dV
    end

    return vol_sum > 0 ? age_sum / vol_sum : 0.0
end

"""
    compute_zircon_ages(tracers; nx=100, elements=ZirconGrowth.default_element_data(),
                         return_results=false, T_zr_min=650.0, t_ref_Myr=nothing)

Run the ZirconGrowth.jl crystal-growth model on each tracer's `(time_vec, T_vec)`
cooling path (Myr, °C) and return a `NamedTuple` with one entry per successfully
simulated tracer:

- `age_years`: volume-averaged crystallisation age [yr] (see [`volume_averaged_age`](@ref))
- `zircon_radius_um`: final crystal radius [µm]

Skipped tracers: fewer than 2 recorded time steps, never hotter than `T_zr_min` [°C]
(too cold to ever grow zircon - saves the expensive simulation and keeps fake zero-age
entries out of the spectra), or no zircon growth beyond the seed radius.

All ages are referenced to a **common clock** `t_ref_Myr` (default: the latest recorded
time across `tracers`): an erupted-cargo tracer whose history stops at eruption gets the
eruption-to-reference offset added, so erupted-cargo and whole-reservoir spectra are
directly comparable. Each cooling path is passed to ZirconGrowth birth-relative, so the
internal uniform time grid resolves the tracer's actual history rather than padding back
to t=0.

Set `return_results=true` to also get the full `Vector{ZirconGrowth.SimulationResult}`.

Each tracer is independent, so the loop runs on all available Julia threads
(start Julia with `--threads auto` or `julia -t auto` for a proportional speedup;
otherwise this runs serially and can be slow for many tracers).
"""
function compute_zircon_ages(tracers; nx::Int=100,
                              elements::ZirconGrowth.ElementData=ZirconGrowth.default_element_data(),
                              return_results::Bool=false,
                              T_zr_min::Float64=650.0,
                              t_ref_Myr::Union{Nothing,Float64}=nothing)
    n                = length(tracers)
    age_years        = Vector{Union{Nothing,Float64}}(nothing, n)
    zircon_radius_um = Vector{Union{Nothing,Float64}}(nothing, n)
    _results         = return_results ? Vector{Union{Nothing,ZirconGrowth.SimulationResult}}(nothing, n) : nothing

    t_ref = isnothing(t_ref_Myr) ?
        maximum((tr.time_vec[end] for tr in tracers if length(tr.time_vec) >= 2); init=0.0) :
        t_ref_Myr

    Threads.@threads for i in eachindex(tracers)
        tracer = tracers[i]
        length(tracer.time_vec) < 2 && continue
        maximum(tracer.T_vec) < T_zr_min && continue   # never hot enough to grow zircon

        # birth-relative time: ZirconGrowth resamples onto a uniform grid starting at the
        # first control point of the path; passing absolute model time would flat-pad the
        # temperature back to t=0 and squeeze late-born tracers' real history
        time_Myr = Float64.(tracer.time_vec) .- Float64(tracer.time_vec[1])
        T_C      = Float64.(tracer.T_vec)

        params = ZirconGrowth.GrowthParams(time_Myr, T_C; nx=nx)
        res = ZirconGrowth.simulate_from_cooling_path(time_Myr, T_C; params=params, elements=elements)

        # no growth beyond the seed radius -> no zircon, no age
        res.zircon_radius_um[end] - res.zircon_radius_um[1] < 1e-3 && continue

        # common reference clock: add the offset from this tracer's last recorded time
        # (eruption time for erupted cargo) to the shared reference
        age_years[i]        = volume_averaged_age(res) + (t_ref - tracer.time_vec[end])*1e6
        zircon_radius_um[i] = res.zircon_radius_um[end]
        return_results && (_results[i] = res)
    end

    age_years        = Float64[v for v in age_years        if !isnothing(v)]
    zircon_radius_um = Float64[v for v in zircon_radius_um if !isnothing(v)]

    if return_results
        results = ZirconGrowth.SimulationResult[r for r in _results if !isnothing(r)]
        return (; age_years, zircon_radius_um, results)
    end

    return (; age_years, zircon_radius_um)
end
