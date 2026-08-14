# Passive tracers carrying T-t histories, and the zircon ages computed from them.

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
    z::Float64
    T::Float64
    phi::Float64
    phase::Int
    time_vec::Vector{Float64}
    T_vec::Vector{Float64}
end

"""
    init_tracers(Silltop, Sillbot; n=20)

Seed `n` passive tracers uniformly across the injection zone `z ∈ [-Sillbot, -Silltop]`
(in km, as elsewhere in the GUI), with `phase=0` (host rock) and an as-yet-empty
temperature-time history.
"""
function init_tracers(Silltop, Sillbot; n = 20)
    zs = range(-Sillbot * 1.0e3, -Silltop * 1.0e3, length = n)
    return [Tracer(z, 0.0, 0.0, 0, Float64[], Float64[]) for z in zs]
end

"""
    add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=5)

Seed `n` new passive tracers within a freshly-inserted sill centered at `Sill_z0` [m]
with thickness `Sill_thick` [m], at temperature `Sill_T` [°C] and `phase=1` (injected
material), and append them to `tracers`.
"""
function add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n = 5)
    zs = range(Sill_z0 - Sill_thick / 2, Sill_z0 + Sill_thick / 2, length = n)
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
function add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n = 2)
    z0 = -(Silltop + Sillbot) / 2 * 1.0e3
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
    w_interp = linear_interpolation(Params.z, Params.w; extrapolation_bc = Line())
    for tracer in tracers
        tracer.z += w_interp(tracer.z) * Params.Δt
    end
    return tracers
end

"""
    advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic)

Displace each tracer's depth `tracer.z` by the same per-point displacement law that
`insert_sill` applies to the temperature/rock fields when a sill of full thickness
`Sill_thick` [m] is emplaced at `Sill_z0` [m]: zero inside the sill, and each wall
moves by at most `Sill_thick/2`.
Use this for the discrete-sill path; `compute_Q_magma!`/`advect_w!` does not displace
the column this way and should use `advect_tracers!` instead.
"""
function advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType = :elastic, r = 5.0e3)
    Sill_thick > 0 || throw(ArgumentError("Sill_thick must be positive"))
    SillType in (:constant, :elastic) ||
        throw(ArgumentError("SillType must be :constant or :elastic"))
    r > 0 || throw(ArgumentError("r must be positive"))
    for tracer in tracers
        z_shift = tracer.z - Sill_z0
        if abs(z_shift) <= Sill_thick / 2
            continue   # inside the sill: no host-rock displacement to apply
        elseif SillType == :constant
            tracer.z += z_shift > 0 ? Sill_thick / 2 : -Sill_thick / 2
        elseif SillType == :elastic
            d = crack_perp_displacement(z_shift, Sill_thick / 2; r = r)
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
function update_tracers_T!(tracers, T, z, time_Myr, phi = nothing)
    T_interp = linear_interpolation(z, T; extrapolation_bc = Line())
    phi_interp = phi === nothing ? nothing : linear_interpolation(z, phi; extrapolation_bc = Line())
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
    t_end = t[end]

    for i in 1:(length(r) - 1)
        dV = r[i + 1]^3 - r[i]^3
        dV <= 0 && continue
        age_mid = t_end - 0.5 * (t[i] + t[i + 1])
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
function compute_zircon_ages(
        tracers; nx::Int = 100,
        elements::ZirconGrowth.ElementData = ZirconGrowth.default_element_data(),
        return_results::Bool = false,
        T_zr_min::Float64 = 650.0,
        t_ref_Myr::Union{Nothing, Float64} = nothing
    )
    n = length(tracers)
    age_years = Vector{Union{Nothing, Float64}}(nothing, n)
    zircon_radius_um = Vector{Union{Nothing, Float64}}(nothing, n)
    _results = return_results ? Vector{Union{Nothing, ZirconGrowth.SimulationResult}}(nothing, n) : nothing

    t_ref = isnothing(t_ref_Myr) ?
        maximum((tr.time_vec[end] for tr in tracers if length(tr.time_vec) >= 2); init = 0.0) :
        t_ref_Myr

    Threads.@threads for i in eachindex(tracers)
        tracer = tracers[i]
        length(tracer.time_vec) < 2 && continue
        maximum(tracer.T_vec) < T_zr_min && continue   # never hot enough to grow zircon

        # birth-relative time: ZirconGrowth resamples onto a uniform grid starting at the
        # first control point of the path; passing absolute model time would flat-pad the
        # temperature back to t=0 and squeeze late-born tracers' real history
        time_Myr = Float64.(tracer.time_vec) .- Float64(tracer.time_vec[1])
        T_C = Float64.(tracer.T_vec)

        params = ZirconGrowth.GrowthParams(time_Myr, T_C; nx = nx)
        res = ZirconGrowth.simulate_from_cooling_path(time_Myr, T_C; params = params, elements = elements)

        # no growth beyond the seed radius -> no zircon, no age
        res.zircon_radius_um[end] - res.zircon_radius_um[1] < 1.0e-3 && continue

        # common reference clock: add the offset from this tracer's last recorded time
        # (eruption time for erupted cargo) to the shared reference
        age_years[i] = volume_averaged_age(res) + (t_ref - tracer.time_vec[end]) * 1.0e6
        zircon_radius_um[i] = res.zircon_radius_um[end]
        return_results && (_results[i] = res)
    end

    age_years = Float64[v for v in age_years        if !isnothing(v)]
    zircon_radius_um = Float64[v for v in zircon_radius_um if !isnothing(v)]

    if return_results
        results = ZirconGrowth.SimulationResult[r for r in _results if !isnothing(r)]
        return (; age_years, zircon_radius_um, results)
    end

    return (; age_years, zircon_radius_um)
end
