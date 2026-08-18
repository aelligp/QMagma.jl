# Magma accretion forcing: the flux history both emplacement branches derive from.

"""
    FluxHistory(mode; base, peak=base, t_start=0, t_end=0, times=[], rates=[], depths=[])

Validated magma-accretion history in internal units (seconds and m/s). Supported modes:
`:constant`, `:ramp` (linear from `base` to `peak` across `[t_start, t_end)`), `:pulse`
(`peak` across `[t_start, t_end)`), and `:table` (piecewise-linear `times`,`rates` and
optional positive-downward `depths`, with flat extrapolation). Internal units are seconds,
m/s, and meters.

`:ramp` and `:pulse` are *episodes*: the rate is zero outside their window, so injection
stops at `t_end`. `base` is therefore the rate the ramp starts from and has no meaning for
a pulse, which rejects a nonzero one rather than ignore it. Use `:table` for a history that
combines a steady background with a surge.
"""
struct FluxHistory
    mode::Symbol
    base::Float64
    peak::Float64
    t_start::Float64
    t_end::Float64
    times::Vector{Float64}
    rates::Vector{Float64}
    depths::Vector{Float64}
end

function FluxHistory(
        mode; base = 0.0, peak = base, t_start = 0.0, t_end = 0.0,
        times = Float64[], rates = Float64[], depths = Float64[]
    )
    mode in (:constant, :ramp, :pulse, :table) ||
        throw(ArgumentError("flux mode must be :constant, :ramp, :pulse, or :table"))
    base, peak, t_start, t_end = Float64.((base, peak, t_start, t_end))
    times, rates, depths = Float64.(times), Float64.(rates), Float64.(depths)
    all(isfinite, (base, peak, t_start, t_end)) &&
        all(isfinite, times) && all(isfinite, rates) && all(isfinite, depths) ||
        throw(ArgumentError("flux-history values must be finite"))
    base >= 0 && peak >= 0 && all(>=(0), rates) ||
        throw(ArgumentError("flux rates must be nonnegative"))
    all(>=(0), depths) || throw(ArgumentError("injection depths must be nonnegative"))
    isempty(depths) || mode == :table ||
        throw(ArgumentError("injection depths are supported only for table histories"))
    if mode in (:ramp, :pulse)
        0 <= t_start < t_end ||
            throw(ArgumentError("ramp/pulse times must satisfy 0 ≤ t_start < t_end"))
        mode == :ramp || base == 0 ||
            throw(ArgumentError("a pulse has no base flux; use :table for a background rate"))
    elseif mode == :table
        length(times) == length(rates) ||
            throw(DimensionMismatch("table times and rates must have equal length"))
        isempty(depths) || length(times) == length(depths) ||
            throw(DimensionMismatch("table times and depths must have equal length"))
        length(times) >= 2 || throw(ArgumentError("flux table must contain at least two rows"))
        all(>(0), diff(times)) || throw(ArgumentError("flux-table times must be strictly increasing"))
    end
    return FluxHistory(mode, base, peak, t_start, t_end, times, rates, depths)
end

function _table_value(times, values, time)
    time <= times[1] && return values[1]
    time >= times[end] && return values[end]
    i = searchsortedlast(times, time)
    f = (time - times[i]) / (times[i + 1] - times[i])
    return values[i] + f * (values[i + 1] - values[i])
end

function (history::FluxHistory)(time)
    isfinite(time) || throw(ArgumentError("time must be finite"))
    if history.mode == :constant
        return history.base
    elseif history.mode == :ramp
        history.t_start <= time < history.t_end || return 0.0
        f = (time - history.t_start) / (history.t_end - history.t_start)
        return history.base + f * (history.peak - history.base)
    elseif history.mode == :pulse
        return history.t_start <= time < history.t_end ? history.peak : 0.0
    end
    return _table_value(history.times, history.rates, time)
end

"""
    injection_depth(history::FluxHistory, time) -> Union{Nothing, Float64}

Positive-downward injection depth [m] at `time` [s]. Table depths are linearly
interpolated with flat extrapolation; histories without depths return `nothing`.
"""
function injection_depth(history::FluxHistory, time)
    isfinite(time) || throw(ArgumentError("time must be finite"))
    isempty(history.depths) && return nothing
    return _table_value(history.times, history.depths, time)
end

"""
    load_flux_history(path; time_scale=1000SecYear, rate_scale=1/SecYear, depth_scale=1000)

Read a text/CSV table (`time_kyr, flux_m_per_yr[, depth_km]` by default) and return a
piecewise-linear [`FluxHistory`](@ref). Depth is positive downward. One header row plus blank
and `#` comment lines are accepted; malformed data fail with the source line number.
"""
function load_flux_history(
        path; time_scale = 1000SecYear, rate_scale = 1 / SecYear, depth_scale = 1.0e3
    )
    isfile(path) || throw(ArgumentError("flux table does not exist: $path"))
    all(isfinite, (time_scale, rate_scale, depth_scale)) &&
        time_scale > 0 && rate_scale > 0 && depth_scale > 0 ||
        throw(ArgumentError("table unit scales must be finite and positive"))
    times, rates, depths = Float64[], Float64[], Float64[]
    header_skipped = false
    column_count = 0
    for (line_number, raw) in enumerate(eachline(path))
        line = strip(raw)
        (isempty(line) || startswith(line, '#')) && continue
        fields = split(replace(line, ',' => ' '))
        if length(fields) ∉ (2, 3)
            throw(ArgumentError("flux table line $line_number must contain two or three columns"))
        end
        values = tryparse.(Float64, fields)
        if any(isnothing, values)
            if isempty(times) && !header_skipped
                header_skipped = true
                column_count = length(fields)
                continue
            end
            throw(ArgumentError("flux table line $line_number contains nonnumeric data"))
        end
        column_count == 0 && (column_count = length(fields))
        length(fields) == column_count || throw(
            ArgumentError("flux table line $line_number has a different number of columns")
        )
        time, rate = values[1], values[2]
        if column_count == 3
            depth = values[3]
            isfinite(depth) && depth >= 0 || throw(
                ArgumentError("flux table line $line_number depth must be finite and nonnegative")
            )
            push!(depths, depth * depth_scale)
        end
        push!(times, time * time_scale)
        push!(rates, rate * rate_scale)
    end
    return FluxHistory(:table; times, rates, depths)
end

function _integrate_linear_flux(history::FluxHistory, time, Δt, breakpoints)
    stop = time + Δt
    interior = [t for t in breakpoints if time < t < stop]
    points = vcat(time, interior, stop)
    return sum(
        (history(points[i]) + history(points[i + 1])) *
            (points[i + 1] - points[i]) / 2 for i in 1:(length(points) - 1)
    )
end

function injected_thickness(history::FluxHistory, time, Δt)
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    isfinite(time) || throw(ArgumentError("time must be finite"))
    if history.mode == :constant
        return history.base * Δt
    elseif history.mode in (:ramp, :pulse)
        # Integrate over the episode window only. Both modes step to zero at its edges, so
        # the breakpoint trapezoid used for tables would smear the discontinuity; inside the
        # window the integrand is linear and the midpoint rule is exact.
        a = max(time, history.t_start)
        b = min(time + Δt, history.t_end)
        b > a || return 0.0
        history.mode == :pulse && return history.peak * (b - a)
        slope = (history.peak - history.base) / (history.t_end - history.t_start)
        return (history.base + slope * ((a + b) / 2 - history.t_start)) * (b - a)
    end
    return _integrate_linear_flux(history, time, Δt, history.times)
end

"""
    injected_thickness(ȧ, time, Δt) -> Δh

Magma thickness [m] accreted during the step `(time, time + Δt]` for the accretion-rate
history `ȧ` [m/s], integrated by the midpoint rule (exact for a constant rate). `ȧ` is
either a number or a callable `ȧ(t)`.

This is the single forcing quantity both emplacement branches derive from: the discrete
branch counts sills per accumulated thickness ([`sills_due`](@ref)) and the smeared branch
uses the step-mean rate `Δh/Δt` ([`compute_Q_magma!`](@ref)), so the two cannot drift apart
under a variable rate.
"""
function injected_thickness(ȧ, time, Δt)
    Δt > 0 || throw(ArgumentError("Δt must be positive"))
    isfinite(time) || throw(ArgumentError("time must be finite"))
    rate = ȧ isa Number ? ȧ : ȧ(time + Δt / 2)
    isfinite(rate) && rate >= 0 ||
        throw(ArgumentError("accretion rate must be finite and nonnegative, got $rate"))
    return rate * Δt
end

"""
    sills_due(A, ΔA, d) -> Int

Number of sills of aperture `d` [m] completed while the cumulative injected thickness grows
from `A` to `A + ΔA`, i.e. `floor((A+ΔA)/d) - floor(A/d)`. More than one sill is returned
when a step delivers more than one aperture, and none while the accretion rate is zero.

Emplacement is keyed to injected thickness rather than to elapsed time so that a varying
accretion rate changes the event *frequency* at fixed aperture. For a constant rate
`ȧ = d/interval` this reproduces the event times of an interval-keyed schedule.
"""
function sills_due(A, ΔA, d)
    all(isfinite, (A, ΔA, d)) || throw(ArgumentError("A, ΔA, and d must be finite"))
    A >= 0 || throw(ArgumentError("A must be nonnegative"))
    ΔA >= 0 || throw(ArgumentError("ΔA must be nonnegative"))
    d > 0 || throw(ArgumentError("sill aperture must be positive"))

    sill_index(a) = floor(Int, a / d + 8eps(a / d))
    return sill_index(A + ΔA) - sill_index(A)
end
