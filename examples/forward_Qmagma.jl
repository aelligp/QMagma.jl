# Headless Q_magma forward model with zircon ages.
#
#   julia --project=examples -t auto examples/forward_Qmagma.jl
#
# Example use of `QMagma.run_Q_forward` (src/forward.jl): a plotting-free Q_magma run that
# returns the final tracer populations and the zircon age distribution.

using QMagma, GeoParams
using QMagma: SecYear, FluxHistory, EruptionParams, compute_zircon_ages, run_Q_forward

if abspath(PROGRAM_FILE) == @__FILE__
    # ramp 0.05 → 0.15 m/yr between 50 and 150 kyr, as in docs/src/man/scripting.md
    ȧ = FluxHistory(
        :ramp; base = 0.05 / SecYear, peak = 0.15 / SecYear,
        t_start = 50.0e3SecYear, t_end = 150.0e3SecYear
    )
    @time res = run_Q_forward(ȧ)
    println("tracers: $(length(res.tracers)), datable zircon ages: $(length(res.ages.age_years))")
    ka = res.ages.age_years ./ 1.0e3
    println("zircon age [ka before end of run]: min=$(round(minimum(ka), digits = 1)) ",
        "median=$(round(sort(ka)[end ÷ 2], digits = 1)) max=$(round(maximum(ka), digits = 1))")
end
