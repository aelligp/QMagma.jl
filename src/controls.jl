# Translation of control-panel selections into model inputs.

function gui_flux_history(
        mode; base_m_per_yr, peak_m_per_yr,
        start_kyr, end_kyr, table_path = ""
    )
    if mode == "CSV table"
        isempty(strip(table_path)) && throw(ArgumentError("Flux CSV path must not be empty"))
        path = expanduser(strip(table_path))
        if !isabspath(path) && !isfile(path)
            bundled = joinpath(pkgdir(@__MODULE__), path)
            isfile(bundled) && (path = bundled)
        end
        return load_flux_history(path)
    end
    symbol = mode == "Constant" ? :constant :
        mode == "Linear ramp" ? :ramp :
        mode == "Pulse" ? :pulse :
        throw(ArgumentError("unknown flux mode: $mode"))
    return FluxHistory(
        symbol;
        base = symbol == :pulse ? 0.0 : base_m_per_yr / SecYear,
        peak = peak_m_per_yr / SecYear,
        t_start = start_kyr * 1000SecYear,
        t_end = end_kyr * 1000SecYear
    )
end

"""
    gui_composition(name) -> (; melting, solubility, melt_viscosity)

Melting parameterisation, H₂O saturation law, and melt viscosity for a magma composition, as
one choice.

The three are not independent: Liu et al. (2005) is calibrated on silicic melts and the mafic
law on basaltic ones, so pairing either with the wrong melting parameterisation describes a
magma that does not exist. The melt viscosity is bound the same way — it drives the
dike-propagation criterion ([`max_ascent_length`](@ref)), whose reach goes as `1/η`, so a
silicic melt run on the basaltic law is granted an ascent it could not achieve. Returning
them together makes those pairings unselectable rather than merely discouraged — there is no
control that can set one without the others. `MeltingParam_Assimilation` is crustal anatexis
and takes the silicic law and viscosity.
"""
function gui_composition(name)
    name == "MeltingParam_Basalt" &&
        return (;
        melting = MeltingParam_Smooth3rdOrder(),
        solubility = Mafic_Solubility(),
        melt_viscosity = LinearMeltViscosity(),
    )
    name == "MeltingParam_Rhyolite" &&
        return (;
        melting = MeltingParam_Smooth3rdOrder(
            a = 3043.0, b = -10552.0,
            c = 12204.9, d = -4709.0
        ),
        solubility = Liu2005_Solubility(),
        melt_viscosity = LinearMeltViscosity(A = -8.1590, B = 2.4050e4K, T0 = -430.9606K, η0 = 1Pas),
    )
    name == "MeltingParam_Assimilation" &&
        return (;
        melting = MeltingParam_Assimilation(),
        solubility = Liu2005_Solubility(),
        melt_viscosity = LinearMeltViscosity(A = -8.1590, B = 2.4050e4K, T0 = -430.9606K, η0 = 1Pas),
    )
    throw(ArgumentError("unknown magma composition: $name"))
end

"""
    eruption_control_state(trigger)

Return which eruption controls apply to a selected eruption trigger. The 3-phase chamber
uses all of them, so they activate together.
"""
function eruption_control_state(trigger)
    trigger in ("None", "D&H 3-phase") ||
        throw(ArgumentError("unknown eruption trigger: $trigger"))
    active = trigger != "None"
    return (;
        radius = active, pressure = active, shear_modulus = active,
        water = active, collapse = active,
    )
end

"""
    eruption_fires(; h_erupt, near_boundary, Δz) -> Bool

Whether a drained melt thickness becomes a realized withdrawal from the column. Every
physical criterion — roof failure, gas lock-up, dike propagation — has already been applied
at the moment of drainage inside [`step_overpressure!`](@ref), which is the only place they
can be tested against the overpressure that actually drove the dike. What remains here are
the constraints the *grid* imposes: the chamber must not touch a domain boundary, where the
closure has no host rock to draw on, and the withdrawal must exceed two cells to be
resolvable.

Re-testing dike propagation at this point would test the wrong state. The drain resets the
chamber to `ΔP_relax`, so a criterion evaluated afterwards sees an unpressurized chamber and
rejects the very eruption that just occurred; melt that physically left the chamber would
never leave the column.
"""
function eruption_fires(; h_erupt, near_boundary, Δz)
    return h_erupt > 0 && !near_boundary && h_erupt > 2Δz
end

function collapse_surface_subsidence(method, h_erupt)
    method in (:caldera, :hybrid) ||
        throw(ArgumentError("unknown eruption collapse method: $method"))
    return h_erupt
end
