module QMagmaGLMakieExt

using GLMakie, MathTeXEngine
GLMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Mirrors the `using` block of QMagma itself: the app code was written against that
# namespace, and an extension does not inherit it.
using GeoParams
using ForwardDiff, SparseArrays, SparseDiffTools, LinearAlgebra, Interpolations, Random
using ZirconGrowth
using JLD2, WriteVTK
using Statistics: std, quantile
using SpecialFunctions: erf

using QMagma
using QMagma: EnthalpyBudget, EruptionEvent, EruptionParams, EruptionState, MassBudget,
    SecYear, Tracer, add_sill_tracers!, add_uniform_content!, add_zone_tracers!,
    advect_markers!, advect_tracers!, advect_tracers_sill!, advect_w!,
    check_density_consistency, check_sill_temperature, collapse_markers!,
    collapse_surface_subsidence, column_enthalpy, compute_Q_magma!, compute_zircon_ages,
    conductive_boundary_energy, conservative_advection, crust_reference_density,
    crustal_relaxation_viscosity, enthalpy_budget_snapshot, erupt_melt!,
    eruption_control_state, export_thermal_structure,
    gui_composition, gui_flux_history, init_model, init_tracers, injected_thickness,
    injection_depth, insert_sill, integrated_content, lateral_effective_area,
    lateral_loss_energy, lateral_profile, lateral_thermal_structure, magma_heat_input,
    mass_budget_snapshot, melt_fraction_from_temperature, melt_thickness,
    nonlinear_solution, nonnegative_debit, sills_due, source_energy,
    step_chamber_eruption!, step_overpressure!, update_enthalpy_budget!,
    update_mass_budget!, update_tracers_T!

include("widgets.jl")
include("layout.jl")
include("callbacks.jl")
include("simulation.jl")
include("app.jl")

end # module QMagmaGLMakieExt
