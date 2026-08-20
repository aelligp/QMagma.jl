using QMagma
using GeoParams
using JLD2
using LinearAlgebra, SparseArrays, SparseDiffTools
using Random
using Test

# Loading any Makie backend activates QMagmaMakieExt, so the GUI testset exercises it.
# CairoMakie renders in software, so the layout and wiring are testable without the OpenGL
# context (and hence display) that the GLMakie backend of the shipped app requires.
using CairoMakie

const SecYear = 3600 * 24 * 365.25

function thermal_jacobian_workspace(nz)
    J = Tridiagonal(ones(nz - 1), ones(nz), ones(nz - 1))
    J[1, 2] = 0
    J[2, 1] = 0
    J[nz - 1, nz] = 0
    J[nz, nz - 1] = 0
    Jac = sparse(Float64.(abs.(J) .> 0))
    return Jac, matrix_colors(Jac)
end

@testset "QMagma.jl" begin

    @testset "thermal structure export" begin
        mktempdir() do directory
            z = [-2.0, -1.0, 0.0]
            T = [900.0, 500.0, 0.0]

            files_1d = QMagma.export_thermal_structure(
                joinpath(directory, "one"), z;
                fields = (temperature = T,)
            )
            @test all(isfile, files_1d)
            data_1d = JLD2.load(joinpath(directory, "one.jld2"))
            @test data_1d["dimensionality"] == 1
            @test data_1d["temperature"] == T

            background = [200.0, 100.0, 0.0]
            x = [-2.0, 0.0, 2.0]
            T2 = QMagma.lateral_thermal_structure(T, background, x; R = 3.0)
            @test T2[2, :] == T
            @test T2[1, :] == T2[3, :]
            tapered_anomaly = abs.(T2[1, :] .- background)
            original_anomaly = abs.(T .- background)
            @test all(tapered_anomaly .<= original_anomaly)
            @test any(tapered_anomaly .< original_anomaly)
            # outside the body the field is background, with no tail to truncate
            T2_edge = QMagma.lateral_thermal_structure(T, background, [-4.0, 0.0, 4.0]; R = 3.0)
            @test T2_edge[1, :] == background
            @test T2_edge[3, :] == background
            Params, = QMagma.init_model(nz = 3, L = 2.0, Ttop = 0.0, Tbot = 200.0, Δt = 1.0)
            ϕ2 = QMagma.melt_fraction_from_temperature(T2, Params.MatParam)
            @test size(ϕ2) == size(T2)
            @test all((0 .<= ϕ2) .& (ϕ2 .<= 1))
            files_2d = QMagma.export_thermal_structure(
                joinpath(directory, "two"), z;
                x, fields = (temperature = T2, melt_fraction = ϕ2)
            )
            @test all(isfile, files_2d)
            @test size(JLD2.load(joinpath(directory, "two.jld2"))["temperature"]) == (3, 3)

            y = [-2.0, 0.0, 2.0]
            T3 = QMagma.lateral_thermal_structure(T, background, x; y, R = 3.0)
            @test T3[2, 2, :] == T
            files_3d = QMagma.export_thermal_structure(
                joinpath(directory, "three"), z;
                x, y, fields = (temperature = T3,)
            )
            @test all(isfile, files_3d)
            @test size(JLD2.load(joinpath(directory, "three.jld2"))["temperature"]) == (3, 3, 3)

            @test_throws "field temperature has size (2, 2)" QMagma.export_thermal_structure(
                joinpath(directory, "bad"), z; x, fields = (temperature = zeros(2, 2),)
            )
        end
    end

    @testset "av" begin
        @test QMagma.av([1.0, 2.0, 4.0]) ≈ [1.5, 3.0]
        @test QMagma.av([0.0, 0.0]) ≈ [0.0]
    end

    @testset "node-centered control-volume integration" begin
        z = collect(-1000.0:100.0:0.0)
        @test QMagma.integrated_content(ones(length(z)), z) == 1000.0
        @test QMagma.melt_thickness(ones(length(z)), z, -750.0, -250.0) == 500.0
        field = zeros(length(z))
        QMagma.add_uniform_content!(field, z, -750.0, -250.0, 123.0)
        @test QMagma.integrated_content(field, z) ≈ 123.0
        @test_throws "folds a control volume" QMagma.conservative_advection(
            field, reverse(z), z
        )

        # an outflow-only remap may lose content, and gains within roundoff of the summed
        # control volumes are clamped, but a real gain is a conservation failure
        @test QMagma.nonnegative_debit(10.0, 4.0, "melt") == 6.0
        @test QMagma.nonnegative_debit(10.0, 10.0 + 4eps(10.0), "melt") == 0.0
        @test_throws "melt increased by" QMagma.nonnegative_debit(10.0, 11.0, "melt")
    end

    @testset "init_model" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 11, L = 10.0e3, Geotherm = 10.0,
            Ttop = 0.0, Tbot = 100.0, Δt = 100SecYear
        )

        @test N == (11,)
        @test length(z) == 11
        @test length(T) == 11
        @test Δ[1] ≈ 1.0e3                       # dz = L/(nz-1)
        @test extrema(z) == (-10.0e3, 0.0)
        @test BC.Ttop == 0.0
        @test BC.Tbot == 100.0

        # linear geotherm: T = -Geotherm/1e3 * z + Ttop
        @test T[1] ≈ 100.0                     # bottom (z = -L)
        @test T[end] ≈ 0.0                     # top (z = 0)
        @test issorted(T, rev = true)            # temperature decreases from bottom to top
        @test Params.Told == T                  # previous state starts from the same geotherm

        # a custom MatParam tuple should be used as-is
        Params2, = QMagma.init_model(nz = 5, L = 1.0e3, Δt = 1SecYear, MatParam = Params.MatParam)
        @test Params2.MatParam === Params.MatParam
        @test_throws "MatParam cannot be combined" QMagma.init_model(
            nz = 5, L = 1.0e3, Δt = 1SecYear, MatParam = Params.MatParam, ρ = 2500.0
        )

        # the constitutive laws are assembled here, so the caller sets them by keyword
        # instead of building a competing MatParam tuple
        Params3, = QMagma.init_model(
            nz = 5, L = 1.0e3, Δt = 1SecYear, ρ = 2500.0, Q_L = 3.0e5,
            Conductivity = ConstantConductivity(k = 3.0),
            HeatCapacity = ConstantHeatCapacity(),
            Melting = MeltingParam_Caricchi()
        )
        args = (T = [1000.0 + 273.15],)
        ρ3 = zeros(1); k3 = zeros(1); ϕ3 = zeros(1)
        compute_density!(ρ3, Params3.MatParam, [0], args)
        compute_conductivity!(k3, Params3.MatParam, [0], args)
        compute_meltfraction!(ϕ3, Params3.MatParam, [0], args)
        @test ρ3[1] ≈ 2500.0
        @test k3[1] ≈ 3.0
        @test GeoParams.NumValue(Params3.MatParam[1].LatentHeat[1].Q_L) ≈ 3.0e5
        ϕ_default = zeros(1)
        compute_meltfraction!(ϕ_default, Params.MatParam, [0], args)
        @test ϕ3[1] != ϕ_default[1]     # MeltingParam_Caricchi, not the default parameterization
    end

    @testset "check_density_consistency" begin
        ep = QMagma.EruptionParams()
        @test ep.m_w == 0.0
        @test ep.ρ_gas isa RedlichKwong_Density
        ρ_ref = QMagma.crust_reference_density(ep)
        Params, = QMagma.init_model(nz = 5, L = 1.0e3, Δt = 1SecYear, ρ = ρ_ref)
        @test QMagma.check_density_consistency(Params.MatParam, ep) ≈ ρ_ref
        # thermal expansion alone at the 800 °C reference, with P_ref = 0
        @test ρ_ref ≈ 2700.0 * (1 - 3.0e-5 * 800.0)

        # Host-rock thermal and lithostatic densities describe the same crustal column.
        @test_throws "host-rock density mismatch" QMagma.check_density_consistency(
            Params.MatParam, QMagma.EruptionParams(crust = ConstantDensity(ρ = 2600kg / m^3))
        )
        # Chamber crystals and melt are distinct materials and need not equal host rock.
        @test QMagma.check_density_consistency(
            Params.MatParam,
            QMagma.EruptionParams(ρ_x = ConstantDensity(ρ = 2900kg / m^3))
        ) ≈ ρ_ref
        @test QMagma.check_density_consistency(
            Params.MatParam,
            QMagma.EruptionParams(ρ_melt = ConstantDensity(ρ = 1000kg / m^3))
        ) ≈ ρ_ref
        @test_throws "must give positive densities" QMagma.validate_eruption_params(
            QMagma.EruptionParams(ρ_melt = ConstantDensity(ρ = -1kg / m^3))
        )
        @test_throws "gas density law returned a nonpositive density" QMagma.validate_eruption_params(
            QMagma.EruptionParams(
                m_w = 0.05, ρ_gas = ConstantDensity(ρ = -1kg / m^3)
            )
        )
        # The Huber et al. (2010) H2O fit has a spurious density minimum: below it the gas
        # gets denser as it decompresses, which flips the sign of the mixture
        # compressibility and leaves the chamber mass solve without a descent direction.
        # The floor is where the fit's own pressure derivative vanishes.
        let ep = QMagma.EruptionParams(m_w = 0.04)
            floor_900 = QMagma.gas_pressure_floor(ep.ρ_gas, 1173.15)
            @test 25.0e6 < floor_900 < 25.7e6
            # rises with temperature, and stays inside the 23-27 MPa band over magmatic T
            @test QMagma.gas_pressure_floor(ep.ρ_gas, 973.15) < floor_900
            @test 23.0e6 < QMagma.gas_pressure_floor(ep.ρ_gas, 973.15) < 27.5e6
            # the derivative really does vanish there, and is negative just below
            dρ = QMagma.ForwardDiff.derivative(
                P -> compute_density(ep.ρ_gas, (; P, T = 1173.15)), floor_900
            )
            @test isapprox(dρ, 0, atol = 1.0e-7)
            @test QMagma.ForwardDiff.derivative(
                P -> compute_density(ep.ρ_gas, (; P, T = 1173.15)), 0.5floor_900
            ) < 0

            @test QMagma.gas_density(ep, 1.0e8, 1173.15) > 0
            @test_throws "below the" QMagma.gas_density(ep, 0.5floor_900, 1173.15)
            # above ~1050 C the fit returns a negative density just above the pressure
            # floor, so positivity is a separate check rather than a consequence of it
            @test_throws "nonpositive density" QMagma.gas_density(ep, 2.8e7, 1373.15)
            # a law without a low-pressure pathology gets no floor imposed on it
            @test QMagma.gas_pressure_floor(ConstantDensity(ρ = 500kg / m^3), 1173.15) == 0
        end

        @test_throws "rtol must be nonnegative" QMagma.check_density_consistency(
            Params.MatParam, ep; rtol = -1.0
        )
    end

    @testset "nonlinear_solution: steady linear geotherm is (near) a fixed point" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 21, L = 10.0e3, Geotherm = 10.0,
            Ttop = 0.0, Tbot = 100.0, Δt = 100SecYear
        )
        Params.Told .= T

        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T)

        Tsol, converged, its = QMagma.nonlinear_solution(
            F, copy(T), Jac, colors;
            Δ = Δ, N = N, BC = BC, Params = Params,
            MatParam = Params.MatParam, verbose = false
        )

        @test its == 1                          # linear problem -> single Newton step
        @test Tsol[1] ≈ BC.Tbot
        @test Tsol[end] ≈ BC.Ttop
        @test maximum(abs.(Tsol .- T)) < 1.0e-2    # steady-state geotherm barely changes

        # verbose runs trace every Newton iteration with its error and step length
        trace = mktemp() do path, io
            redirect_stdout(io) do
                QMagma.nonlinear_solution(
                    F, copy(T), Jac, colors;
                    Δ = Δ, N = N, BC = BC, Params = Params,
                    MatParam = Params.MatParam, verbose = true
                )
            end
            flush(io)
            return read(path, String)
        end
        @test occursin("Nonlinear iteration 1", trace)
    end

    @testset "source term Q heats the interior" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 21, L = 10.0e3, Geotherm = 0.0,
            Ttop = 0.0, Tbot = 0.0, Δt = 1.0e3SecYear
        )
        Params.Told .= T

        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T)

        @test all(Params.Q .== 0)               # Q defaults to zero -> no change in behaviour

        Tsol0, = QMagma.nonlinear_solution(
            F, copy(T), Jac, colors;
            Δ = Δ, N = N, BC = BC, Params = Params,
            MatParam = Params.MatParam, verbose = false
        )
        @test all(Tsol0 .≈ 0.0)                  # no source, zero BCs -> stays at zero

        Params.Q .= 1.0e-5                         # uniform volumetric heat source [W/m^3]
        Tsol1, = QMagma.nonlinear_solution(
            F, copy(T), Jac, colors;
            Δ = Δ, N = N, BC = BC, Params = Params,
            MatParam = Params.MatParam, verbose = false
        )

        @test all(Tsol1[2:(end - 1)] .> Tsol0[2:(end - 1)])  # interior heats up due to Q
        @test Tsol1[1] ≈ BC.Tbot                     # boundary conditions still enforced
        @test Tsol1[end] ≈ BC.Ttop
    end

    @testset "lateral (3-D) heat loss" begin
        # a hot interior anomaly on an isothermal background: the same source, cooled by a
        # finite lateral extent, must end colder than the 1-D column, and the loss must be
        # booked in the energy budget with the sign that cools it
        function run(R_lat)
            Params, BC, N, Δ, T, z = QMagma.init_model(
                nz = 21, L = 10.0e3, Geotherm = 0.0,
                Ttop = 0.0, Tbot = 0.0, Δt = 1.0e3SecYear,
                R_lat = R_lat
            )
            Params.Told .= T
            Params.Q .= 1.0e-5
            Jac, colors = thermal_jacobian_workspace(N[1])
            Tsol, = QMagma.nonlinear_solution(
                zero(T), copy(T), Jac, colors;
                Δ = Δ, N = N, BC = BC, Params = Params,
                MatParam = Params.MatParam, verbose = false
            )
            return Tsol, QMagma.lateral_loss_energy(Tsol, Params, z, Params.Δt), BC
        end

        T_1d, E_1d = run(Inf)
        T_3d, E_3d, BC = run(2.0e3)
        T_wide, _ = run(20.0e3)

        @test E_1d == 0.0                                     # R_lat = Inf -> purely 1-D
        @test all(T_3d[2:(end - 1)] .< T_1d[2:(end - 1)])              # sideways conduction cools
        @test all(T_wide[2:(end - 1)] .< T_1d[2:(end - 1)])
        @test all(T_wide[2:(end - 1)] .> T_3d[2:(end - 1)])            # weaker for a wider body
        @test E_3d < 0                                         # booked as heat leaving
        @test T_3d[1] ≈ BC.Tbot && T_3d[end] ≈ BC.Ttop         # BCs still enforced

        @test_throws "R_lat must be positive" QMagma.init_model(R_lat = 0.0)
    end

    @testset "lateral profile is the emplaced crack opening" begin
        R = 5.0e3
        w(ρ) = QMagma.lateral_profile(ρ, R)

        # Sneddon's penny-shaped crack under uniform pressure
        @test w(0.0) ≈ 1.0
        @test w(0.6R) ≈ sqrt(1 - 0.6^2)
        @test w(R) == 0.0                       # compact support: the body has an edge
        @test w(1.5R) == 0.0
        @test w(0.3R) > w(0.7R)                 # thickest on the axis

        # an unbounded body is laterally uniform, the same 1-D limit R_lat = Inf takes
        @test QMagma.lateral_profile(1.0e9, Inf) ≈ 1.0
    end

    @testset "the crack that opens is the crack the geometry assumes" begin
        # the volume convention of the penny-shaped sills in InjectSills.jl and
        # MagmaThermoKinematics.jl: Q = 2*pi*H*W^2/3 for radius W and axial aperture H
        for (W, H) in ((5.0e3, 400.0), (1.2e3, 50.0))
            @test QMagma.lateral_effective_area(W) * H ≈ 2π * H * W^2 / 3
        end

        # the opening kernel is the Sun (1969) crack: one at the face, and its far field
        # carries the shape's own decay rather than a disc's solid angle
        R, ν = 5.0e3, 0.3
        @test QMagma.crack_perp_shape(0.0, R, ν) ≈ 1.0
        @test QMagma.crack_perp_shape(1.0e9, R, ν) ≈ 0.0 atol = 1.0e-9
        @test QMagma.crack_perp_shape(0.5R, R, ν) > QMagma.crack_perp_shape(R, R, ν)
        @test QMagma.crack_perp_shape(1.0e6, Inf, ν) ≈ 1.0        # unbounded: uniform

        # crack_perp_integral must be the antiderivative of crack_perp_shape
        for t in (0.3R, R, 4R)
            n = 20_000
            quad = sum(QMagma.crack_perp_shape((i - 0.5) * t / n, R, ν) for i in 1:n) * t / n
            @test QMagma.crack_perp_integral(t, R, ν) ≈ quad rtol = 1.0e-6
            @test QMagma.crack_perp_integral(-t, R, ν) == QMagma.crack_perp_integral(t, R, ν)
        end
    end

    @testset "effective area is the lateral profile integrated over the plane" begin
        # the area that turns an axial thickness into a volume must be the one the exported
        # body actually occupies, not the disc πR² it is inscribed in
        for R in (1.0, 5.0e3, 2.7e4)
            n = 400_000
            dr = R / n
            numeric = sum(QMagma.lateral_profile((i - 0.5) * dr, R) * 2π * (i - 0.5) * dr * dr for i in 1:n)
            @test QMagma.lateral_effective_area(R) ≈ numeric rtol = 1.0e-6
            @test QMagma.lateral_effective_area(R) < π * R^2
        end
        @test QMagma.lateral_effective_area(3.0) ≈ 2π * 9 / 3
    end

    @testset "exported body carries the column's anomaly enthalpy" begin
        # expanding the column must neither create nor destroy the anomaly: the 3-D anomaly
        # integrated over the plane is the axial anomaly times the effective area
        R = 4.0e3
        z = collect(-2.0e3:100.0:0.0)
        background = fill(300.0, length(z))
        T = background .+ 400.0 .* exp.(-((z .+ 1.0e3) ./ 400.0) .^ 2)

        n = 2001
        x = collect(range(-R, R; length = n))
        T2 = QMagma.lateral_thermal_structure(T, background, x; R)
        anomaly_2d = T2 .- reshape(background, 1, :)
        # revolve the 2-D slice: ∫anomaly(r) 2πr dr over r ≥ 0, against the axial anomaly
        half = (n + 1) ÷ 2
        r = x[half:end]
        planar = [sum(anomaly_2d[half:end, k] .* 2π .* r .* step(range(0, R; length = half))) for k in eachindex(z)]
        @test planar ≈ (T .- background) .* QMagma.lateral_effective_area(R) rtol = 1.0e-3
    end

    @testset "lateral loss is the axis curvature of the lateral profile" begin
        # The residual's lateral sink and the shape the model assumes laterally are one
        # assumption, not two: Hlat is -k times the profile's radial Laplacian on the axis.
        # Changing the profile without changing the coefficient must fail here.
        for (R, k) in ((5.0e3, 3.0), (1.2e3, 2.25), (4.0e4, 1.7))
            w(ρ) = QMagma.lateral_profile(ρ, R)
            h = R / 1.0e3
            curvature = 2 * (w(h) - 2w(0.0) + w(h)) / h^2 / w(0.0)   # 2 lateral dimensions
            @test QMagma.lateral_loss_coefficient(k, R) ≈ -k * curvature rtol = 1.0e-5
        end
        @test QMagma.lateral_loss_coefficient(3.0, Inf) == 0.0
    end

    @testset "sill opening uses the model's sill radius" begin
        # Advecting a linear profile makes the applied displacement readable off the
        # temperature field as (T - T_adv)/gradient. At |z| = r the Sun crack shape is
        # 1 - π/4 + (π/4 - 1/2)/(2(1-ν)), so that crossing locates the radius the opening
        # actually used. The recovered profile sits one departure point further out than
        # the analytic one, which is what sets the tolerance.
        z = collect(-30.0e3:100.0:0.0)
        grad, z0, thick = 0.02, -15.0e3, 200.0
        T0 = grad .* (z .+ 30.0e3)
        rocks = zero(z)
        ν = 0.3
        target = (thick / 2) * (1 - π / 4 + (π / 4 - 1 / 2) / (2 * (1 - ν)))

        function opening_radius(T1)
            applied = (T0 .- T1) ./ grad
            up = findall(zz -> zz > z0 + 500.0, z)
            j = findfirst(k -> applied[up[k]] < target, eachindex(up))
            z1, z2 = z[up[j - 1]], z[up[j]]
            a1, a2 = applied[up[j - 1]], applied[up[j]]
            return z1 + (a1 - target) * (z2 - z1) / (a1 - a2) - z0
        end

        for R in (3.0e3, 5.0e3, 9.0e3)
            T1, = QMagma.insert_sill(
                T0, rocks, z; Sill_thick = thick, Sill_z0 = z0, Sill_T = 1200.0, r = R
            )
            @test opening_radius(T1) ≈ R rtol = 0.02
        end

        # a crack of unbounded radius opens uniformly - the constant-displacement limit
        T_inf, = QMagma.insert_sill(
            T0, rocks, z; Sill_thick = thick, Sill_z0 = z0, Sill_T = 1200.0, r = Inf
        )
        T_const, = QMagma.insert_sill(
            T0, rocks, z; Sill_thick = thick, Sill_z0 = z0, Sill_T = 1200.0,
            SillType = :constant
        )
        @test T_inf ≈ T_const

        @test_throws "r must be positive" QMagma.insert_sill(T0, rocks, z; r = 0.0)
    end

    @testset "shared accretion forcing" begin
        # midpoint integration of the rate history: exact for a constant rate, and a
        # callable is sampled at the step midpoint
        @test QMagma.injected_thickness(2.0, 10.0, 3.0) == 6.0
        @test QMagma.injected_thickness(t -> t, 0.0, 4.0) == 8.0        # ∫₀⁴ t dt = 8
        @test QMagma.injected_thickness(t -> 0.0, 5.0, 1.0) == 0.0
        @test_throws "Δt must be positive" QMagma.injected_thickness(1.0, 0.0, 0.0)
        @test_throws "time must be finite" QMagma.injected_thickness(1.0, NaN, 1.0)
        @test_throws "accretion rate must be finite and nonnegative" QMagma.injected_thickness(
            -1.0, 0.0, 1.0
        )

        constant = QMagma.FluxHistory(:constant; base = 2.0)
        ramp_history = QMagma.FluxHistory(
            :ramp; base = 1.0, peak = 3.0,
            t_start = 2.0, t_end = 4.0
        )
        pulse = QMagma.FluxHistory(:pulse; peak = 3.0, t_start = 2.0, t_end = 4.0)
        table = QMagma.FluxHistory(:table; times = [0.0, 2.0, 4.0], rates = [1.0, 3.0, 1.0])
        depth_table = QMagma.FluxHistory(
            :table; times = [0.0, 2.0, 4.0], rates = [1.0, 3.0, 1.0],
            depths = [10.0, 20.0, 30.0]
        )
        @test QMagma.injected_thickness(constant, 1.0, 4.0) == 8.0
        @test QMagma.injected_thickness(table, 0.0, 4.0) == 8.0
        @test constant(3.0) == 2.0
        @test ramp_history(3.0) == 2.0
        @test pulse(3.0) == 3.0
        @test table(3.0) == 2.0
        @test QMagma.injection_depth(constant, 1.0) === nothing
        @test QMagma.injection_depth(depth_table, -1.0) == 10.0
        @test QMagma.injection_depth(depth_table, 1.0) == 15.0
        @test QMagma.injection_depth(depth_table, 5.0) == 30.0
        @test_throws "times and depths must have equal length" QMagma.FluxHistory(
            :table; times = [0.0, 1.0], rates = [1.0, 1.0], depths = [10.0]
        )
        @test_throws "depths must be nonnegative" QMagma.FluxHistory(
            :table; times = [0.0, 1.0], rates = [1.0, 1.0], depths = [10.0, -1.0]
        )
        @test_throws "strictly increasing" QMagma.FluxHistory(
            :table;
            times = [0.0, 0.0], rates = [1.0, 2.0]
        )
        @test_throws "a pulse has no base flux" QMagma.FluxHistory(
            :pulse;
            base = 1.0, peak = 3.0, t_start = 2.0, t_end = 4.0
        )

        # ramp and pulse are episodes: nothing is injected outside [t_start, t_end), and a
        # step straddling either edge contributes only its overlap with the window
        for episode in (ramp_history, pulse)
            @test episode(1.0) == 0.0
            @test episode(4.0) == 0.0                                  # window is half-open
            @test episode(1.0e6) == 0.0                                  # injection has ended
            @test QMagma.injected_thickness(episode, 4.0, 100.0) == 0.0
            @test QMagma.injected_thickness(episode, 0.0, 1.0) == 0.0
        end
        # base 1 ramping to peak 3 over [2,4): mean rate 2 across the whole window
        @test QMagma.injected_thickness(ramp_history, 1.0, 4.0) == 4.0
        @test QMagma.injected_thickness(ramp_history, 0.0, 3.0) == 1.5   # overlap [2,3)
        @test QMagma.injected_thickness(pulse, 1.0, 4.0) == 6.0
        @test QMagma.injected_thickness(pulse, 3.0, 100.0) == 3.0        # overlap [3,4)

        mktemp() do path, io
            write(io, "time_kyr,flux_m_per_yr\n0,0.1\n10,0.3\n")
            close(io)
            loaded = QMagma.load_flux_history(path; time_scale = 1.0, rate_scale = 1.0)
            @test loaded.mode == :table
            @test loaded(5.0) == 0.2
            @test QMagma.injected_thickness(loaded, 0.0, 10.0) == 2.0
        end

        example_path = joinpath(@__DIR__, "..", "examples", "flux_history.csv")
        example = QMagma.load_flux_history(
            example_path; time_scale = 1.0, rate_scale = 1.0
        )
        @test example.times == [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]
        @test example.rates == [0.05, 0.05, 0.15, 0.30, 0.15, 0.05, 0.05]
        @test example.depths == [20.0, 20.0, 17.5, 15.0, 12.5, 10.0, 8.0] .* 1.0e3
        @test QMagma.injection_depth(example, 62.5) == 16.25e3

        # The shipped Unzen scenario is the reference for the two branches' end-of-run
        # forcing mismatch: Q_magma represents the whole delivered thickness, while
        # fixed-aperture sills emplace it in whole units and leave the remainder pending.
        unzen = QMagma.load_flux_history(
            joinpath(@__DIR__, "..", "examples", "unzen_example_flux_history.csv")
        )
        delivered = QMagma.injected_thickness(unzen, 0.0, unzen.times[end])
        @test delivered ≈ 425.5
        @test QMagma.sills_due(0.0, delivered, 100.0) == 4
        @test mod(delivered, 100.0) ≈ 25.5

        # malformed tables name the offending source line. A single unparsable leading row
        # is the header and is skipped; anything after it is data and must parse.
        mktemp() do path, io
            write(io, "time_kyr,flux_m_per_yr\n0,0.1\n10,0.3,0.5,1.0\n")
            close(io)
            @test_throws "line 3 must contain two or three columns" QMagma.load_flux_history(path)
        end
        mktemp() do path, io
            write(io, "time_kyr,flux_m_per_yr\n0,0.1\nten,0.3\n")
            close(io)
            @test_throws "line 3 contains nonnumeric data" QMagma.load_flux_history(path)
        end

        # emplacement keyed to injected thickness
        @test [QMagma.sills_due(a, 40.0, 100.0) for a in 0.0:40.0:160.0] == [0, 0, 1, 0, 1]
        @test QMagma.sills_due(0.0, 240.0, 100.0) == 2
        @test QMagma.sills_due(100.0, 0.0, 100.0) == 0                  # a paused flux emplaces nothing
        @test_throws "sill aperture must be positive" QMagma.sills_due(0.0, 1.0, 0.0)
        @test_throws "ΔA must be nonnegative" QMagma.sills_due(0.0, -1.0, 100.0)
        @test_throws "A must be nonnegative" QMagma.sills_due(-1.0, 1.0, 100.0)
        @test_throws "A, ΔA, and d must be finite" QMagma.sills_due(0.0, Inf, 100.0)

        d, interval, Δt = 100.0, 500.0, 200.0
        ȧ = d / interval
        # (1) a constant rate reproduces the event times of the interval-keyed schedule
        A, time = 0.0, 0.0
        thickness_keyed = Int[]
        time_keyed = Int[]
        for _ in 1:2000
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            push!(thickness_keyed, QMagma.sills_due(A, Δh, d))
            push!(time_keyed, floor(Int, (time + Δt) / interval) - floor(Int, time / interval))
            A += Δh
            time += Δt
        end
        @test thickness_keyed == time_keyed
        @test A ≈ ȧ * time

        # (2) a linear ramp emplaces the sills its integrated flux delivers, to within one
        ramp(t) = 2ȧ * t / (2000Δt)
        A, time, n_ramp = 0.0, 0.0, 0
        for _ in 1:2000
            Δh = QMagma.injected_thickness(ramp, time, Δt)
            n_ramp += QMagma.sills_due(A, Δh, d)
            A += Δh
            time += Δt
        end
        @test A ≈ ȧ * time                     # same total as the constant rate it ramps around
        @test abs(n_ramp - A / d) <= 1

        # (3) a flux that switches off emplaces nothing afterwards
        t_off = 1000Δt
        switched(t) = t < t_off ? ȧ : 0.0
        A, time, n_before, n_after = 0.0, 0.0, 0, 0
        for _ in 1:2000
            Δh = QMagma.injected_thickness(switched, time, Δt)
            n = QMagma.sills_due(A, Δh, d)
            time < t_off ? (n_before += n) : (n_after += n)
            A += Δh
            time += Δt
        end
        @test n_before == 400
        @test n_after == 0
    end

    @testset "compute_Q_magma!" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 41, L = 40.0e3, Geotherm = 20.0,
            Ttop = 0.0, Tbot = 800.0, Δt = 200SecYear
        )
        Params.Told .= T

        Tsill = 1200.0
        Silltop = 10.0    # km
        Sillbot = 20.0    # km
        ȧ = 100.0 / 500SecYear   # Sillthick/Sill_interval [m/s]

        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill = Tsill, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)

        ind_zone = findall(z .>= -Sillbot * 1.0e3 .&& z .<= -Silltop * 1.0e3)
        ind_out = setdiff(1:length(z), ind_zone)

        @test all(Params.Q[ind_out] .== 0)          # zero outside the injection zone
        @test all(Params.Q[ind_zone] .> 0)          # positive heat input while Told < Tsill everywhere in zone

        # Ensemble-mean host-rock velocity from uniformly distributed discrete sills.
        ind_below = findall(z .< -Sillbot * 1.0e3)
        ind_above = findall(z .> -Silltop * 1.0e3)
        @test all(Params.w[ind_below] .< 0)         # pushed down below the zone
        @test all(Params.w[ind_above] .> 0)         # pushed up above the zone
        @test Params.w[argmin(abs.(z .+ 15.0e3))] ≈ 0.0 atol = eps(ȧ)
        @test Params.w[argmin(abs.(z .+ 18.0e3))] < 0
        @test Params.w[argmin(abs.(z .+ 12.0e3))] > 0
        @test issorted(abs.(Params.w[ind_below]))            # |w| grows approaching the zone from below (z increasing)
        @test issorted(abs.(Params.w[ind_above]), rev = true)  # |w| decays moving away from the zone above (z increasing)

        centers = range(-Sillbot * 1.0e3, -Silltop * 1.0e3; length = 20_001)
        d = 100.0
        event_rate = ȧ / d
        for i in (5, argmin(abs.(z .+ 18.0e3)), argmin(abs.(z .+ 12.0e3)), length(z) - 5)
            offsets = z[i] .- centers
            numerical_mean = event_rate * sum(sign.(offsets) .* QMagma.crack_perp_displacement(offsets, d / 2)) / length(centers)
            @test Params.w[i] ≈ numerical_mean rtol = 5.0e-4 atol = eps(ȧ)
        end

        # heat input vanishes once the column has fully equilibrated to Tsill (ϕ=1, T=Tsill)
        Params.Told .= Tsill
        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill = Tsill, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)
        @test all(isapprox.(Params.Q[ind_zone], 0.0, atol = 1.0e-6))
    end

    @testset "discrete sills vs Q_magma agree in the many-sill limit" begin
        H = 40.0
        γ = 20.0
        Ttop = 0.0
        Tbot = Ttop + H * γ
        nz = floor(Int64, H * 1.0e3 / 100.0)
        Δt = 200SecYear
        Tsill = 1200.0
        Sillthick = 100.0      # small, frequent sills -> many-sill limit
        Sill_int_yr = 500.0
        Silltop = 10.0
        Sillbot = 20.0
        nt = 4000

        Params, BC, N, Δ, T, z = QMagma.init_model(nz = nz, L = H * 1.0e3, Geotherm = γ, Ttop = Ttop, Tbot = Tbot, Δt = Δt)
        MatParam = Params.MatParam
        Params.Told .= T

        Params_Q = deepcopy(Params)
        T_Q = deepcopy(T)
        Params_Q.Told .= T_Q
        ȧ = Sillthick / Sill_int_yr / SecYear

        rocks = zero(T)
        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0
        A_inj = 0.0
        rng = MersenneTwister(42)
        injected_count = 0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ = Δ, N = N, BC = BC, Params = Params, MatParam = MatParam, verbose = false)
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            for _ in 1:n_injections
                Sill_z0 = rand(rng, (-Sillbot * 1.0e3):1:(-Silltop * 1.0e3))
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick = Sillthick, Sill_z0 = Sill_z0, Sill_T = Tsill)
                Params.Told .= T
                injected_count += 1
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill = Tsill, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ = Δ, N = N, BC = BC, Params = Params_Q, MatParam = MatParam, verbose = false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        zone = findall(-Sillbot * 1.0e3 .<= z .<= -Silltop * 1.0e3)
        profile_rms = sqrt(sum(abs2, T .- T_Q) / length(T))
        zone_mean_diff = abs(sum(T[zone] .- T_Q[zone]) / length(zone))
        @test injected_count == floor(Int, nt * Δt / (Sill_int_yr * SecYear))
        @test profile_rms < 5.0
        @test zone_mean_diff < 1.0
    end

    @testset "discrete sills vs Q_magma agree with larger, less-frequent sills (advection matters)" begin
        H = 40.0
        γ = 20.0
        Ttop = 0.0
        Tbot = Ttop + H * γ
        nz = floor(Int64, H * 1.0e3 / 20.0)
        Δt = 100SecYear
        Tsill = 1200.0
        Sillthick = 100.0
        Sill_int_yr = 1000.0
        Silltop = 10.0
        Sillbot = 20.0
        nt = 5000

        MatParam = (
            SetMaterialParams(
                Name = "RockMelt", Phase = 0,
                Density = ConstantDensity(ρ = 2700kg / m^3),
                LatentHeat = ConstantLatentHeat(Q_L = 0.0J / kg),
                RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0 = 0.0e-7Watt / m^3),
                Conductivity = ConstantConductivity(k = 3.0),
                HeatCapacity = ConstantHeatCapacity(),
                Melting = MeltingParam_Assimilation()
            ),
        )

        Params, BC, N, Δ, T, z = QMagma.init_model(nz = nz, L = H * 1.0e3, Geotherm = γ, Ttop = Ttop, Tbot = Tbot, Δt = Δt, MatParam = MatParam)
        Params.Told .= T

        Params_Q = deepcopy(Params)
        T_Q = deepcopy(T)
        Params_Q.Told .= T_Q
        ȧ = Sillthick / Sill_int_yr / SecYear

        rocks = zero(T)
        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0
        A_inj = 0.0
        rng = MersenneTwister(42)
        injected_count = 0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ = Δ, N = N, BC = BC, Params = Params, MatParam = MatParam, verbose = false)
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            for _ in 1:n_injections
                Sill_z0 = rand(rng, (-Sillbot * 1.0e3):1:(-Silltop * 1.0e3))
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick = Sillthick, Sill_z0 = Sill_z0, Sill_T = Tsill)
                Params.Told .= T
                injected_count += 1
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill = Tsill, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ = Δ, N = N, BC = BC, Params = Params_Q, MatParam = MatParam, verbose = false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        zone = findall(-Sillbot * 1.0e3 .<= z .<= -Silltop * 1.0e3)
        profile_rms = sqrt(sum(abs2, T .- T_Q) / length(T))
        zone_mean_diff = abs(sum(T[zone] .- T_Q[zone]) / length(zone))
        @test injected_count == floor(Int, nt * Δt / (Sill_int_yr * SecYear))
        @test profile_rms < 10.0
        @test zone_mean_diff < 2.0
    end

    @testset "crack_perp_displacement" begin
        d = 100.0
        @test QMagma.crack_perp_displacement(0.0, d) ≈ d            # max displacement at sill center
        @test QMagma.crack_perp_displacement(1.0e6, d) ≈ 0.0 atol = 1.0e-2 # decays far from the sill
        @test QMagma.crack_perp_displacement(0.0, d) > QMagma.crack_perp_displacement(1.0e3, d)
    end

    @testset "semilagrangian_advection" begin
        z = collect(-10.0e3:100.0:0.0)
        T = collect(range(0.0, 100.0, length = length(z)))

        # zero displacement leaves the field unchanged
        T_same = QMagma.semilagrangian_advection(T, zero(z), z)
        @test T_same ≈ T
    end

    @testset "advect_markers!" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 41, L = 40.0e3, Geotherm = 20.0,
            Ttop = 0.0, Tbot = 800.0, Δt = 200SecYear
        )
        Params.Told .= T

        Tsill = 1200.0
        Silltop = 10.0
        Sillbot = 20.0
        ȧ = 100.0 / 500SecYear

        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill = Tsill, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)

        markers = [-30.0e3, -15.0e3, -5.0e3]   # below, inside, and above the injection zone
        markers0 = copy(markers)
        QMagma.advect_markers!(markers, Params)

        @test markers[1] < markers0[1]   # marker below the zone is pushed further down
        @test markers[2] ≈ markers0[2]   # marker inside the zone (w=0) does not move
        @test markers[3] > markers0[3]   # marker above the zone is pushed further up
    end

    @testset "passive tracers" begin
        Silltop, Sillbot = 10.0, 20.0

        tracers = QMagma.init_tracers(Silltop, Sillbot; n = 5)
        @test length(tracers) == 5
        @test all(t -> t.phase == 0, tracers)
        @test all(t -> isempty(t.time_vec) && isempty(t.T_vec), tracers)
        @test extrema(t.z for t in tracers) == (-Sillbot * 1.0e3, -Silltop * 1.0e3)

        Sill_z0, Sill_thick, Sill_T = -15.0e3, 100.0, 1200.0
        QMagma.add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n = 3)
        @test length(tracers) == 8
        new_tracers = tracers[(end - 2):end]
        @test all(t -> t.phase == 1, new_tracers)
        @test all(t -> t.T == Sill_T, new_tracers)
        @test all(t -> Sill_z0 - Sill_thick / 2 <= t.z <= Sill_z0 + Sill_thick / 2, new_tracers)

        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 41, L = 40.0e3, Geotherm = 20.0,
            Ttop = 0.0, Tbot = 800.0, Δt = 200SecYear
        )
        Params.Told .= T
        ȧ = 100.0 / 500SecYear
        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill = Sill_T, ȧ = ȧ, Silltop = Silltop, Sillbot = Sillbot)

        push!(tracers, QMagma.Tracer(-30.0e3, 0.0, 0.0, 0, Float64[], Float64[]))  # below the zone

        z_before = [t.z for t in tracers]
        QMagma.advect_tracers!(tracers, Params)
        z_after = [t.z for t in tracers]
        @test z_after[end] != z_before[end]        # tracer outside the (w=0) zone moved
        @test tracers[3].z ≈ z_before[3]            # a zone-interior tracer (w=0) stays put

        QMagma.update_tracers_T!(tracers, T, z, 0.001)
        @test all(t -> length(t.time_vec) == 1 && length(t.T_vec) == 1, tracers)
        @test all(t -> t.time_vec[end] == 0.001, tracers)
        @test all(t -> t.T_vec[end] == t.T, tracers)
        @test all(t -> t.phi == 0.0, tracers[1:5])   # zone tracers' phi untouched when not passed in

        QMagma.update_tracers_T!(tracers, T, z, 0.002, Params.ϕ)
        @test all(t -> length(t.time_vec) == 2 && length(t.T_vec) == 2, tracers)
        @test tracers[1].phi >= 0.0   # phi now interpolated from Params.ϕ
    end

    @testset "compute_zircon_ages" begin
        time_Myr = collect(range(0.0, 0.2, length = 20))
        T_C = collect(range(900.0, 650.0, length = 20))   # monotonic cooling path

        tracers = [
            QMagma.Tracer(-10.0e3, T_C[end], 0.0, 1, copy(time_Myr), copy(T_C)),
            QMagma.Tracer(-12.0e3, 0.0, 0.0, 0, Float64[], Float64[]),   # too few points: skipped
        ]

        result = QMagma.compute_zircon_ages(tracers; nx = 30)
        @test length(result.age_years) == 1
        @test length(result.zircon_radius_um) == 1
        @test result.age_years[1] > 0
        @test result.zircon_radius_um[1] > 0

        result2 = QMagma.compute_zircon_ages(tracers; nx = 30, return_results = true)
        @test length(result2.results) == 1
        @test result2.age_years[1] ≈ QMagma.volume_averaged_age(result2.results[1])
    end

    @testset "add_zone_tracers!" begin
        Silltop, Sillbot, Tsill = 10.0, 20.0, 1200.0
        tracers = QMagma.Tracer[]

        QMagma.add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n = 4)
        @test length(tracers) == 4
        @test all(t -> t.phase == 1, tracers)
        @test all(t -> t.T == Tsill, tracers)
        @test all(t -> t.z ≈ -(Silltop + Sillbot) / 2 * 1.0e3, tracers)

        QMagma.add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n = 2)
        @test length(tracers) == 6
    end

    @testset "advect_tracers_sill!" begin
        Sill_z0, Sill_thick = -15.0e3, 200.0

        tracers = [
            QMagma.Tracer(-15.0e3, 0.0, 0.0, 1, Float64[], Float64[]),    # inside the sill
            QMagma.Tracer(-10.0e3, 0.0, 0.0, 0, Float64[], Float64[]),    # above the sill
            QMagma.Tracer(-20.0e3, 0.0, 0.0, 0, Float64[], Float64[]),    # below the sill
        ]
        z0 = [t.z for t in tracers]

        QMagma.advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType = :elastic)

        @test tracers[1].z ≈ z0[1]      # inside the sill: unaffected
        @test tracers[2].z > z0[2]      # above the sill: pushed further up
        @test tracers[3].z < z0[3]      # below the sill: pushed further down

        # Sill_thick is the full aperture, shared equally by the two walls.
        tracers2 = [QMagma.Tracer(-10.0e3, 0.0, 0.0, 0, Float64[], Float64[])]
        QMagma.advect_tracers_sill!(tracers2, Sill_z0, Sill_thick; SillType = :constant)
        @test tracers2[1].z ≈ -10.0e3 + Sill_thick / 2
    end

    @testset "insert_sill" begin
        z = collect(-10.0e3:100.0:0.0)
        T = fill(200.0, length(z))
        rocks = zeros(length(z))

        Sill_z0 = -5.0e3
        Sill_thick = 400.0
        Sill_T = 1200.0
        T2, rocks2 = QMagma.insert_sill(
            T, rocks, z; Sill_thick = Sill_thick, Sill_z0 = Sill_z0,
            Sill_T = Sill_T, SillType = :constant
        )

        @test length(T2) == length(z)
        @test maximum(T2) ≈ Sill_T

        ind = findall(abs.(z .- Sill_z0) .<= Sill_thick / 2)
        @test T2[ind] ≈ [700.0, 1200.0, 1200.0, 1200.0, 700.0]
        @test rocks2[ind] ≈ [0.5, 1.0, 1.0, 1.0, 0.5]
        @test sum(rocks2) > 0
        @test QMagma.integrated_content(rocks2, z) == Sill_thick
        @test all(0 .<= rocks2 .<= 1)

        # magma pushed out through a domain boundary: nothing leaves while the grey sits
        # deep in the column, and a sill emplaced right under the surface expels the grey
        # the displacement carries past z = 0
        Δz_grid = z[2] - z[1]
        deep = zeros(length(z)); deep[findall(-6.0e3 .<= z .<= -4.0e3)] .= 1.0
        _, _, h_out_deep = QMagma.insert_sill(
            T, deep, z; Sill_thick = Sill_thick, Sill_z0 = Sill_z0,
            Sill_T = Sill_T
        )
        @test h_out_deep ≈ 0.0 atol = 1.0e-9

        shallow = zeros(length(z)); shallow[findall(z .>= -500.0)] .= 1.0
        _, rocks_shallow, h_out_shallow = QMagma.insert_sill(
            T, shallow, z; Sill_thick = Sill_thick,
            Sill_z0 = -1.0e3, Sill_T = Sill_T
        )
        @test h_out_shallow > 0
        @test length(rocks_shallow) == length(z)
        # the outflow is exactly what the opening displacement carries off the grid
        z_shift = z .- (-1.0e3)
        Displ = zero(z_shift)
        above, below = findall(z_shift .> 0), findall(z_shift .< 0)
        Displ[above] .= QMagma.crack_perp_displacement(z_shift[above], Sill_thick / 2; r = 5.0e3)
        Displ[below] .= -QMagma.crack_perp_displacement(z_shift[below], Sill_thick / 2; r = 5.0e3)
        expected_out = QMagma.integrated_content(shallow, z) -
            QMagma.integrated_content(QMagma.conservative_advection(shallow, Displ, z), z)
        @test h_out_shallow ≈ expected_out
    end

    @testset "insert_sill conserves injected crust (grey rocks)" begin
        # regression: the phase indicator must not leak under repeated advection. The old
        # semi-Lagrangian + round scheme lost ~25% of the grey over 40 injections; the
        # conservative remap keeps Σ rocks·Δz ≈ total injected thickness.
        z = collect(-40.0e3:100.0:0.0)
        Δz = 100.0
        T = fill(400.0, length(z))
        rocks = zero(z)
        injected = 0.0
        depths = collect(-30.0e3:1.0:-5.0e3)
        for k in 1:40
            z0 = depths[(k * 911) % length(depths) + 1]   # deterministic spread of depths
            T, rocks = QMagma.insert_sill(T, rocks, z; Sill_thick = 400.0, Sill_z0 = z0, Sill_T = 1200.0)
            injected += 400.0
        end
        grey = QMagma.integrated_content(rocks, z)
        @test grey ≈ injected atol = 1.0e-8
    end

    @testset "insert_sill adds its full content inside intruded magma" begin
        z = collect(-40.0e3:200.0:0.0)
        Δz = 200.0
        T = fill(600.0, length(z))
        rocks = zeros(length(z))
        rocks[findall(-20.0e3 .<= z .<= -10.0e3)] .= 1.0
        content = QMagma.integrated_content(rocks, z)

        _, rocks2, h_out = QMagma.insert_sill(
            T, rocks, z; Sill_thick = 100.0, Sill_z0 = -15.0e3,
            Sill_T = 1200.0
        )
        @test h_out ≈ 0.0 atol = 1.0e-9                 # the pile is nowhere near a boundary
        gain = QMagma.integrated_content(rocks2, z) - content
        @test gain ≈ 100.0

        # the loss is the clipping, not the remap: the advection alone conserves content
        # and pushes cells past 1
        z_shift = z .- (-15.0e3)
        Displ = zero(z_shift)
        above, below = findall(z_shift .> 0), findall(z_shift .< 0)
        Displ[above] .= QMagma.crack_perp_displacement(z_shift[above], 50.0; r = 5.0e3)
        Displ[below] .= -QMagma.crack_perp_displacement(z_shift[below], 50.0; r = 5.0e3)
        advected = QMagma.conservative_advection(rocks, Displ, z)
        @test QMagma.integrated_content(advected, z) ≈ content
        @test maximum(advected) > 1.0
    end

    @testset "eruption trigger control states" begin
        none = QMagma.eruption_control_state("None")
        @test !any(values(none))

        dh = QMagma.eruption_control_state("D&H 3-phase")
        @test all(values(dh))
        @test_throws "unknown eruption trigger" QMagma.eruption_control_state("typo")
    end

    @testset "composition pairs melting with its solubility law and melt viscosity" begin
        # the three laws are calibrated on different melts, so they are one choice
        basalt = QMagma.gui_composition("MeltingParam_Basalt")
        rhyo = QMagma.gui_composition("MeltingParam_Rhyolite")
        assim = QMagma.gui_composition("MeltingParam_Assimilation")
        @test basalt.solubility isa Mafic_Solubility
        @test rhyo.solubility isa Liu2005_Solubility
        @test assim.solubility isa Liu2005_Solubility      # crustal anatexis is silicic
        @test assim.melting isa MeltingParam_Assimilation
        # rhyolite melts at a lower T than basalt under the same parameterisation type
        @test QMagma.liquidus_temperature(rhyo.melting) <
            QMagma.liquidus_temperature(basalt.melting)
        # and the two solubility laws genuinely differ at the same P,T
        m_maf = first(compute_dissolved(basalt.solubility, (; P = 2.0e8, T = 1123.15, X_co2 = 0.0)))
        m_liu = first(compute_dissolved(rhyo.solubility, (; P = 2.0e8, T = 1123.15, X_co2 = 0.0)))
        @test m_maf != m_liu
        # both silicic compositions carry the same melt viscosity, and it is not the basaltic one
        @test rhyo.melt_viscosity === assim.melt_viscosity !== basalt.melt_viscosity
        # each LinearMeltViscosity fit is linear in 1/T over its own melt's range, so the
        # laws are compared where each applies
        ep_maf = QMagma.EruptionParams(melt_viscosity = basalt.melt_viscosity)
        ep_sil = QMagma.EruptionParams(melt_viscosity = rhyo.melt_viscosity)
        η_maf, η_sil = QMagma.magma_viscosity(ep_maf, 1473.15), QMagma.magma_viscosity(ep_sil, 1173.15)
        @test η_sil > 1.0e3 * η_maf
        Δρg, ΔP = 300.0 * ep_maf.g, 20.0e6
        @test QMagma.max_ascent_length(ΔP, Δρg, η_sil, ep_sil) <
            0.1 * QMagma.max_ascent_length(ΔP, Δρg, η_maf, ep_maf)
        @test_throws "unknown magma composition" QMagma.gui_composition("typo")
    end

    @testset "GUI flux histories" begin
        constant = QMagma.gui_flux_history(
            "Constant";
            base_m_per_yr = 0.1, peak_m_per_yr = 0.2, start_kyr = 10.0, end_kyr = 20.0
        )
        ramp = QMagma.gui_flux_history(
            "Linear ramp";
            base_m_per_yr = 0.1, peak_m_per_yr = 0.3, start_kyr = 10.0, end_kyr = 20.0
        )
        pulse = QMagma.gui_flux_history(
            "Pulse";
            base_m_per_yr = 0.1, peak_m_per_yr = 0.3, start_kyr = 10.0, end_kyr = 20.0
        )
        @test constant.mode == :constant
        @test ramp(15.0e3SecYear) * SecYear ≈ 0.2
        @test pulse(15.0e3SecYear) * SecYear ≈ 0.3
        @test_throws "unknown flux mode" QMagma.gui_flux_history(
            "typo";
            base_m_per_yr = 0.1, peak_m_per_yr = 0.2, start_kyr = 10.0, end_kyr = 20.0
        )
    end

    @testset "eruption trigger criteria" begin
        fires(; kwargs...) = QMagma.eruption_fires(;
            h_erupt = 300.0, near_boundary = false, Δz = 100.0, kwargs...
        )

        @test fires()
        @test !fires(; h_erupt = 0.0)
        @test !fires(; near_boundary = true)
        # a withdrawal thinner than two cells stays queued rather than firing
        @test !fires(; h_erupt = 200.0)
    end

    @testset "recorded surface subsidence" begin
        @test QMagma.collapse_surface_subsidence(:caldera, 250.0) == 250.0
        @test QMagma.collapse_surface_subsidence(:hybrid, 250.0) == 250.0
        @test_throws "unknown eruption collapse method" QMagma.collapse_surface_subsidence(:typo, 250.0)
    end

    @testset "erupt_melt!" begin
        z = collect(-10.0e3:100.0:0.0)
        T = fill(800.0, length(z))
        rocks = zeros(length(z))

        Erupt_z0 = -5.0e3
        Erupt_thick = 1000.0
        ind = findall(abs.(z .- Erupt_z0) .<= Erupt_thick / 2)
        T[ind] .= 1200.0
        rocks[ind] .= 1.0

        T2, rocks2 = QMagma.erupt_melt!(T, rocks, z; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick)

        @test length(T2) == length(z)
        # caldera subsidence: the erupted band's heat leaves exactly, replaced at the
        # top by the surface-temperature fill; no rock parcel changes temperature
        @test sum(T2) ≈ sum(T) - sum(T[ind]) + length(ind) * T[end]
        # the roof block (uniformly 800.0) dropped onto the chamber floor
        @test all(T2[ind] .≈ 800.0)
        @test all(rocks2[ind] .== 0)

        # erupting from the middle of a wider intruded pile: the roof grey drops onto
        # the floor grey, so the total drops by exactly the erupted band's content and
        # no host-rock gap is left inside the remaining grey
        rocks_wide = zeros(length(z))
        ind_wide = findall(abs.(z .- Erupt_z0) .<= 2000.0)
        rocks_wide[ind_wide] .= 1.0
        _, rocks3 = QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick)
        @test length(rocks3) == length(z)
        @test sum(rocks3) == sum(rocks_wide) - length(ind)
        @test all((rocks3 .== 0) .| (rocks3 .== 1))
        grey_idx = findall(rocks3 .> 0)
        @test all(diff(grey_idx) .== 1)

        # the whole grey envelope above the vent subsides with the roof: its top edge
        # drops by the band's actual grid footprint (length(ind) cells)
        Δz_grid = z[2] - z[1]
        @test maximum(z[rocks3 .> 0]) ≈ maximum(z[rocks_wide .> 0]) - length(ind) * Δz_grid

        # hybrid variant: floor rises elastically, roof transitions to a rigid
        # subsidence - the surface sinks by ~the erupted thickness and the warped
        # grid stays monotonic
        D = QMagma.collapse_displacement(z; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = :hybrid)
        @test all(diff(z .+ D) .> 0)
        @test isapprox(D[end], -Erupt_thick; rtol = 0.1)
        @test D[1] == 0.0
        T5, rocks6 = QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = :hybrid)
        @test all(T5[ind] .≈ 800.0)                    # walls meet at their own T
        @test minimum(T5) >= minimum(T) - 1.0e-8         # intensive transport: no dilution values
        @test all(rocks6 .>= -1.0e-9)
        @test sum(rocks6) <= sum(rocks_wide)           # grey conserved minus the erupted band
        # grey is the conservative fractional remap (co-moves with T, no round), so the
        # surviving grey stays non-negative and does not inflate away from a phase indicator
        @test maximum(rocks6) <= 2.0

        # every closure reports the intruded magma it took out of the column, the term the
        # magma-volume budget debits (grey removed = erupted band + boundary outflow)
        for closure in (:caldera, :hybrid)
            rock_in = copy(rocks_wide)
            _, rock_out, h_out = QMagma.erupt_melt!(
                T, rock_in, z; Erupt_z0 = Erupt_z0,
                Erupt_thick = Erupt_thick, method = closure
            )
            @test h_out ≈ (sum(rock_in) - sum(rock_out)) * Δz_grid
            @test h_out > 0
        end
        # a band that misses the grid removes nothing
        @test QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0 = 1.0e4, Erupt_thick = Erupt_thick)[3] == 0.0

        @test_throws "unknown eruption collapse method" QMagma.erupt_melt!(
            T, rocks, z;
            Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = :typo
        )
        @test_throws "collapse_displacement supports only" QMagma.collapse_displacement(
            z;
            Erupt_z0 = Erupt_z0, Erupt_thick = Erupt_thick, method = :caldera
        )
    end

    @testset "melt_thickness" begin
        z = collect(-10.0e3:100.0:0.0)
        ϕ = zeros(length(z))
        ind = findall(-6.0e3 .<= z .<= -4.0e3)
        ϕ[ind] .= 0.8
        @test QMagma.melt_thickness(ϕ, z, -6.0e3, -4.0e3) ≈ 0.8 * 2.0e3
        # empty interval
        @test QMagma.melt_thickness(ϕ, z, -2.0e3, -1.0e3) == 0.0
    end

    @testset "collapse_markers!/collapse_tracers!" begin
        Erupt_z0, Erupt_thick = -5.0e3, 1000.0

        # caldera: above drops by the full thickness, inside lands on the floor, below stays
        m = [-2.0e3, -5.0e3, -8.0e3]
        QMagma.collapse_markers!(m, Erupt_z0, Erupt_thick)
        @test m ≈ [-3.0e3, -5.5e3, -8.0e3]

        # hybrid: inside snaps to the vent center, the floor below rises toward it by
        # less than the half-thickness, the roof above subsides toward the full thickness
        m2 = [-2.0e3, -5.0e3, -8.0e3]
        QMagma.collapse_markers!(m2, Erupt_z0, Erupt_thick; method = :hybrid)
        @test m2[2] == Erupt_z0
        @test -3.0e3 <= m2[1] < -2.5e3
        @test -8.0e3 < m2[3] < -7.5e3

        # tracers follow the same displacement
        tracers = [
            QMagma.Tracer(-2.0e3, 600.0, 0.0, 0, Float64[], Float64[]),
            QMagma.Tracer(-8.0e3, 600.0, 0.0, 0, Float64[], Float64[]),
        ]
        QMagma.collapse_tracers!(tracers, Erupt_z0, Erupt_thick)
        @test tracers[1].z ≈ -3.0e3
        @test tracers[2].z ≈ -8.0e3
    end

    @testset "extract_erupted_tracers!" begin
        z = collect(-6.0e3:500.0:-4.0e3)
        ϕ = [0.2, 0.4, 0.8, 0.6, 0.2]
        tracers = [
            QMagma.Tracer(-5.0e3, 1200.0, 1.0, 1, Float64[], Float64[]),
            QMagma.Tracer(-4.5e3, 1100.0, 0.7, 1, Float64[], Float64[]),
            QMagma.Tracer(-5.5e3, 900.0, 0.4, 0, Float64[], Float64[]),
            QMagma.Tracer(-6.0e3, 600.0, 0.0, 0, Float64[], Float64[]),
        ]

        erupted, h_cargo, _ = QMagma.extract_erupted_tracers!(
            MersenneTwister(1), tracers,
            ϕ, z, -5.5e3, -4.5e3, 1.0e3; eligible_phase = 1
        )

        @test length(erupted) == 2
        @test h_cargo ≈ 0.8 * 500 + 0.6 * 250
        @test length(tracers) == 2
        @test sort([tracer.phase for tracer in tracers]) == [0, 0]

        # Each cell's represented melt is independent of how many tracers were seeded.
        one = [QMagma.Tracer(-5.0e3, 1200.0, 0.8, 1, Float64[], Float64[])]
        many = [QMagma.Tracer(-5.0e3, 1200.0, 0.8, 1, Float64[], Float64[]) for _ in 1:4]
        _, h_one, _ = QMagma.extract_erupted_tracers!(
            MersenneTwister(2), one, ϕ, z,
            -5.25e3, -4.75e3, 1.0e3; eligible_phase = 1
        )
        _, h_many, _ = QMagma.extract_erupted_tracers!(
            MersenneTwister(2), many, ϕ, z,
            -5.25e3, -4.75e3, 1.0e3; eligible_phase = 1
        )
        @test h_one ≈ h_many
        @test h_one ≈ 0.8 * 500

        # All-phase cargo represents the full mush even when some melt-bearing cells
        # have no tracer. Sparse diagnostic sampling must not shrink physical withdrawal.
        sparse = [QMagma.Tracer(-5.0e3, 1200.0, 0.8, 1, Float64[], Float64[])]
        _, h_sparse, q_sparse = QMagma.extract_erupted_tracers!(
            MersenneTwister(3), sparse, ϕ, z,
            -5.5e3, -4.5e3, 1.0e3; eligible_phase = nothing
        )
        @test h_sparse ≈ 0.4 * 250 + 0.8 * 500 + 0.6 * 250
        # ...and that lone tracer is therefore the sampler's resolution: it can only
        # represent the withdrawal in one all-or-nothing lump
        @test q_sparse ≈ h_sparse

        partial = [
            QMagma.Tracer(-5.5e3, 900.0, 0.4, 1, Float64[], Float64[]),
            QMagma.Tracer(-5.0e3, 1200.0, 0.8, 1, Float64[], Float64[]),
            QMagma.Tracer(-4.5e3, 1100.0, 0.6, 1, Float64[], Float64[]),
        ]
        cargo, h_partial, _ = QMagma.extract_erupted_tracers!(
            MersenneTwister(4), partial,
            ϕ, z, -5.5e3, -4.5e3, 350.0; eligible_phase = 1
        )
        @test 1 <= length(cargo) <= 2
        @test length(cargo) + length(partial) == 3
        @test h_partial > 0

        @test_throws "h_erupt must be nonnegative" QMagma.extract_erupted_tracers!(
            MersenneTwister(3), QMagma.Tracer[], ϕ, z, -5.5e3, -4.5e3, -1.0;
            eligible_phase = nothing
        )

        # The last tracer is kept only if taking it lands nearer the target than leaving it.
        # Three equal-weight tracers share the 400 m of melt in one cell, 133.3 m each; a
        # 160 m request overshoots by 106.7 m on the second and undershoots by 26.7 m on the
        # first, so the cargo is one tracer.
        equal = [QMagma.Tracer(-5.0e3, 1200.0, 0.8, 1, Float64[], Float64[]) for _ in 1:3]
        under, h_under, _ = QMagma.extract_erupted_tracers!(
            MersenneTwister(5), equal, ϕ, z, -5.25e3, -4.75e3, 160.0; eligible_phase = 1
        )
        @test length(under) == 1
        @test length(equal) == 2
        @test h_under ≈ 0.8 * 500 / 3
    end

    @testset "unified eruption event fails on split bookkeeping" begin
        z = collect(-1000.0:100.0:0.0)
        T = fill(1000.0, length(z))
        ϕ = zeros(length(z)); ϕ[4:8] .= 0.8
        Params, = QMagma.init_model(
            nz = length(z), L = 1000.0, Geotherm = 0.0,
            Ttop = 1000.0, Tbot = 1000.0, Δt = 1.0
        )
        tracers = [QMagma.Tracer(zi, 1000.0, 0.8, 1, [0.0], [1000.0]) for zi in z[4:8]]

        Tnew, _, cargo, event = QMagma.realize_eruption!(
            MersenneTwister(7), T,
            zeros(length(z)), tracers, ϕ, z, Params.MatParam, Params.Phases;
            realization_time = 1.0, h_requested = 240.0,
            z_lo = z[4], z_hi = z[8],
            trigger = :test, closure = :caldera, eligible_phase = 1
        )

        @test event.requested == 240.0
        # the withdrawal is bulk magma; at the band's width-averaged ϕ = 0.8 it carries
        # 192 m of liquid, and that is what the tracers are sampled to. Whole tracers
        # stand for 80 m of melt apiece, so the cargo lands one tracer short of the target.
        @test event.melt_requested ≈ 192.0
        @test event.cargo_represented ≈ 160.0
        @test event.cargo_count == length(cargo)
        @test event.enthalpy_after == QMagma.column_enthalpy(
            Tnew, z,
            Params.MatParam, Params.Phases
        )
        @test event.magma_removed == 0.0        # no intruded magma in this column to remove

        # with intruded magma present the event carries the grey the closure withdrew,
        # independently of the melt-thickness accounts
        rocks = zeros(length(z)); rocks[3:9] .= 1.0
        _, rocks_after, _, event_grey = QMagma.realize_eruption!(
            MersenneTwister(7), T,
            rocks, [QMagma.Tracer(zi, 1000.0, 0.8, 1, [0.0], [1000.0]) for zi in z[4:8]],
            ϕ, z, Params.MatParam, Params.Phases;
            realization_time = 1.0, h_requested = 240.0,
            z_lo = z[4], z_hi = z[8],
            trigger = :test, closure = :caldera, eligible_phase = 1
        )
        @test event_grey.magma_removed ≈ QMagma.integrated_content(rocks, z) -
            QMagma.integrated_content(rocks_after, z)
        @test event_grey.magma_removed > 0

        # A thinly seeded band resolves a withdrawal only to the melt one tracer stands
        # for, which is many cells' worth. The event's cargo tolerance must widen to that
        # quantum, or a sound eruption is rejected for a sampling artifact.
        sparse_tracers = [QMagma.Tracer(z[6], 1000.0, 0.8, 1, [0.0], [1000.0])]
        _, _, _, event_sparse = QMagma.realize_eruption!(
            MersenneTwister(7), copy(T),
            zeros(length(z)), sparse_tracers, ϕ, z, Params.MatParam, Params.Phases;
            realization_time = 1.0, h_requested = 240.0,
            z_lo = z[4], z_hi = z[8],
            trigger = :test, closure = :caldera, eligible_phase = nothing
        )
        # one tracer carries the whole band, so the cargo it represents overshoots the
        # 192 m of liquid in that withdrawal by far more than the half-cell floor of 50 m
        @test event_sparse.cargo_represented ≈ QMagma.melt_thickness(ϕ, z, z[4], z[8])
        @test abs(event_sparse.cargo_represented - event_sparse.melt_requested) > 50.0

        event_args = function (; cargo = 10.0)
            return (;
                trigger_time = 1.0, realization_time = 1.0, requested = 10.0,
                cargo_represented = cargo, z_lo = -1.0, z_hi = 1.0,
                z_centroid = 0.0, trigger = :test, closure = :caldera,
                enthalpy_before = 100.0, enthalpy_after = 90.0, erupted_enthalpy = 10.0,
            )
        end
        @test_throws "cargo-represented=8.0" QMagma.EruptionEvent(;
            event_args(cargo = 8.0)..., cargo_atol = 1.0
        )
        # cargo within the declared tracer-resolution tolerance is accepted
        @test QMagma.EruptionEvent(; event_args(cargo = 8.0)..., cargo_atol = 3.0) isa
            QMagma.EruptionEvent
    end

    @testset "overpressure trigger (D&H 3-phase)" begin
        ep = QMagma.EruptionParams(ΔP_crit = 20.0e6, m_w = 0.05)
        # higher P dissolves more water -> less gas -> denser
        ρlo, glo = QMagma.mixture_density(5.0e7, 1100.0, 0.7, ep)
        ρhi, ghi = QMagma.mixture_density(3.0e8, 1100.0, 0.7, ep)
        @test 0 <= glo <= 1 && 0 <= ghi <= 1
        @test ρhi > ρlo && ghi < glo

        z = collect(-30.0e3:100.0:0.0)
        T_col = 600.0 .- z ./ 1.0e3 .* 20.0          # 20 K/km geotherm, 600 °C at the surface
        P_lith(z_target) = QMagma.lithostatic_pressure(ep, T_col, z, z_target)
        function impose_pressure!(state, params, P, T_K, ϕ_mush)
            state.P = P
            state.M = first(QMagma.mixture_density(P, T_K, ϕ_mush, params)) * state.V
            state.M_H2O = params.m_w * state.M
            state.M_initial = state.M
            state.M_in = state.M_out = state.M_boundary = state.mass_residual = 0.0
            return state
        end
        ϕ = [(-15.0e3 <= zi <= -12.0e3) ? 0.8 : 0.1 for zi in z]
        ind, V_e, zc = QMagma.eruptible_mush(ϕ, z; ϕ_erupt = ep.ϕ_erupt)
        @test !isempty(ind) && V_e == 3.0e3 && zc ≈ -13.5e3

        # A second disconnected lens is a different pressure reservoir, not extra volume.
        ϕ[-5.0e3 .<= z .<= -4.0e3] .= 0.9
        ind2, V_e2, zc2 = QMagma.eruptible_mush(ϕ, z; ϕ_erupt = ep.ϕ_erupt)
        @test ind2 == ind
        @test V_e2 == V_e
        @test zc2 == zc

        # a chamber at lithostatic with no recharge drains nothing
        st = QMagma.EruptionState(); QMagma.init_eruption!(st, P_lith(zc))
        QMagma.step_overpressure!(st, ep, 1200.0 + 273.15, 0.8, V_e, 0.0, 1.0e10; z_centroid = zc)
        QMagma.step_overpressure!(st, ep, 1200.0 + 273.15, 0.8, V_e, 0.0, 1.0e10; z_centroid = zc)
        @test st.P ≈ st.P_lith
        @test st.h_erupt == 0.0

        # gas lock-up: a chamber whose gas fraction exceeds ϕ_g_crit cannot drain however
        # overpressured, because the criteria are applied at the moment of drainage. The
        # chamber must be shallow enough to exsolve gas at all — at 13.5 km the solubility
        # law keeps every bit of m_w dissolved and ϕ_g is identically zero.
        ep_locked = QMagma.EruptionParams(ΔP_crit = 20.0e6, ϕ_g_crit = 1.0e-9, m_w = 0.05)
        st_lock = QMagma.EruptionState(); QMagma.init_eruption!(st_lock, 1.324e8)
        QMagma.step_overpressure!(
            st_lock, ep_locked, 1100.0, 0.7, 1000.0, 1.0e-9, 1.0e11;
            z_centroid = -5.0e3
        )
        impose_pressure!(st_lock, ep_locked, st_lock.P_lith + 5ep_locked.ΔP_crit, 1100.0, 0.7)
        QMagma.step_overpressure!(
            st_lock, ep_locked, 1100.0, 0.7, 1000.0, 1.0e-9, 1.0e9;
            z_centroid = -5.0e3
        )
        @test st_lock.ϕ_g > ep_locked.ϕ_g_crit
        @test st_lock.h_erupt == 0.0

        # second boiling (D&H's dominant trigger): crystallizing (ϕ_melt↓) must exsolve gas
        _, g_wet = QMagma.mixture_density(1.0e8, 1100.0, 0.8, ep)
        _, g_dry = QMagma.mixture_density(1.0e8, 1100.0, 0.4, ep)
        @test g_dry > g_wet

        # Recharge builds pressure without an accessible failure threshold.
        ep.η_r = 1.0e30
        ep.ΔP_crit = 1.0e15
        QMagma.init_eruption!(st, P_lith(zc))
        for _ in 1:200
            QMagma.step_overpressure!(
                st, ep, 1200.0 + 273.15, 0.8, V_e, 1.0e-9, 1.0e10;
                z_centroid = zc
            )
        end
        @test st.P - st.P_lith > 0
        @test st.h_erupt == 0.0

        # A reachable threshold drains no more melt than was recharged.
        ep.ΔP_crit = 20.0e6
        st2 = QMagma.EruptionState(); QMagma.init_eruption!(st2, P_lith(zc))
        ȧ_t, Δt_t, nstep = 1.0e-9, 1.0e11, 60
        QMagma.step_overpressure!(
            st2, ep, 1200.0 + 273.15, 0.8, V_e, ȧ_t, Δt_t;
            z_centroid = zc
        )
        erupted = 0.0
        for _ in 1:nstep
            QMagma.step_overpressure!(
                st2, ep, 1200.0 + 273.15, 0.8, V_e, ȧ_t, Δt_t;
                z_centroid = zc
            )
            erupted += st2.h_erupt
        end
        recharge = ȧ_t * Δt_t * nstep
        @test 0 < erupted <= 1.05recharge

        # a barrier beyond the reach of overpressure plus buoyancy stalls every drain
        ep_strong = QMagma.EruptionParams(
            ΔP_crit = ep.ΔP_crit, η_r = ep.η_r, σ_barrier = 1.0e12, m_w = 0.05
        )
        st3 = QMagma.EruptionState(); QMagma.init_eruption!(st3, P_lith(zc))
        QMagma.step_overpressure!(
            st3, ep_strong, 1200.0 + 273.15, 0.8, V_e, ȧ_t, Δt_t;
            z_centroid = zc
        )
        impose_pressure!(
            st3, ep_strong, st3.P_lith + 5ep_strong.ΔP_crit, 1200.0 + 273.15, 0.8
        )
        QMagma.step_overpressure!(
            st3, ep_strong, 1200.0 + 273.15, 0.8, V_e, ȧ_t, 1.0e9;
            z_centroid = zc
        )
        @test st3.h_erupt == 0.0

        # --- the chamber owns its volume ---------------------------------------------
        # seeded from the grid the first time the mush is seen
        st_v = QMagma.EruptionState(); QMagma.init_eruption!(st_v, P_lith(zc))
        QMagma.step_overpressure!(
            st_v, ep, 1200.0 + 273.15, 0.8, V_e, 0.0, 1.0e10;
            z_centroid = zc, η_r = 1.0e19
        )
        @test st_v.V == V_e

        # a chamber held above lithostatic creeps open: the wall relaxation that sets
        # dP/dt also moves the wall, so V grows while ΔP stays positive
        ep_hold = QMagma.EruptionParams(
            ΔP_crit = 1.0e15, σ_barrier = 1.0e12, m_w = 0.05
        )   # never drains
        st_g = QMagma.EruptionState(); QMagma.init_eruption!(st_g, P_lith(zc))
        for _ in 1:40
            QMagma.step_overpressure!(
                st_g, ep_hold, 1200.0 + 273.15, 0.8, V_e, 3.0e-8, 1.0e10;
                z_centroid = zc, η_r = 1.0e19
            )
        end
        @test st_g.P - st_g.P_lith > 0
        @test st_g.V > V_e

        # and that creeping open is a relief path: the same recharge into a chamber whose
        # walls cannot creep (a very stiff wall viscosity) has nowhere to put the volume
        # but into pressure, and ends orders of magnitude more overpressured
        st_r = QMagma.EruptionState(); QMagma.init_eruption!(st_r, P_lith(zc))
        for _ in 1:40
            QMagma.step_overpressure!(
                st_r, ep_hold, 1200.0 + 273.15, 0.8, V_e, 3.0e-8, 1.0e10;
                z_centroid = zc, η_r = 1.0e30
            )
        end
        @test st_r.P - st_r.P_lith > 100 * (st_g.P - st_g.P_lith)

        # melting at the mush margins is the grid's contribution, and only its change
        # enters: the chamber does not forget the volume it crept open
        V_before = st_g.V
        QMagma.step_overpressure!(
            st_g, ep_hold, 1200.0 + 273.15, 0.8, V_e + 500.0, 0.0, 1.0e5;
            z_centroid = zc, η_r = 1.0e30
        )
        @test st_g.V ≈ V_before + 500.0 rtol = 1.0e-6

        # η_r is per call, so two chambers sharing one parameter object stay independent
        sh = QMagma.EruptionParams(ΔP_crit = 1.0e15, σ_barrier = 1.0e12, m_w = 0.05)
        st_a = QMagma.EruptionState(); QMagma.init_eruption!(st_a, P_lith(zc))
        st_b = QMagma.EruptionState(); QMagma.init_eruption!(st_b, P_lith(zc))
        for _ in 1:20
            QMagma.step_overpressure!(
                st_a, sh, 1200.0 + 273.15, 0.8, V_e, 3.0e-8, 1.0e10;
                z_centroid = zc, η_r = 1.0e19
            )
            QMagma.step_overpressure!(
                st_b, sh, 1200.0 + 273.15, 0.8, V_e, 3.0e-8, 1.0e10;
                z_centroid = zc, η_r = 1.0e30
            )
        end
        @test st_a.η_r == 1.0e19 && st_b.η_r == 1.0e30
        @test st_a.V > st_b.V
        @test_throws "η_r must be finite and positive" QMagma.step_overpressure!(
            st_a, sh, 1200.0 + 273.15, 0.8, V_e, 0.0, 1.0e10; z_centroid = zc, η_r = -1.0
        )

        # --- an immature body is not a chamber ------------------------------------------
        # A region holding less than h_melt_min of melt evolves thermally with no chamber:
        # without this a single freshly emplaced sill, itself connected at ϕ ≈ 1, registers
        # as its own reservoir and the recharge term counts it as both the chamber and the
        # flux into it, so it erupts in the step it arrives.
        z_s = collect(-30.0e3:20.0:0.0)
        T_s = 600.0 .- z_s ./ 1.0e3 .* 20.0
        ϕ_sill = [(-12.05e3 <= zi <= -11.95e3) ? 1.0 : 0.1 for zi in z_s]   # one 100 m sill
        @test QMagma.melt_thickness(
            ϕ_sill, z_s,
            QMagma.eruptible_mush(ϕ_sill, z_s; ϕ_erupt = 0.5)[1] |> ind -> z_s[first(ind)],
            QMagma.eruptible_mush(ϕ_sill, z_s; ϕ_erupt = 0.5)[1] |> ind -> z_s[last(ind)]
        ) < 500.0
        ep_gate = QMagma.EruptionParams(
            ΔP_crit = 20.0e6, h_melt_min = 500.0, m_w = 0.05
        )
        st_s = QMagma.EruptionState()
        Params_s, _, _, _, _, _ = QMagma.init_model(
            nz = length(z_s), L = 30.0e3, Geotherm = 20.0,
            Ttop = 600.0, Tbot = 1200.0, Δt = 100SecYear
        )
        _, _, _, ev_s = QMagma.step_chamber_eruption!(
            MersenneTwister(3), st_s, ep_gate,
            T_s, zero(T_s), QMagma.Tracer[], ϕ_sill, z_s, Params_s.MatParam, Params_s.Phases;
            ȧ = 1.0e-8, Δt = 1.0e10, time = 0.0, closure = :caldera, margin = 100.0,
            T_background = T_s
        )
        @test ev_s === nothing
        @test !st_s.init          # no chamber state accumulates below the threshold
        @test st_s.h_erupt == 0.0

        # widen the mush past the threshold and the same call engages the chamber
        ϕ_body = [(-15.0e3 <= zi <= -12.0e3) ? 0.8 : 0.1 for zi in z_s]
        QMagma.step_chamber_eruption!(
            MersenneTwister(3), st_s, ep_gate,
            T_s, zero(T_s), QMagma.Tracer[], ϕ_body, z_s, Params_s.MatParam, Params_s.Phases;
            ȧ = 1.0e-8, Δt = 1.0e10, time = 0.0, closure = :caldera, margin = 100.0,
            T_background = T_s
        )
        @test st_s.init && st_s.V > 0

        # --- and charging it far enough realizes an eruption -----------------------------
        # A wide dike (reach goes as w⁴) clears the ascent criterion, so the chamber drains
        # once ΔP crosses ΔP_crit: the column cools, tracers leave as cargo, and the
        # overpressure falls back to ΔP_relax.
        ep_fire = QMagma.EruptionParams(
            ΔP_crit = 2.0e6, σ_barrier = 1.0e4, h_melt_min = 500.0, w_dike = 20.0,
            m_w = 0.05
        )
        st_f = QMagma.EruptionState()
        T_f, rocks_f = copy(T_s), zero(T_s)
        tracers_f = [
            QMagma.Tracer(zi, 1000.0, 1.0, 1, Float64[], Float64[])
                for zi in range(-15.0e3, -12.0e3, length = 40)
        ]
        event_f, cargo_f = nothing, QMagma.Tracer[]
        for k in 1:20
            T_f, rocks_f, cargo, ev = QMagma.step_chamber_eruption!(
                MersenneTwister(3), st_f, ep_fire,
                T_f, rocks_f, tracers_f, ϕ_body, z_s, Params_s.MatParam, Params_s.Phases;
                ȧ = 1.0e-7, Δt = 1.0e10, time = k * 1.0e10, closure = :caldera,
                margin = 100.0, T_background = T_s
            )
            ev === nothing && continue
            event_f, cargo_f = ev, cargo
            break
        end
        @test event_f !== nothing
        @test event_f.trigger == Symbol("D&H 3-phase")
        @test event_f.closure == :caldera
        @test event_f.requested > z_s[2] - z_s[1]     # resolvable on the grid, or it would queue
        @test event_f.z_lo == -15.0e3 && event_f.z_hi == -12.0e3
        # withdrawal is booked, not left pending, and the pressure drops back to ΔP_relax
        @test st_f.h_pending == 0.0
        @test st_f.P - st_f.P_lith ≈ ep_fire.ΔP_relax atol = 1.0e-6
        # the erupted cargo leaves the reservoir population and the column cools
        @test length(cargo_f) + length(tracers_f) == 40
        @test !isempty(cargo_f)
        @test event_f.cargo_count == length(cargo_f)
        @test event_f.chamber == st_f.id               # the event names the body it came from
        @test maximum(T_s .- T_f) > 0                 # the drained column cools

        # --- a disconnected lens is a different chamber ----------------------------------
        # eruptible_mush hands back whichever lens is largest, so the state has to check that
        # the body it charged is still the one in front of it. Without the interval test a
        # deeper lens inherits the shallow chamber's pressure, inventory, and queued
        # withdrawal, and can erupt melt drained from a body that no longer exists.
        st_i = QMagma.EruptionState()
        step_body(ϕ_in) = QMagma.step_chamber_eruption!(
            MersenneTwister(3), st_i, ep_gate,
            T_s, zero(T_s), QMagma.Tracer[], ϕ_in, z_s, Params_s.MatParam, Params_s.Phases;
            ȧ = 1.0e-8, Δt = 1.0e10, time = 1.0e10, closure = :caldera, margin = 100.0,
            T_background = T_s
        )
        step_body(ϕ_body)
        @test st_i.id == 1 && (st_i.z_lo, st_i.z_hi) == (-15.0e3, -12.0e3)
        QMagma.pending_withdrawal!(st_i, 40.0, 500.0, z_s[2] - z_s[1]; time = 1.0e10)
        @test st_i.h_pending == 40.0

        ϕ_deep = [(-25.0e3 <= zi <= -21.0e3) ? 0.8 : 0.1 for zi in z_s]  # larger, disconnected
        step_body(ϕ_deep)
        @test st_i.id == 2                                  # a new body, not the old one moved
        @test (st_i.z_lo, st_i.z_hi) == (-25.0e3, -21.0e3)
        @test st_i.h_pending == 0.0 && isnan(st_i.pending_since)
        @test st_i.V == 4.0e3 && st_i.M_in == 0.0           # inventory seeded from the new lens
        @test st_i.P == st_i.P_lith                         # starting at zero overpressure

        # losing the mush entirely ends the chamber; the next one to form is a third body
        step_body(fill(0.1, length(z_s)))
        @test !st_i.init && st_i.h_pending == 0.0 && st_i.M == 0.0
        step_body(ϕ_body)
        @test st_i.id == 3

        @test_throws "h_melt_min must be finite and nonnegative" QMagma.validate_eruption_params(
            QMagma.EruptionParams(h_melt_min = -1.0)
        )

        # --- host compliance follows the chamber's shape -------------------------------
        ep_c = QMagma.EruptionParams(μ_shear = 1.0e10, R_sill = 5.0e3)
        # a chamber as thick as it is wide is the spherical case, 3/(4μ)
        @test QMagma.host_compliance(ep_c, 2 * ep_c.R_sill) ≈ 3 / (4 * ep_c.μ_shear)
        # flattening it against a fixed radius makes it more compliant, as 1/ε
        @test QMagma.host_compliance(ep_c, 3.0e3) ≈ 3 / (4 * ep_c.μ_shear) / (1.5e3 / 5.0e3)
        @test QMagma.host_compliance(ep_c, 3.0e3) > QMagma.host_compliance(ep_c, 6.0e3)
        # a body thicker than it is wide is not a sill: ε caps at 1 and the sphere is the floor
        @test QMagma.host_compliance(ep_c, 50 * ep_c.R_sill) ≈ 3 / (4 * ep_c.μ_shear)
        @test_throws DomainError QMagma.host_compliance(ep_c, 0.0)
        # and a stiffer host is less compliant at fixed shape
        @test QMagma.host_compliance(QMagma.EruptionParams(μ_shear = 4.0e10, R_sill = 5.0e3), 3.0e3) <
            QMagma.host_compliance(ep_c, 3.0e3)

        # A migrating chamber changes lithostatic pressure without jumping overpressure.
        st_m = QMagma.EruptionState()
        QMagma.update_lithostatic!(st_m, P_lith(-13.0e3))
        @test st_m.P ≈ st_m.P_lith
        st_m.P += 5.0e6
        QMagma.update_lithostatic!(st_m, P_lith(-14.0e3))
        @test isapprox(st_m.P_lith, 2700.0 * ep.g * 14.0e3; rtol = 0.05)
        @test st_m.P - st_m.P_lith ≈ 5.0e6
    end

    @testset "RK gas EOS (Huber 2010, item 1)" begin
        rk(P, T) = compute_density(RedlichKwong_Density(), (; P, T))
        @test rk(1.0e8, 1123.15) > rk(1.0e8, 1173.15)   # hotter -> lighter
        @test rk(3.0e8, 1123.15) > rk(1.0e8, 1123.15)   # higher P -> denser
        @test rk(3.0e7, 1173.15) > 0                  # positive over the calibration box
        # a mush carrying a freshly injected sill runs above the 1173 K calibration
        # ceiling; the fit stays positive and monotone there
        @test rk(2.0e8, 1473.15) > 0
        @test rk(2.0e8, 1473.15) < rk(2.0e8, 1173.15)
        # the EOS the chamber uses is the one the gas-density diagnostic reports
        ep = QMagma.EruptionParams(m_w = 0.05)
        _, _, ρ_g, _ = QMagma.water_gas_partition(2.0e8, 1100.0, 0.7, ep)
        @test ρ_g ≈ rk(2.0e8, 1100.0)

        # Gas density is a phase parameterization, not a hard-coded EOS.
        fixed = QMagma.EruptionParams(
            m_w = 0.05, ρ_gas = ConstantDensity(ρ = 123kg / m^3)
        )
        _, X_g, ρ_fixed, _ = QMagma.water_gas_partition(5.0e7, 1100.0, 0.7, fixed)
        @test X_g > 0
        @test ρ_fixed == 123.0
    end

    @testset "Liu 2005 H₂O solubility (item 4)" begin
        # mass-fraction form (reference exsolve_silicic.m includes the 1e-2 wt%->fraction factor)
        meq(Pmpa) = 1.0e-2 * ((354.94 * sqrt(Pmpa) + 9.623 * Pmpa - 1.5223 * Pmpa^1.5) / 1200.0 + 1.2439e-3 * Pmpa^1.5)
        @test 0.03 < meq(200.0) < 0.07     # ~5 wt% at 200 MPa / 1200 K
        @test meq(400.0) > meq(200.0)      # more soluble at higher P (monotone in range)
        # the saturation the chamber uses is that same law
        ep = QMagma.EruptionParams()
        _, _, _, m_eq = QMagma.water_gas_partition(2.0e8, 1200.0, 0.7, ep)
        @test m_eq ≈ meq(200.0)
        ρ, _ = QMagma.mixture_density(2.0e8, 1100.0, 1.0, ep)
        @test isfinite(ρ) && ρ > 0
    end

    @testset "shell relaxation viscosity η_r (D&H A.18)" begin
        ep = QMagma.EruptionParams()
        # reproduces the reference implementation's crust viscosity for D&H's own chamber:
        # mush at 1200 K, far field at 500 K, shell out to 11a. The 1.5 % offset is
        # B_gas: 8.314 here against the 8.31 the reference rounds to.
        @test QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0) ≈ 1.885e19 rtol = 2.0e-2
        # A.13 and the r⁻⁴ kernel are both functions of r/a, so the chamber radius cancels:
        # η_r is fixed by the two temperatures alone
        @test QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0; shell_ratio = 11.0) ==
            QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0)
        # the far field dominates: 100 K colder crust stiffens the shell by orders of
        # magnitude, while the same change at the wall barely moves it
        far = QMagma.crustal_relaxation_viscosity(ep, 1200.0, 400.0) /
            QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0)
        wall = QMagma.crustal_relaxation_viscosity(ep, 1100.0, 500.0) /
            QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0)
        @test far > 1.0e3
        @test 0.5 < wall < 2
        # quadrature is converged at the default resolution
        @test QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0; n = 512) ≈
            QMagma.crustal_relaxation_viscosity(ep, 1200.0, 500.0) rtol = 1.0e-4
        @test 1.0e17 <= QMagma.crustal_relaxation_viscosity(ep, 1200.0, 800.0) <= 1.0e24
        @test QMagma.crustal_relaxation_viscosity(ep, 1200.0, 300.0) == 1.0e24
        @test_throws "hotter than the mush it encloses" QMagma.crustal_relaxation_viscosity(ep, 800.0, 900.0)
        # the local Arrhenius law is not itself a relaxation viscosity
        @test QMagma.crustal_viscosity(ep, 500.0) > QMagma.crustal_viscosity(ep, 650.0)
        @test QMagma.crustal_viscosity(ep, 1200.0) < 1.0e15
    end

    @testset "H₂O speciation diagnostics on EruptionState" begin
        ep = QMagma.EruptionParams(m_w = 0.05)
        # partition splits total water: dissolved + exsolved == m_w when saturated (X_g>0)
        m_diss, X_g, ρ_g, m_eq = QMagma.water_gas_partition(2.0e8, 1100.0, 0.7, ep)
        ρm7 = QMagma.melt_density(ep, 2.0e8, 1100.0); ρx7 = QMagma.crystal_density(ep, 2.0e8, 1100.0)
        @test m_diss ≈ m_eq * (0.7ρm7 / (0.7ρm7 + 0.3ρx7))
        @test isapprox(m_diss + X_g, ep.m_w; atol = 1.0e-12)   # water conserved while gas-saturated
        @test ρ_g > 0
        # crystallizing (ϕ_melt↓) exsolves more gas -> X_g rises (second boiling)
        _, X_wet, _, _ = QMagma.water_gas_partition(2.0e8, 1100.0, 0.8, ep)
        _, X_dry, _, _ = QMagma.water_gas_partition(2.0e8, 1100.0, 0.4, ep)
        @test X_dry > X_wet
        # m_eq is water per melt MASS, so it is weighted by the melt mass fraction. ϕ is
        # volumetric, and with ρ_melt < ρ_x the two differ; weighting by ϕ directly
        # overstates dissolved water and so understates the exsolved gas.
        for ϕ_m in (0.5, 0.6, 0.8)
            m_diss, X_g, _, m_eq = QMagma.water_gas_partition(2.0e8, 1100.0, ϕ_m, ep)
            ρm = QMagma.melt_density(ep, 2.0e8, 1100.0); ρx = QMagma.crystal_density(ep, 2.0e8, 1100.0)
            x_mass = ϕ_m * ρm / (ϕ_m * ρm + (1 - ϕ_m) * ρx)
            @test x_mass < ϕ_m                       # melt is the lighter phase
            @test m_diss ≈ min(m_eq * x_mass, ep.m_w)
            @test X_g ≈ ep.m_w - m_diss
            @test X_g > ep.m_w - m_eq * ϕ_m            # more gas than the volume weighting gives
        end
        # the endpoints carry no density contrast and must be untouched by the conversion
        @test QMagma.water_gas_partition(2.0e8, 1100.0, 0.0, ep)[1] == 0.0
        @test QMagma.water_gas_partition(2.0e8, 1100.0, 1.0, ep)[1] ≈
            min(QMagma.water_gas_partition(2.0e8, 1100.0, 1.0, ep)[4], ep.m_w)
        # A melt that dissolves all of m_w has no gas phase: X_g and ρ_g reach zero and the
        # mixture is the condensed density, with no EOS evaluation and no discontinuity.
        md, Xg, ρg, meq = QMagma.water_gas_partition(1.0e9, 1500.0, 1.0, ep)
        @test md == ep.m_w && Xg == 0.0 && ρg == 0.0
        @test meq > ep.m_w
        ρdeep, ϕgdeep = QMagma.mixture_density(1.0e9, 1500.0, 0.7, ep)
        @test ρdeep ≈ QMagma.condensed_density(ep, 1.0e9, 1500.0, 0.7)
        @test ϕgdeep == 0.0
        # step_overpressure! must populate the diagnostics on the state. Use a shallow chamber
        # (~5 km, ~130 MPa) so the melt is water-saturated and X_g > 0; a deep chamber at high
        # lithostatic P keeps all water dissolved (X_g = 0), which is correct but trivial.
        st = QMagma.EruptionState(); QMagma.init_eruption!(st, 1.324e8)
        QMagma.step_overpressure!(
            st, ep, 1100.0, 0.7, 1000.0, 1.0e-9, 1.0e10;
            z_centroid = -5.0e3
        )
        @test st.m_diss > 0 && st.ρ_gas > 0 && st.X_g > 0
        @test st.ρ_magma ≈ first(QMagma.mixture_density(st.P, 1100.0, 0.7, ep))
        @test st.ϕ_mush == 0.7 && st.η_r == ep.η_r   # mush ϕ and wall viscosity recorded too
    end

    @testset "explicit chamber mass and H₂O inventories" begin
        ep = QMagma.EruptionParams(m_w = 0.05, ΔP_crit = 1.0e9)
        st = QMagma.EruptionState(P_lith = 1.324e8, P = 1.324e8)
        QMagma.step_overpressure!(
            st, ep, 1100.0, 0.7, 1000.0, 1.0e-9, 1.0e10;
            z_centroid = -5.0e3
        )
        @test st.M_initial == st.M ≈ st.ρ_magma * st.V
        @test st.M_H2O ≈ ep.m_w * st.M

        QMagma.step_overpressure!(
            st, ep, 1100.0, 0.7, 1000.0, 1.0e-9, 1.0e10;
            z_centroid = -5.0e3
        )
        @test st.M_in > 0 && st.M_out == 0.0
        @test st.M ≈ st.M_initial + st.M_in - st.M_out + st.M_boundary
        @test st.M_H2O / st.M ≈ ep.m_w
        @test abs(st.mass_residual) <= 5.0e-3 * st.M

        # A recharge pulse may change the inventory by more than the pressure integrator's
        # local truncation scale. The EOS projection must close that expected mismatch.
        pulse = QMagma.EruptionState(P_lith = 1.324e8, P = 1.324e8)
        dry = QMagma.EruptionParams(m_w = 0.0, ΔP_crit = 1.0e12)
        QMagma.step_overpressure!(
            pulse, dry, 1100.0, 0.7, 500.0, 0.0, 1.0;
            z_centroid = -5.0e3
        )
        QMagma.step_overpressure!(
            pulse, dry, 1100.0, 0.7, 500.0, 1.0e-8, 1.0e10;
            z_centroid = -5.0e3
        )
        @test pulse.M_in > 0.1pulse.M_initial
        @test pulse.M ≈ pulse.M_initial + pulse.M_in
        @test abs(pulse.mass_residual) <= 64eps(pulse.M)

        # Growth of the grid-defined mush is material crossing the chamber control-volume
        # boundary. It belongs in the ledger, but is not magmatic recharge.
        M_in = st.M_in
        QMagma.step_overpressure!(
            st, ep, 1100.0, 0.7, 1100.0, 0.0, 1.0;
            z_centroid = -5.0e3
        )
        @test st.M_boundary > 0
        @test st.M_in == M_in
        @test st.M ≈ st.M_initial + st.M_in - st.M_out + st.M_boundary

        # A deliberately corrupted independent inventory must fail instead of allowing
        # pressure (ρV) and the mass ODE to evolve as two incompatible chambers.
        bad = QMagma.EruptionState(P_lith = 1.324e8, P = 1.324e8)
        QMagma.step_overpressure!(
            bad, ep, 1100.0, 0.7, 1000.0, 0.0, 1.0;
            z_centroid = -5.0e3
        )
        bad.M *= 1.1
        bad.M_H2O *= 1.1
        @test_throws "chamber mass closure failed" QMagma.step_overpressure!(
            bad, ep, 1100.0, 0.7, 1000.0, 0.0, 1.0;
            z_centroid = -5.0e3
        )
    end

    @testset "exact magma compressibility 1/β_m" begin
        # differentiating through the solubility law and gas EOS must reproduce the
        # finite-difference value to the FD step's own truncation error
        ep = QMagma.EruptionParams(m_w = 0.05)
        P, T_K, ϕ_m = 1.324e8, 1173.15, 0.7
        ρ, _ = QMagma.mixture_density(P, T_K, ϕ_m, ep)
        dP = 1.0e-4 * P
        ρp, _ = QMagma.mixture_density(P + dP, T_K, ϕ_m, ep)
        inv_βm_fd = (ρp - ρ) / (ρ * dP)
        st = QMagma.EruptionState(P_lith = P, P = P)
        QMagma.step_overpressure!(st, ep, T_K, ϕ_m, 1000.0, 0.0, 1.0e10; z_centroid = -5.0e3)
        QMagma.step_overpressure!(st, ep, T_K, ϕ_m, 1000.0, 0.0, 1.0e10; z_centroid = -5.0e3)
        @test isapprox(st.inv_βm, inv_βm_fd; rtol = 1.0e-3)
        @test st.inv_βm > QMagma.host_compliance(ep, st.V)   # gas-bearing magma is the softer of the two
        # A dry chamber has no gas to compress, but its melt and crystals are compressible
        # density laws, so it stays finite and positive rather than collapsing to zero —
        # this is what keeps 1/β_m continuous across the water-saturation crossing.
        dry = QMagma.EruptionParams(m_w = 0.0)
        st_dry = QMagma.EruptionState(P_lith = P, P = P)
        QMagma.step_overpressure!(st_dry, dry, T_K, ϕ_m, 1000.0, 0.0, 1.0e10; z_centroid = -5.0e3)
        QMagma.step_overpressure!(st_dry, dry, T_K, ϕ_m, 1000.0, 0.0, 1.0e10; z_centroid = -5.0e3)
        @test 0 < st_dry.inv_βm < st.inv_βm
        # crossing water saturation is now a finite step, not a drop to exactly zero
        wet = QMagma.EruptionParams(m_w = 0.05)
        βm(ϕ) = (
            ρ = first(QMagma.mixture_density(3.2e8, 1123.15, ϕ, wet));
            QMagma.ForwardDiff.derivative(
                p -> first(QMagma.mixture_density(p, 1123.15, ϕ, wet)), 3.2e8
            ) / ρ
        )
        @test last(QMagma.mixture_density(3.2e8, 1123.15, 0.7, wet)) == 0.0   # undersaturated
        @test βm(0.7) > 0
        @test βm(0.66) / βm(0.7) < 10
    end

    @testset "crustal column and dike propagation" begin
        ep = QMagma.EruptionParams(σ_barrier = 10.0e6)
        z = collect(-20.0e3:100.0:0.0)
        T = 400.0 .- z ./ 1.0e3 .* 20.0
        P5 = QMagma.lithostatic_pressure(ep, T, z, -5.0e3)
        P10 = QMagma.lithostatic_pressure(ep, T, z, -10.0e3)

        # the column integrates a real ρ(P,T): monotone with depth, zero at the surface,
        # and within a few percent of a uniform 2700 kg/m³ crust
        @test QMagma.lithostatic_pressure(ep, T, z, 0.0) == 0.0
        @test P10 > P5 > 0
        @test isapprox(P5, 2700.0 * ep.g * 5.0e3; rtol = 0.05)
        # thermal expansion dominates compression down a geotherm, so a hotter column is
        # lighter; the effect is small next to the magma-crust density contrast
        P5_hot = QMagma.lithostatic_pressure(ep, T .+ 200.0, z, -5.0e3)
        @test P5_hot < P5
        @test (P5 - P5_hot) / P5 < 0.02
        @test_throws "outside the column" QMagma.lithostatic_pressure(ep, T, z, -30.0e3)
        @test_throws DimensionMismatch QMagma.lithostatic_pressure(ep, T[1:(end - 1)], z, -5.0e3)

        # buoyancy alone carries a light magma up from 5 km; magma as dense as the column
        # it sits in needs the full barrier as overpressure
        η_b = QMagma.magma_viscosity(ep, 1473.15)          # basalt melt, ~1200 °C
        ρ_neutral = P5 / (ep.g * 5.0e3)
        @test QMagma.dike_ascends(0.0, P5, 2300.0, η_b, -5.0e3, ep)
        @test !QMagma.dike_ascends(0.0, P5, ρ_neutral, η_b, -5.0e3, ep)
        @test QMagma.dike_ascends(ep.σ_barrier, P5, ρ_neutral, η_b, -5.0e3, ep)
        # exsolved gas lightens the magma and eases ascent at fixed overpressure
        wet = QMagma.EruptionParams(m_w = 0.05, σ_barrier = 10.0e6)
        ρ_wet, ϕ_g = QMagma.mixture_density(P5, 1173.15, 0.7, wet)
        ρ_dry, _ = QMagma.mixture_density(
            P5, 1173.15, 0.7,
            QMagma.EruptionParams(m_w = 0.0)
        )
        @test ϕ_g > 0 && ρ_wet < ρ_dry
        @test QMagma.dike_ascends(0.0, P5, ρ_wet, η_b, -5.0e3, ep) &&
            !QMagma.dike_ascends(
            0.0, P5, ρ_dry, η_b, -5.0e3,
            QMagma.EruptionParams(σ_barrier = 1.6e7)
        )
        @test_throws DomainError QMagma.dike_ascends(0.0, P5, -1.0, η_b, -5.0e3, ep)

    end

    @testset "dike freezing sets the reachable depth" begin
        ep = QMagma.EruptionParams(σ_barrier = 0.0)
        rhyolite = LinearMeltViscosity(A = -8.159, B = 2.405e4K, T0 = -430.9606K, η0 = 1Pas)
        ep_rhy = QMagma.EruptionParams(σ_barrier = 0.0, melt_viscosity = rhyolite)

        # melt viscosity is the control: rhyolite is orders stiffer than basalt at the
        # same temperature, and the reachable length carries that straight through
        η_bas = QMagma.magma_viscosity(ep, 1473.15)
        η_rhy = QMagma.magma_viscosity(ep_rhy, 1173.15)
        @test η_rhy > 1.0e3 * η_bas
        Δρg, ΔP = 300.0 * ep.g, 20.0e6
        L_bas = QMagma.max_ascent_length(ΔP, Δρg, η_bas, ep)
        L_rhy = QMagma.max_ascent_length(ΔP, Δρg, η_rhy, ep)
        @test L_bas > 100.0e3            # basalt clears any crustal depth
        @test L_rhy < 10.0e3             # a rhyolite dike freezes in the upper crust

        # The two drivers put a dike in different regimes, with different scalings, and
        # the quadratic interpolates between them. Buoyancy-driven (ΔP = 0): L = Δρg·c,
        # so L ∝ 1/η and ∝ w⁴ — this is where a basalt dike lives.
        wide = QMagma.EruptionParams(
            σ_barrier = 0.0, melt_viscosity = rhyolite, w_dike = 2 * ep.w_dike
        )
        Lb_bas = QMagma.max_ascent_length(0.0, Δρg, η_bas, ep)
        Lb_rhy = QMagma.max_ascent_length(0.0, Δρg, η_rhy, ep)
        @test Lb_rhy ≈ Δρg * ep.w_dike^4 / (3 * η_rhy * ep.κ_magma)
        @test Lb_bas / Lb_rhy ≈ η_rhy / η_bas
        @test QMagma.max_ascent_length(0.0, Δρg, η_rhy, wide) / Lb_rhy ≈ 16
        # Overpressure-driven (Δρg = 0): L = √(ΔP·c), so L ∝ 1/√η and ∝ w². A stiff
        # silicic dike is pushed here, which is why its reach is less viscosity-sensitive
        # than the buoyancy limit suggests.
        Lp_rhy = QMagma.max_ascent_length(ΔP, 0.0, η_rhy, ep)
        @test Lp_rhy ≈ sqrt(ΔP * ep.w_dike^4 / (3 * η_rhy * ep.κ_magma))
        @test QMagma.max_ascent_length(ΔP, 0.0, η_rhy, wide) / Lp_rhy ≈ 4
        @test Lp_rhy > Lb_rhy                       # 20 MPa outweighs rhyolite buoyancy
        @test Lb_bas > QMagma.max_ascent_length(ΔP, 0.0, η_bas, ep)  # and not for basalt
        # no drive at all, no ascent
        @test QMagma.max_ascent_length(0.0, 0.0, η_rhy, ep) == 0.0
        @test_throws DomainError QMagma.max_ascent_length(ΔP, Δρg, -1.0, ep)

        # the criterion the triggers see: a viscous deep chamber is stalled by freezing
        # even though its buoyancy drive exceeds a shallow one's
        z = collect(-30.0e3:100.0:0.0)
        T = 400.0 .- z ./ 1.0e3 .* 20.0
        P15 = QMagma.lithostatic_pressure(ep_rhy, T, z, -15.0e3)
        P5 = QMagma.lithostatic_pressure(ep_rhy, T, z, -5.0e3)
        ρ_m = 2400.0
        @test P15 - ρ_m * ep.g * 15.0e3 > P5 - ρ_m * ep.g * 5.0e3      # deep buoyancy drive is larger
        @test !QMagma.dike_ascends(ΔP, P15, ρ_m, η_rhy, -15.0e3, ep_rhy)
        @test QMagma.dike_ascends(ΔP, P5, ρ_m, η_bas, -5.0e3, ep_rhy)
    end

    @testset "sill temperature against the melting-law liquidus" begin
        rhyolite = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0)
        basalt = MeltingParam_Smooth3rdOrder()
        T_liq_rhy = QMagma.liquidus_temperature(rhyolite)
        @test compute_meltfraction(rhyolite, (; T = T_liq_rhy)) ≈ 1
        @test compute_meltfraction(rhyolite, (; T = T_liq_rhy - 5.0)) < 1
        @test QMagma.liquidus_temperature(basalt) > T_liq_rhy
        # a 1200 °C sill is superheated for rhyolite and must be rejected at setup, naming
        # both temperatures; the same sill is subliquidus for basalt
        @test_throws "sill temperature 1200.0 °C exceeds" QMagma.check_sill_temperature(
            rhyolite, 1200.0
        )
        @test QMagma.check_sill_temperature(basalt, 1200.0) ≈ QMagma.liquidus_temperature(basalt)
        @test_throws "no liquidus" QMagma.liquidus_temperature(rhyolite; T_max = 1000.0)
    end

    @testset "step_overpressure! nsub does not overflow Int64" begin
        # a soft (floored) η_r + big ΔP + large Δt makes the raw sub-step count ≫ typemax(Int64);
        # the clamp must happen in float space so ceil(Int, …) never sees the overflowing value
        ep = QMagma.EruptionParams(η_r = 1.0e17, ΔP_crit = 20.0e6)
        st = QMagma.EruptionState(); QMagma.init_eruption!(st, 3.44e8)
        st.P = st.P_lith + 100 * ep.ΔP_crit          # far over threshold -> huge dPdt0
        QMagma.step_overpressure!(
            st, ep, 1200.0 + 273.15, 0.8, 1000.0, 1.0e-9, 1.0e11;
            z_centroid = -13.0e3
        )  # init
        # The sub-step count reaches its cap without overflowing Int64, then the EOS
        # projection closes the pressure and inventory descriptions of the chamber.
        QMagma.step_overpressure!(
            st, ep, 1200.0 + 273.15, 0.8, 1000.0, 1.0e-9, 1.0e11;
            z_centroid = -13.0e3
        )
        @test isfinite(st.P) && st.P > 0
        @test abs(st.mass_residual) <= 64eps(st.M)
    end

    @testset "step_overpressure! is stable when Δt ≫ the relaxation time" begin
        # a hot wall floors η_r, making the relaxation time constant η_r·(1/β_r+1/β_m) tens
        # of times shorter than the thermal Δt. The overpressure must relax toward S·η_r
        # from either side without ever passing through negative absolute pressure.
        ep = QMagma.EruptionParams(η_r = 1.0e17)
        P_lith = 3.216e8                       # ~12.4 km of crust
        Δt = 100SecYear
        for ΔP0 in (1.0e5, 2.4e6, 25.0e6), ȧ in (0.0, 1.0e-11)
            st = QMagma.EruptionState(); QMagma.init_eruption!(st, P_lith)
            st.P = P_lith + ΔP0
            QMagma.step_overpressure!(
                st, ep, 900.0 + 273.15, 0.6, 2000.0, ȧ, Δt;
                z_centroid = -12.37e3
            )           # first call inits
            for _ in 1:20
                QMagma.step_overpressure!(
                    st, ep, 900.0 + 273.15, 0.6, 2000.0, ȧ, Δt;
                    z_centroid = -12.37e3
                )
                @test st.P > 0
            end
            # Δt/τ ≈ 66, so twenty steps land on the equilibrium overpressure S·η_r. The
            # source term carries the chamber's own volume, which the wall relaxation has
            # been moving, so the equilibrium is set by st.V rather than the seeded extent.
            S = ȧ * QMagma.melt_density(ep, st.P, 900.0 + 273.15) / (st.ρ_magma * st.V)
            @test isapprox(st.P - st.P_lith, S * ep.η_r; rtol = 1.0e-6, atol = 1.0)
        end
    end

    @testset "D&H sub-grid withdrawals accumulate before booking" begin
        st = QMagma.EruptionState()
        @test QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time = 1.0) == 0.0
        @test QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time = 2.0) == 0.0
        h_realized = QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time = 3.0)
        @test h_realized == 225.0
        @test st.h_pending == 225.0
        @test st.pending_since == 1.0
        QMagma.commit_pending_withdrawal!(st, h_realized)
        @test st.h_pending == 0.0
        @test isnan(st.pending_since)

        st.h_pending = 300.0
        st.pending_since = 4.0
        @test QMagma.pending_withdrawal!(st, 0.0, 250.0, 100.0) == 250.0
        QMagma.commit_pending_withdrawal!(st, 250.0)
        @test st.h_pending == 50.0
        @test_throws "h_realized must lie between zero and the pending withdrawal" QMagma.commit_pending_withdrawal!(
            st, 51.0
        )
    end

    @testset "V2: analytic overpressure relaxation" begin
        # constant properties, no exsolution (m_w=0) and no recharge (ȧ=0): the master ODE
        # reduces to dΔP/dt = -ΔP/(η_r/β_r), i.e. exponential decay with τ = η_r/β_r.
        # Incompressible condensed phases are what isolates the integrator here: with the
        # compressible defaults 1/β_m stays finite and τ picks up the magma's own
        # compliance, which is correct physics but no longer an analytic one-liner.
        # The chamber is seeded at V=100 m against R_sill, so ε is fixed for the whole run
        # and 1/β_r with it — that is what leaves a single analytic exponential to check.
        ep = QMagma.EruptionParams(
            m_w = 0.0, μ_shear = 1.0e10, R_sill = 50.0, η_r = 1.0e18,
            ρ_melt = ConstantDensity(ρ = 2400kg / m^3), ρ_x = ConstantDensity(ρ = 2700kg / m^3)
        )
        st = QMagma.EruptionState(P_lith = 2.0e8, P = 2.0e8)
        τ = ep.η_r * QMagma.host_compliance(ep, 100.0)
        ΔP0 = 15.0e6; st.P = st.P_lith + ΔP0
        Δt = τ / 2000; nstep = 2000
        QMagma.step_overpressure!(
            st, ep, 1200.0 + 273.15, 0.7, 100.0, 0.0, Δt;
            z_centroid = -13.0e3
        )  # first call inits
        for _ in 1:nstep
            QMagma.step_overpressure!(
                st, ep, 1200.0 + 273.15, 0.7, 100.0, 0.0, Δt;
                z_centroid = -13.0e3
            )
        end
        ΔP_num = st.P - st.P_lith
        ΔP_exact = ΔP0 * exp(-nstep * Δt / τ)                # t = nstep·Δt = τ
        @test isapprox(ΔP_num, ΔP_exact; rtol = 1.0e-9)    # each step integrates the decay exactly
    end

    @testset "column enthalpy drift across eruption closures" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = 201, L = 40.0e3, Geotherm = 20.0,
            Ttop = 0.0, Tbot = 800.0, Δt = 200SecYear
        )
        # a hot mobile mush embedded in the geotherm
        T[(-17.0e3 .<= z .<= -13.0e3)] .= 1000.0
        rocks = zeros(length(z))
        MatParam = Params.MatParam
        H0 = QMagma.column_enthalpy(T, z, MatParam, Params.Phases)

        Tc, _ = QMagma.erupt_melt!(T, rocks, z; Erupt_z0 = -15.0e3, Erupt_thick = 800.0, method = :caldera)
        Th, _ = QMagma.erupt_melt!(T, rocks, z; Erupt_z0 = -15.0e3, Erupt_thick = 800.0, method = :hybrid)
        Hc = QMagma.column_enthalpy(Tc, z, MatParam, Params.Phases)
        Hh = QMagma.column_enthalpy(Th, z, MatParam, Params.Phases)

        # both closures remove heat. :caldera translates the roof rigidly, so the debit is
        # the band's own content; :hybrid transports T intensively, and the elastically
        # raised floor re-covers part of the vent at its own temperature, so its debit
        # differs by however non-uniform the near-vent column is.
        @test H0 - Hc > 0
        @test H0 - Hh > 0
        # a hot band in a cooler column: the rigid closure removes the band outright
        @test H0 - Hc > H0 - Hh
    end

    @testset "erupted enthalpy is bulk magma, not liquid" begin
        # The eruption withdraws the multi-phase mixture, so the booked debit must use the
        # integrand column_enthalpy integrates, ρ(cₚT + Lϕ), averaged over the band by width.
        # Weighting by melt content instead charges the whole parcel the latent heat only
        # its liquid carries, and the error grows as the mush crystallizes - so sweep ϕ̄.
        z = collect(-40.0e3:20.0:0.0)
        z_lo, z_hi = -17.0e3, -13.0e3
        band = z_lo .<= z .<= z_hi
        h = 400.0                                   # withdrawn from the middle of the band
        Params, = QMagma.init_model(
            nz = length(z), L = 40.0e3, Geotherm = 20.0,
            Ttop = 0.0, Tbot = 800.0, Δt = 200SecYear
        )
        for T_band in (1000.0, 800.0, 760.0)
            T = 800.0 .- (z .+ 40.0e3) ./ 1.0e3 .* 20.0
            T[band] .= T_band
            ϕ = similar(T)
            compute_meltfraction!(ϕ, Params.MatParam, Params.Phases, (T = T .+ 273.15,))
            ϕ_bar = QMagma.melt_thickness(ϕ, z, z_lo, z_hi) / (z_hi - z_lo)
            tracers = [
                QMagma.Tracer(zi, T_band, ϕ[i], 1, [0.0], [T_band])
                    for (i, zi) in pairs(z) if band[i]
            ]
            _, _, _, event = QMagma.realize_eruption!(
                MersenneTwister(1), T, zeros(length(z)), tracers, ϕ, z,
                Params.MatParam, Params.Phases;
                realization_time = 1.0, h_requested = h, z_lo, z_hi,
                trigger = :test, closure = :caldera, eligible_phase = nothing
            )
            # the liquid content of the withdrawal tracks the band's crystallinity...
            @test event.requested == h
            @test event.melt_requested ≈ h * ϕ_bar
            # ...while the enthalpy residual does not. :caldera deletes whole control
            # volumes, so it removes one cell more than the h it was asked for; that
            # quantization is the entire residual, at every ϕ̄. A melt-weighted debit would
            # instead drift with crystallinity, reaching ~25% at the lowest ϕ̄ here.
            @test event.enthalpy_residual / event.erupted_enthalpy ≈ -(z[2] - z[1]) / h
        end
    end

    @testset "cumulative enthalpy budget accounting" begin
        z = collect(-2.0:1.0:0.0)
        T = [2.0, 1.0, 0.0]
        @test QMagma.conductive_boundary_energy(T, [3.0, 3.0], z, 5.0) == 0.0
        @test QMagma.source_energy(fill(2.0, 3), z, 5.0) == 10.0

        budget = QMagma.EnthalpyBudget(100.0)
        QMagma.update_enthalpy_budget!(
            budget, 118.0;
            boundary = 2.0, injected = 10.0, source = 10.0, erupted = 4.0
        )
        snapshot = QMagma.enthalpy_budget_snapshot(budget)
        @test snapshot.storage_change == 18.0
        @test snapshot.residual == 0.0
        QMagma.update_enthalpy_budget!(budget, 120.0; boundary = 2.0)
        @test budget.boundary == 4.0
        @test budget.residual == 0.0
    end

    @testset "magma and melt budget accounting" begin
        budget = QMagma.MassBudget(0.0, 100.0)

        # 200 m of magma in, 10 m of it displaced off the grid, all of it stored as grey:
        # the magma-volume budget closes. Only 60 m of it is still melt; the unresolved
        # melt-content residual combines crystallization, host melting, and boundary
        # transport, and is not a measurement of any one of them.
        QMagma.update_mass_budget!(budget, 190.0, 160.0; injected = 200.0, withdrawn = 10.0)
        snapshot = QMagma.mass_budget_snapshot(budget)
        @test snapshot.magma_change == 190.0
        @test snapshot.melt_change == 60.0
        @test snapshot.residual == 0.0
        @test snapshot.melt_residual == 140.0

        # an eruption debits both accounts, each with its own withdrawal: grey leaves with
        # the closure, melt leaves as the booked thickness
        QMagma.update_mass_budget!(budget, 150.0, 130.0; withdrawn = 40.0, erupted = 30.0)
        @test budget.injected == 200.0
        @test budget.withdrawn == 50.0
        @test budget.residual == 0.0
        @test budget.melt_residual == 200.0 - 30.0 - 30.0

        # grey that vanishes without being withdrawn is a transport leak, and shows up
        QMagma.update_mass_budget!(budget, 140.0, 130.0)
        @test budget.residual == 10.0

        @test_throws "mass-budget terms must be finite" QMagma.update_mass_budget!(budget, NaN, 1.0)
    end

    @testset "injection-only mass budgets over 300 kyr" begin
        # With no eruptions the magma-volume budget closes to the discretization of
        # emplacing a sill on a fixed grid, while the melt residual must stay strictly
        # positive: it combines crystallization, host-rock melting and boundary transport
        # (see MassBudget), and cooling injections make crystallization dominate.
        H, γ, Ttop = 40.0, 20.0, 0.0
        Δt = 200SecYear
        nz = 201
        Tsill, Sillthick, Sill_int_yr = 1200.0, 100.0, 500.0
        Silltop, Sillbot = 10.0, 20.0
        nt = 1500                                   # 300 kyr

        Params, BC, N, Δ, T, z = QMagma.init_model(
            nz = nz, L = H * 1.0e3, Geotherm = γ, Ttop = Ttop,
            Tbot = Ttop + H * γ, Δt = Δt
        )
        MatParam = Params.MatParam
        Params.Told .= T
        Δz = Δ[1]
        rocks = zero(T)

        Jac, colors = thermal_jacobian_workspace(nz)
        F = zero(T)

        compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
        budget = QMagma.MassBudget(
            QMagma.integrated_content(rocks, z),
            QMagma.melt_thickness(Params.ϕ, z, z[1], z[end])
        )

        ȧ = Sillthick / Sill_int_yr / SecYear
        time, A_inj = 0.0, 0.0
        rng = MersenneTwister(1234)
        for _ in 1:nt
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            T, = QMagma.nonlinear_solution(
                F, T, Jac, colors; Δ = Δ, N = N, BC = BC, Params = Params,
                MatParam = MatParam, verbose = false
            )
            injected_step, withdrawn_step = 0.0, 0.0
            for _ in 1:n_injections
                Sill_z0 = rand(rng, (-Sillbot * 1.0e3):1.0:(-Silltop * 1.0e3))
                T, rocks, h_out = QMagma.insert_sill(
                    T, rocks, z; Sill_thick = Sillthick,
                    Sill_z0 = Sill_z0, Sill_T = Tsill
                )
                injected_step += Sillthick
                withdrawn_step += h_out
                Params.Told .= T
            end
            Params.Told .= T
            compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
            QMagma.update_mass_budget!(
                budget, QMagma.integrated_content(rocks, z),
                QMagma.melt_thickness(Params.ϕ, z, z[1], z[end]);
                injected = injected_step, withdrawn = withdrawn_step
            )
            time += Δt
        end

        @test budget.injected ≈ nt * Δt * ȧ
        @test budget.injected ≈ 600 * Sillthick          # 300 kyr / 500 yr
        # The elastic opening reaches the free surface, so a trace of magma leaves through
        # it - centimetres against the 60 km injected over 600 events.
        @test budget.withdrawn ≈ 0.0 atol = 1.0e-1
        @test budget.magma ≈ budget.injected atol = 1.0e-1
        @test budget.residual ≈ 0.0 atol = 1.0e-6
        @test budget.melt_residual > 0
        @test budget.melt_residual < budget.injected
        @test budget.melt > 0
    end

    @testset "magma_heat_input" begin
        Params, = QMagma.init_model(nz = 11, L = 10.0e3, Ttop = 0.0, Tbot = 800.0, Δt = SecYear)
        MatParam = Params.MatParam

        E = QMagma.magma_heat_input(600.0, 1200.0, 100.0, MatParam)
        @test E > 0
        # Sensible + latent heat both scale linearly with the injected thickness.
        @test QMagma.magma_heat_input(600.0, 1200.0, 200.0, MatParam) ≈ 2E
        @test QMagma.magma_heat_input(600.0, 1200.0, 0.0, MatParam) == 0.0
        # A colder host absorbs more heat from the same sill.
        @test QMagma.magma_heat_input(400.0, 1200.0, 100.0, MatParam) > E
        @test_throws "h must be nonnegative" QMagma.magma_heat_input(600.0, 1200.0, -1.0, MatParam)
    end

    @testset "erupt_displacement closure methods" begin
        Erupt_z0, Erupt_thick = -5.0e3, 1000.0
        half = Erupt_thick / 2

        # :hybrid collapses the band onto the vent
        @test QMagma.erupt_displacement(Erupt_z0, Erupt_z0, Erupt_thick; method = :hybrid) == Erupt_z0

        # below the band the elastic floor rises toward the vent by less than a half-thickness
        below = Erupt_z0 - 3.0e3
        z_hyb = QMagma.erupt_displacement(below, Erupt_z0, Erupt_thick; method = :hybrid)
        @test below < z_hyb < below + half
        # :caldera leaves the floor where it is
        @test QMagma.erupt_displacement(below, Erupt_z0, Erupt_thick; method = :caldera) == below

        # above the band :hybrid subsides toward, but never past, the rigid :caldera drop
        above = Erupt_z0 + 3.0e3
        z_hyb_up = QMagma.erupt_displacement(above, Erupt_z0, Erupt_thick; method = :hybrid)
        z_cal_up = QMagma.erupt_displacement(above, Erupt_z0, Erupt_thick; method = :caldera)
        @test z_cal_up == above - Erupt_thick
        @test z_cal_up <= z_hyb_up < above

        @test_throws "unknown eruption collapse method" QMagma.erupt_displacement(
            above, Erupt_z0, Erupt_thick; method = :nonsense
        )
    end

    @testset "startup banner" begin
        io = IOBuffer()
        QMagma._print_banner(io)
        banner = String(take!(io))
        @test occursin("Version: $(pkgversion(QMagma))", banner)
        @test occursin("\e[38;5;196m", banner)          # 256-color codes, not truecolor
        @test QMagma.__init__(devnull) === nothing
    end

    # GUI internals belong to the Makie extension. Without that weak dependency the symbols
    # do not exist at all; where a backend is loaded, test through the extension module
    # rather than pretending they belong to QMagma itself.
    @testset "GUI layout and wiring" begin
        ext = Base.get_extension(QMagma, :QMagmaMakieExt)
        if ext === nothing
            @test_throws "sill_intrusion_1D requires GLMakie" QMagma.sill_intrusion_1D()
        else
            # The app itself needs GLFW, which only GLMakie provides, so under any other
            # backend it must refuse before opening a window.
            @test_throws "requires the GLMakie backend" ext.sill_intrusion_1D()

            ui = ext.build_layout((1200, 800))
            @test ext.get_valuebox(ui.Δz_box) == 20
            @test ui.menu_trigger.selection[] == "None"
            @test ui.last_matparam[] === nothing
            @test ext.wire_buttons!(ui) === nothing
            @test ext.wire_simulation!(ui) === nothing

            # Zircon spectra are per injection model: a comparison run must hand the
            # zircon button both populations, and never mix them up.
            sill_res, sill_cargo = [QMagma.Tracer(0.0, 0.0, 0.0, 1, [0.0], [900.0])], QMagma.Tracer[]
            Q_res, Q_cargo = QMagma.Tracer[], [QMagma.Tracer(0.0, 0.0, 0.0, 1, [0.0], [800.0])]
            run = Dict{Symbol, Any}(
                :tracers => sill_res, :erupted_tracers => sill_cargo,
                :tracers_Qmagma => Q_res, :erupted_tracers_Qmagma => Q_cargo,
            )
            @test isempty(ext.zircon_populations(Dict{Symbol, Any}()))
            run[:run_discrete], run[:run_Qmagma] = true, false
            @test ext.zircon_populations(run) == [("sill", sill_res, sill_cargo)]
            run[:run_discrete], run[:run_Qmagma] = false, true
            @test ext.zircon_populations(run) == [("Q_magma", Q_res, Q_cargo)]
            run[:run_discrete], run[:run_Qmagma] = true, true
            @test ext.zircon_populations(run) ==
                [("sill", sill_res, sill_cargo), ("Q_magma", Q_res, Q_cargo)]
        end
    end

end
