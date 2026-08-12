using QMagma
using GeoParams
using JLD2
using LinearAlgebra, SparseArrays, SparseDiffTools
using Random
using Test

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

            files_1d = QMagma.export_thermal_structure(joinpath(directory, "one"), z;
                fields=(temperature=T,))
            @test all(isfile, files_1d)
            data_1d = JLD2.load(joinpath(directory, "one.jld2"))
            @test data_1d["dimensionality"] == 1
            @test data_1d["temperature"] == T

            background = [200.0, 100.0, 0.0]
            x = [-2.0, 0.0, 2.0]
            T2 = QMagma.gaussian_thermal_structure(T, background, x; sigma=1.0)
            @test T2[2, :] == T
            @test T2[1, :] == T2[3, :]
            tapered_anomaly = abs.(T2[1, :] .- background)
            original_anomaly = abs.(T .- background)
            @test all(tapered_anomaly .<= original_anomaly)
            @test any(tapered_anomaly .< original_anomaly)
            Params, = QMagma.init_model(nz=3, L=2.0, Ttop=0.0, Tbot=200.0, Δt=1.0)
            ϕ2 = QMagma.melt_fraction_from_temperature(T2, Params.MatParam)
            @test size(ϕ2) == size(T2)
            @test all((0 .<= ϕ2) .& (ϕ2 .<= 1))
            files_2d = QMagma.export_thermal_structure(joinpath(directory, "two"), z;
                x, fields=(temperature=T2, melt_fraction=ϕ2,))
            @test all(isfile, files_2d)
            @test size(JLD2.load(joinpath(directory, "two.jld2"))["temperature"]) == (3, 3)

            y = [-2.0, 0.0, 2.0]
            T3 = QMagma.gaussian_thermal_structure(T, background, x; y, sigma=1.0)
            @test T3[2, 2, :] == T
            files_3d = QMagma.export_thermal_structure(joinpath(directory, "three"), z;
                x, y, fields=(temperature=T3,))
            @test all(isfile, files_3d)
            @test size(JLD2.load(joinpath(directory, "three.jld2"))["temperature"]) == (3, 3, 3)

            @test_throws "field temperature has size (2, 2)" QMagma.export_thermal_structure(
                joinpath(directory, "bad"), z; x, fields=(temperature=zeros(2, 2),))
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
            field, reverse(z), z)
    end

    @testset "init_model" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=11, L=10e3, Geotherm=10.0,
                                                     Ttop=0.0, Tbot=100.0, Δt=100SecYear)

        @test N == (11,)
        @test length(z) == 11
        @test length(T) == 11
        @test Δ[1] ≈ 1e3                       # dz = L/(nz-1)
        @test extrema(z) == (-10e3, 0.0)
        @test BC.Ttop == 0.0
        @test BC.Tbot == 100.0

        # linear geotherm: T = -Geotherm/1e3 * z + Ttop
        @test T[1] ≈ 100.0                     # bottom (z = -L)
        @test T[end] ≈ 0.0                     # top (z = 0)
        @test issorted(T, rev=true)            # temperature decreases from bottom to top
        @test Params.Told == T                  # previous state starts from the same geotherm

        # a custom MatParam tuple should be used as-is
        Params2, = QMagma.init_model(nz=5, L=1e3, Δt=1SecYear, MatParam=Params.MatParam)
        @test Params2.MatParam === Params.MatParam
        @test_throws "MatParam cannot be combined" QMagma.init_model(
            nz=5, L=1e3, Δt=1SecYear, MatParam=Params.MatParam, ρ=2500.0)

        # the constitutive laws are assembled here, so the caller sets them by keyword
        # instead of building a competing MatParam tuple
        Params3, = QMagma.init_model(nz=5, L=1e3, Δt=1SecYear, ρ=2500.0, Q_L=3.0e5,
                                     Conductivity=ConstantConductivity(k=3.0),
                                     HeatCapacity=ConstantHeatCapacity(),
                                     Melting=MeltingParam_Caricchi())
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
        Params, = QMagma.init_model(nz=5, L=1e3, Δt=1SecYear, ρ=2700.0)
        ep = QMagma.EruptionParams()
        @test QMagma.check_density_consistency(Params.MatParam, ep) ≈ 2700.0

        # Host-rock thermal and lithostatic densities describe the same crustal column.
        @test_throws "host-rock density mismatch" QMagma.check_density_consistency(
            Params.MatParam, QMagma.EruptionParams(ρ_crust=2600.0))
        # Chamber crystals and melt are distinct materials and need not equal host rock.
        @test QMagma.check_density_consistency(Params.MatParam,
            QMagma.EruptionParams(ρ_x=2900.0)) ≈ 2700.0
        @test QMagma.check_density_consistency(Params.MatParam,
            QMagma.EruptionParams(ρ_melt=1000.0)) ≈ 2700.0
        @test_throws "rtol must be nonnegative" QMagma.check_density_consistency(
            Params.MatParam, ep; rtol=-1.0)
    end

    @testset "nonlinear_solution: steady linear geotherm is (near) a fixed point" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=21, L=10e3, Geotherm=10.0,
                                                     Ttop=0.0, Tbot=100.0, Δt=100SecYear)
        Params.Told .= T

        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T)

        Tsol, converged, its = QMagma.nonlinear_solution(F, copy(T), Jac, colors;
                                                           Δ=Δ, N=N, BC=BC, Params=Params,
                                                           MatParam=Params.MatParam, verbose=false)

        @test its == 1                          # linear problem -> single Newton step
        @test Tsol[1] ≈ BC.Tbot
        @test Tsol[end] ≈ BC.Ttop
        @test maximum(abs.(Tsol .- T)) < 1e-2    # steady-state geotherm barely changes
    end

    @testset "source term Q heats the interior" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=21, L=10e3, Geotherm=0.0,
                                                     Ttop=0.0, Tbot=0.0, Δt=1e3SecYear)
        Params.Told .= T

        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T)

        @test all(Params.Q .== 0)               # Q defaults to zero -> no change in behaviour

        Tsol0, = QMagma.nonlinear_solution(F, copy(T), Jac, colors;
                                            Δ=Δ, N=N, BC=BC, Params=Params,
                                            MatParam=Params.MatParam, verbose=false)
        @test all(Tsol0 .≈ 0.0)                  # no source, zero BCs -> stays at zero

        Params.Q .= 1e-5                         # uniform volumetric heat source [W/m^3]
        Tsol1, = QMagma.nonlinear_solution(F, copy(T), Jac, colors;
                                            Δ=Δ, N=N, BC=BC, Params=Params,
                                            MatParam=Params.MatParam, verbose=false)

        @test all(Tsol1[2:end-1] .> Tsol0[2:end-1])  # interior heats up due to Q
        @test Tsol1[1] ≈ BC.Tbot                     # boundary conditions still enforced
        @test Tsol1[end] ≈ BC.Ttop
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
            -1.0, 0.0, 1.0)

        constant = QMagma.FluxHistory(:constant; base=2.0)
        ramp_history = QMagma.FluxHistory(:ramp; base=1.0, peak=3.0,
            t_start=2.0, t_end=4.0)
        pulse = QMagma.FluxHistory(:pulse; base=1.0, peak=3.0,
            t_start=2.0, t_end=4.0)
        table = QMagma.FluxHistory(:table; times=[0.0, 2.0, 4.0], rates=[1.0, 3.0, 1.0])
        @test QMagma.injected_thickness(constant, 1.0, 4.0) == 8.0
        @test QMagma.injected_thickness(ramp_history, 1.0, 4.0) == 8.0
        @test QMagma.injected_thickness(pulse, 1.0, 4.0) == 8.0
        @test QMagma.injected_thickness(table, 0.0, 4.0) == 8.0
        @test ramp_history(3.0) == 2.0
        @test pulse(3.0) == 3.0
        @test table(3.0) == 2.0
        @test_throws "strictly increasing" QMagma.FluxHistory(:table;
            times=[0.0, 0.0], rates=[1.0, 2.0])

        mktemp() do path, io
            write(io, "time_kyr,flux_m_per_yr\n0,0.1\n10,0.3\n")
            close(io)
            loaded = QMagma.load_flux_history(path; time_scale=1.0, rate_scale=1.0)
            @test loaded.mode == :table
            @test loaded(5.0) == 0.2
            @test QMagma.injected_thickness(loaded, 0.0, 10.0) == 2.0
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
        ȧ = d/interval
        # (1) a constant rate reproduces the event times of the interval-keyed schedule
        A, time = 0.0, 0.0
        thickness_keyed = Int[]
        time_keyed = Int[]
        for _ in 1:2000
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            push!(thickness_keyed, QMagma.sills_due(A, Δh, d))
            push!(time_keyed, floor(Int, (time + Δt)/interval) - floor(Int, time/interval))
            A += Δh
            time += Δt
        end
        @test thickness_keyed == time_keyed
        @test A ≈ ȧ*time

        # (2) a linear ramp emplaces the sills its integrated flux delivers, to within one
        ramp(t) = 2ȧ*t/(2000Δt)
        A, time, n_ramp = 0.0, 0.0, 0
        for _ in 1:2000
            Δh = QMagma.injected_thickness(ramp, time, Δt)
            n_ramp += QMagma.sills_due(A, Δh, d)
            A += Δh
            time += Δt
        end
        @test A ≈ ȧ*time                     # same total as the constant rate it ramps around
        @test abs(n_ramp - A/d) <= 1

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
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=41, L=40e3, Geotherm=20.0,
                                                     Ttop=0.0, Tbot=800.0, Δt=200SecYear)
        Params.Told .= T

        Tsill   = 1200.0
        Silltop = 10.0    # km
        Sillbot = 20.0    # km
        ȧ       = 100.0/500SecYear   # Sillthick/Sill_interval [m/s]

        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)

        ind_zone = findall(z .>= -Sillbot*1e3 .&& z .<= -Silltop*1e3)
        ind_out  = setdiff(1:length(z), ind_zone)

        @test all(Params.Q[ind_out] .== 0)          # zero outside the injection zone
        @test all(Params.Q[ind_zone] .> 0)          # positive heat input while Told < Tsill everywhere in zone

        # Ensemble-mean host-rock velocity from uniformly distributed discrete sills.
        ind_below = findall(z .< -Sillbot*1e3)
        ind_above = findall(z .> -Silltop*1e3)
        @test all(Params.w[ind_below] .< 0)         # pushed down below the zone
        @test all(Params.w[ind_above] .> 0)         # pushed up above the zone
        @test Params.w[argmin(abs.(z .+ 15e3))] ≈ 0.0 atol=eps(ȧ)
        @test Params.w[argmin(abs.(z .+ 18e3))] < 0
        @test Params.w[argmin(abs.(z .+ 12e3))] > 0
        @test issorted(abs.(Params.w[ind_below]))            # |w| grows approaching the zone from below (z increasing)
        @test issorted(abs.(Params.w[ind_above]), rev=true)  # |w| decays moving away from the zone above (z increasing)

        centers = range(-Sillbot*1e3, -Silltop*1e3; length=20_001)
        d = 100.0
        event_rate = ȧ/d
        for i in (5, argmin(abs.(z .+ 18e3)), argmin(abs.(z .+ 12e3)), length(z)-5)
            offsets = z[i] .- centers
            numerical_mean = event_rate * sum(sign.(offsets) .* QMagma.crack_perp_displacement(offsets, d/2)) / length(centers)
            @test Params.w[i] ≈ numerical_mean rtol=5e-4 atol=eps(ȧ)
        end

        # heat input vanishes once the column has fully equilibrated to Tsill (ϕ=1, T=Tsill)
        Params.Told .= Tsill
        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
        @test all(isapprox.(Params.Q[ind_zone], 0.0, atol=1e-6))
    end

    @testset "discrete sills vs Q_magma agree in the many-sill limit" begin
        H        = 40.0
        γ        = 20.0
        Ttop     = 0.0
        Tbot     = Ttop + H*γ
        nz       = floor(Int64, H*1e3/100.0)
        Δt       = 200SecYear
        Tsill    = 1200.0
        Sillthick   = 100.0      # small, frequent sills -> many-sill limit
        Sill_int_yr = 500.0
        Silltop  = 10.0
        Sillbot  = 20.0
        nt       = 4000

        Params, BC, N, Δ, T, z = QMagma.init_model(nz=nz, L=H*1e3, Geotherm=γ, Ttop=Ttop, Tbot=Tbot, Δt=Δt)
        MatParam = Params.MatParam
        Params.Told .= T

        Params_Q = deepcopy(Params)
        T_Q      = deepcopy(T)
        Params_Q.Told .= T_Q
        ȧ        = Sillthick/Sill_int_yr/SecYear

        rocks = zero(T)
        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0
        A_inj = 0.0
        rng = MersenneTwister(42)
        injected_count = 0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam, verbose=false)
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            for _ in 1:n_injections
                Sill_z0 = rand(rng, -Sillbot*1e3:1:-Silltop*1e3)
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                Params.Told .= T
                injected_count += 1
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam, verbose=false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        zone = findall(-Sillbot*1e3 .<= z .<= -Silltop*1e3)
        profile_rms = sqrt(sum(abs2, T .- T_Q)/length(T))
        zone_mean_diff = abs(sum(T[zone] .- T_Q[zone])/length(zone))
        @test injected_count == floor(Int, nt*Δt/(Sill_int_yr*SecYear))
        @test profile_rms < 5.0
        @test zone_mean_diff < 1.0
    end

    @testset "discrete sills vs Q_magma agree with larger, less-frequent sills (advection matters)" begin
        H        = 40.0
        γ        = 20.0
        Ttop     = 0.0
        Tbot     = Ttop + H*γ
        nz       = floor(Int64, H*1e3/20.0)
        Δt       = 100SecYear
        Tsill    = 1200.0
        Sillthick   = 100.0
        Sill_int_yr = 1000.0
        Silltop  = 10.0
        Sillbot  = 20.0
        nt       = 5000

        MatParam = (SetMaterialParams(Name="RockMelt", Phase=0,
                            Density         = ConstantDensity(ρ=2700kg/m^3),
                            LatentHeat      = ConstantLatentHeat(Q_L=0.0J/kg),
                            RadioactiveHeat = ExpDepthDependentRadioactiveHeat(H_0=0e-7Watt/m^3),
                            Conductivity    = ConstantConductivity(k=3.0),
                            HeatCapacity    = ConstantHeatCapacity(),
                            Melting         = MeltingParam_Assimilation()
                        ),)

        Params, BC, N, Δ, T, z = QMagma.init_model(nz=nz, L=H*1e3, Geotherm=γ, Ttop=Ttop, Tbot=Tbot, Δt=Δt, MatParam=MatParam)
        Params.Told .= T

        Params_Q = deepcopy(Params)
        T_Q      = deepcopy(T)
        Params_Q.Told .= T_Q
        ȧ        = Sillthick/Sill_int_yr/SecYear

        rocks = zero(T)
        Jac, colors = thermal_jacobian_workspace(N[1])
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0
        A_inj = 0.0
        rng = MersenneTwister(42)
        injected_count = 0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam, verbose=false)
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            for _ in 1:n_injections
                Sill_z0 = rand(rng, -Sillbot*1e3:1:-Silltop*1e3)
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                Params.Told .= T
                injected_count += 1
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam, verbose=false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        zone = findall(-Sillbot*1e3 .<= z .<= -Silltop*1e3)
        profile_rms = sqrt(sum(abs2, T .- T_Q)/length(T))
        zone_mean_diff = abs(sum(T[zone] .- T_Q[zone])/length(zone))
        @test injected_count == floor(Int, nt*Δt/(Sill_int_yr*SecYear))
        @test profile_rms < 10.0
        @test zone_mean_diff < 2.0
    end

    @testset "crack_perp_displacement" begin
        d = 100.0
        @test QMagma.crack_perp_displacement(0.0, d) ≈ d            # max displacement at sill center
        @test QMagma.crack_perp_displacement(1e6, d) ≈ 0.0 atol=1e-2 # decays far from the sill
        @test QMagma.crack_perp_displacement(0.0, d) > QMagma.crack_perp_displacement(1e3, d)
    end

    @testset "semilagrangian_advection" begin
        z = collect(-10e3:100.0:0.0)
        T = collect(range(0.0, 100.0, length=length(z)))

        # zero displacement leaves the field unchanged
        T_same = QMagma.semilagrangian_advection(T, zero(z), z)
        @test T_same ≈ T
    end

    @testset "advect_markers!" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=41, L=40e3, Geotherm=20.0,
                                                     Ttop=0.0, Tbot=800.0, Δt=200SecYear)
        Params.Told .= T

        Tsill   = 1200.0
        Silltop = 10.0
        Sillbot = 20.0
        ȧ       = 100.0/500SecYear

        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)

        markers = [-30e3, -15e3, -5e3]   # below, inside, and above the injection zone
        markers0 = copy(markers)
        QMagma.advect_markers!(markers, Params)

        @test markers[1] < markers0[1]   # marker below the zone is pushed further down
        @test markers[2] ≈ markers0[2]   # marker inside the zone (w=0) does not move
        @test markers[3] > markers0[3]   # marker above the zone is pushed further up
    end

    @testset "passive tracers" begin
        Silltop, Sillbot = 10.0, 20.0

        tracers = QMagma.init_tracers(Silltop, Sillbot; n=5)
        @test length(tracers) == 5
        @test all(t -> t.phase == 0, tracers)
        @test all(t -> isempty(t.time_vec) && isempty(t.T_vec), tracers)
        @test extrema(t.z for t in tracers) == (-Sillbot*1e3, -Silltop*1e3)

        Sill_z0, Sill_thick, Sill_T = -15e3, 100.0, 1200.0
        QMagma.add_sill_tracers!(tracers, Sill_z0, Sill_thick, Sill_T; n=3)
        @test length(tracers) == 8
        new_tracers = tracers[end-2:end]
        @test all(t -> t.phase == 1, new_tracers)
        @test all(t -> t.T == Sill_T, new_tracers)
        @test all(t -> Sill_z0 - Sill_thick/2 <= t.z <= Sill_z0 + Sill_thick/2, new_tracers)

        Params, BC, N, Δ, T, z = QMagma.init_model(nz=41, L=40e3, Geotherm=20.0,
                                                     Ttop=0.0, Tbot=800.0, Δt=200SecYear)
        Params.Told .= T
        ȧ = 100.0/500SecYear
        QMagma.compute_Q_magma!(Params, Params.MatParam, z; Tsill=Sill_T, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)

        push!(tracers, QMagma.Tracer(-30e3, 0.0, 0.0, 0, Float64[], Float64[]))  # below the zone

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
        time_Myr = collect(range(0.0, 0.2, length=20))
        T_C      = collect(range(900.0, 650.0, length=20))   # monotonic cooling path

        tracers = [
            QMagma.Tracer(-10e3, T_C[end], 0.0, 1, copy(time_Myr), copy(T_C)),
            QMagma.Tracer(-12e3, 0.0, 0.0, 0, Float64[], Float64[]),   # too few points: skipped
        ]

        result = QMagma.compute_zircon_ages(tracers; nx=30)
        @test length(result.age_years) == 1
        @test length(result.zircon_radius_um) == 1
        @test result.age_years[1] > 0
        @test result.zircon_radius_um[1] > 0

        result2 = QMagma.compute_zircon_ages(tracers; nx=30, return_results=true)
        @test length(result2.results) == 1
        @test result2.age_years[1] ≈ QMagma.volume_averaged_age(result2.results[1])
    end

    @testset "add_zone_tracers!" begin
        Silltop, Sillbot, Tsill = 10.0, 20.0, 1200.0
        tracers = QMagma.Tracer[]

        QMagma.add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=4)
        @test length(tracers) == 4
        @test all(t -> t.phase == 1, tracers)
        @test all(t -> t.T == Tsill, tracers)
        @test all(t -> t.z ≈ -(Silltop + Sillbot)/2*1e3, tracers)

        QMagma.add_zone_tracers!(tracers, Silltop, Sillbot, Tsill; n=2)
        @test length(tracers) == 6
    end

    @testset "advect_tracers_sill!" begin
        Sill_z0, Sill_thick = -15e3, 200.0

        tracers = [
            QMagma.Tracer(-15e3, 0.0, 0.0, 1, Float64[], Float64[]),    # inside the sill
            QMagma.Tracer(-10e3, 0.0, 0.0, 0, Float64[], Float64[]),    # above the sill
            QMagma.Tracer(-20e3, 0.0, 0.0, 0, Float64[], Float64[]),    # below the sill
        ]
        z0 = [t.z for t in tracers]

        QMagma.advect_tracers_sill!(tracers, Sill_z0, Sill_thick; SillType=:elastic)

        @test tracers[1].z ≈ z0[1]      # inside the sill: unaffected
        @test tracers[2].z > z0[2]      # above the sill: pushed further up
        @test tracers[3].z < z0[3]      # below the sill: pushed further down

        # Sill_thick is the full aperture, shared equally by the two walls.
        tracers2 = [QMagma.Tracer(-10e3, 0.0, 0.0, 0, Float64[], Float64[])]
        QMagma.advect_tracers_sill!(tracers2, Sill_z0, Sill_thick; SillType=:constant)
        @test tracers2[1].z ≈ -10e3 + Sill_thick/2
    end

    @testset "insert_sill" begin
        z = collect(-10e3:100.0:0.0)
        T = fill(200.0, length(z))
        rocks = zeros(length(z))

        Sill_z0 = -5e3
        Sill_thick = 400.0
        Sill_T = 1200.0
        T2, rocks2 = QMagma.insert_sill(T, rocks, z; Sill_thick=Sill_thick, Sill_z0=Sill_z0,
                                         Sill_T=Sill_T, SillType=:constant)

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
        deep = zeros(length(z)); deep[findall(-6e3 .<= z .<= -4e3)] .= 1.0
        _, _, h_out_deep = QMagma.insert_sill(T, deep, z; Sill_thick=Sill_thick, Sill_z0=Sill_z0,
                                              Sill_T=Sill_T)
        @test h_out_deep ≈ 0.0 atol=1e-9

        shallow = zeros(length(z)); shallow[findall(z .>= -500.0)] .= 1.0
        _, rocks_shallow, h_out_shallow = QMagma.insert_sill(T, shallow, z; Sill_thick=Sill_thick,
                                                            Sill_z0=-1e3, Sill_T=Sill_T)
        @test h_out_shallow > 0
        @test length(rocks_shallow) == length(z)
        # the outflow is exactly what the opening displacement carries off the grid
        z_shift = z .- (-1e3)
        Displ = zero(z_shift)
        above, below = findall(z_shift .> 0), findall(z_shift .< 0)
        Displ[above] .=  QMagma.crack_perp_displacement(z_shift[above], Sill_thick/2; r=5e3)
        Displ[below] .= -QMagma.crack_perp_displacement(z_shift[below], Sill_thick/2; r=5e3)
        expected_out = QMagma.integrated_content(shallow, z) -
            QMagma.integrated_content(QMagma.conservative_advection(shallow, Displ, z), z)
        @test h_out_shallow ≈ expected_out
    end

    @testset "insert_sill conserves injected crust (grey rocks)" begin
        # regression: the phase indicator must not leak under repeated advection. The old
        # semi-Lagrangian + round scheme lost ~25% of the grey over 40 injections; the
        # conservative remap keeps Σ rocks·Δz ≈ total injected thickness.
        z  = collect(-40e3:100.0:0.0)
        Δz = 100.0
        T  = fill(400.0, length(z))
        rocks = zero(z)
        injected = 0.0
        depths = collect(-30e3:1.0:-5e3)
        for k in 1:40
            z0 = depths[(k*911) % length(depths) + 1]   # deterministic spread of depths
            T, rocks = QMagma.insert_sill(T, rocks, z; Sill_thick=400.0, Sill_z0=z0, Sill_T=1200.0)
            injected += 400.0
        end
        grey = QMagma.integrated_content(rocks, z)
        @test grey ≈ injected atol=1e-8
    end

    @testset "insert_sill adds its full content inside intruded magma" begin
        z  = collect(-40e3:200.0:0.0)
        Δz = 200.0
        T  = fill(600.0, length(z))
        rocks = zeros(length(z))
        rocks[findall(-20e3 .<= z .<= -10e3)] .= 1.0
        content = QMagma.integrated_content(rocks, z)

        _, rocks2, h_out = QMagma.insert_sill(T, rocks, z; Sill_thick=100.0, Sill_z0=-15e3,
                                               Sill_T=1200.0)
        @test h_out == 0.0                              # the pile is nowhere near a boundary
        gain = QMagma.integrated_content(rocks2, z) - content
        @test gain ≈ 100.0

        # the loss is the clipping, not the remap: the advection alone conserves content
        # and pushes cells past 1
        z_shift = z .- (-15e3)
        Displ = zero(z_shift)
        above, below = findall(z_shift .> 0), findall(z_shift .< 0)
        Displ[above] .=  QMagma.crack_perp_displacement(z_shift[above], 50.0; r=5e3)
        Displ[below] .= -QMagma.crack_perp_displacement(z_shift[below], 50.0; r=5e3)
        advected = QMagma.conservative_advection(rocks, Displ, z)
        @test QMagma.integrated_content(advected, z) ≈ content
        @test maximum(advected) > 1.0
    end

    @testset "find_eruptible_region" begin
        z = collect(-10e3:100.0:0.0)
        ϕ = zeros(length(z))

        @test QMagma.find_eruptible_region(ϕ, z) === nothing

        ind = findall(-6e3 .<= z .<= -4e3)
        ϕ[ind] .= 0.8
        region = QMagma.find_eruptible_region(ϕ, z; ϕ_threshold=0.5)
        @test region !== nothing
        zlo, zhi = region
        @test zlo ≈ minimum(z[ind])
        @test zhi ≈ maximum(z[ind])

        # two disjoint melt lenses: only the larger contiguous run should be reported,
        # not the envelope spanning both (which would wildly overstate the thickness)
        ϕ2 = zeros(length(z))
        ind_small = findall(-9e3 .<= z .<= -8.9e3)   # ~100 m lens near the bottom
        ind_big   = findall(-3e3 .<= z .<= -1e3)     # ~2 km lens near the top
        ϕ2[ind_small] .= 0.8
        ϕ2[ind_big]   .= 0.8
        region2 = QMagma.find_eruptible_region(ϕ2, z; ϕ_threshold=0.5)
        @test region2 !== nothing
        zlo2, zhi2 = region2
        @test zlo2 ≈ minimum(z[ind_big])
        @test zhi2 ≈ maximum(z[ind_big])
    end

    @testset "eruption trigger control states" begin
        none = QMagma.eruption_control_state("None")
        @test !any(values(none))

        melt = QMagma.eruption_control_state("Melt thickness")
        @test melt.threshold && melt.radius && melt.max_depth && melt.collapse
        @test !melt.pressure && !melt.shear_modulus && !melt.magma_compressibility

        elastic = QMagma.eruption_control_state("Elastic box model")
        @test !elastic.threshold && all(values(elastic)[2:end])

        dh = QMagma.eruption_control_state("D&H 3-phase")
        @test !dh.threshold && all(values(dh)[2:end])
        @test_throws "unknown eruption trigger" QMagma.eruption_control_state("typo")
    end

    @testset "GUI flux histories" begin
        constant = QMagma.gui_flux_history("Constant";
            base_m_per_yr=0.1, peak_m_per_yr=0.2, start_kyr=10.0, end_kyr=20.0)
        ramp = QMagma.gui_flux_history("Linear ramp";
            base_m_per_yr=0.1, peak_m_per_yr=0.3, start_kyr=10.0, end_kyr=20.0)
        pulse = QMagma.gui_flux_history("Pulse";
            base_m_per_yr=0.1, peak_m_per_yr=0.3, start_kyr=10.0, end_kyr=20.0)
        @test constant.mode == :constant
        @test ramp(15e3SecYear)*SecYear ≈ 0.2
        @test pulse(15e3SecYear)*SecYear ≈ 0.3
        @test_throws "unknown flux mode" QMagma.gui_flux_history("typo";
            base_m_per_yr=0.1, peak_m_per_yr=0.2, start_kyr=10.0, end_kyr=20.0)
    end

    @testset "eruption trigger criteria" begin
        fires(trigger; kwargs...) = QMagma.eruption_fires(trigger;
            thickness=600.0, threshold=500.0, overpressure=25e6,
            pressure_critical=20e6, h_erupt=300.0, z_lo=-6e3, z_hi=-5e3,
            z_erupt_max=15e3, near_boundary=false, Δz=100.0, kwargs...)

        @test fires("Melt thickness")
        @test fires("Elastic box model")
        @test fires("D&H 3-phase")
        @test !fires("Melt thickness"; thickness=400.0)
        @test !fires("Elastic box model"; overpressure=19e6)
        @test !fires("D&H 3-phase"; h_erupt=0.0)
        @test !fires("Melt thickness"; z_lo=-20e3, z_hi=-19e3)
        @test !fires("Melt thickness"; near_boundary=true)
        @test !fires("Melt thickness"; h_erupt=200.0)
        @test_throws "unknown active eruption trigger" fires("None")
    end

    @testset "recorded surface subsidence" begin
        @test QMagma.collapse_surface_subsidence(:caldera, 250.0) == 250.0
        @test QMagma.collapse_surface_subsidence(:hybrid, 250.0) == 250.0
        @test QMagma.collapse_surface_subsidence(:elastic, 250.0) == 0.0
        @test_throws "unknown eruption collapse method" QMagma.collapse_surface_subsidence(:typo, 250.0)
    end

    @testset "erupt_melt!" begin
        z = collect(-10e3:100.0:0.0)
        T = fill(800.0, length(z))
        rocks = zeros(length(z))

        Erupt_z0 = -5e3
        Erupt_thick = 1000.0
        ind = findall(abs.(z .- Erupt_z0) .<= Erupt_thick / 2)
        T[ind] .= 1200.0
        rocks[ind] .= 1.0

        T2, rocks2 = QMagma.erupt_melt!(T, rocks, z; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick)

        @test length(T2) == length(z)
        # caldera subsidence: the erupted band's heat leaves exactly, replaced at the
        # top by the surface-temperature fill; no rock parcel changes temperature
        @test sum(T2) ≈ sum(T) - sum(T[ind]) + length(ind)*T[end]
        # the roof block (uniformly 800.0) dropped onto the chamber floor
        @test all(T2[ind] .≈ 800.0)
        @test all(rocks2[ind] .== 0)

        # erupting from the middle of a wider intruded pile: the roof grey drops onto
        # the floor grey, so the total drops by exactly the erupted band's content and
        # no host-rock gap is left inside the remaining grey
        rocks_wide = zeros(length(z))
        ind_wide = findall(abs.(z .- Erupt_z0) .<= 2000.0)
        rocks_wide[ind_wide] .= 1.0
        _, rocks3 = QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick)
        @test length(rocks3) == length(z)
        @test sum(rocks3) == sum(rocks_wide) - length(ind)
        @test all((rocks3 .== 0) .| (rocks3 .== 1))
        grey_idx = findall(rocks3 .> 0)
        @test all(diff(grey_idx) .== 1)

        # the whole grey envelope above the vent subsides with the roof: its top edge
        # drops by the band's actual grid footprint (length(ind) cells)
        Δz_grid = z[2] - z[1]
        @test maximum(z[rocks3 .> 0]) ≈ maximum(z[rocks_wide .> 0]) - length(ind)*Δz_grid

        # elastic collapse variant: the walls close over the vent at their own
        # temperature (uniform 800 surroundings -> 800 in the band), grey bounded by
        # the band's content
        T4, rocks5 = QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=:elastic)
        @test all(T4[ind] .≈ 800.0)
        @test sum(rocks_wide) - length(ind) <= sum(rocks5) <= sum(rocks_wide)
        # grey is now the conservative fractional remap (co-moves with T, no round): the
        # surviving grey stays non-negative and does not inflate away from a phase indicator
        @test all(rocks5 .>= -1e-9)
        @test maximum(rocks5) <= 2.0

        # hybrid variant: floor rises elastically, roof transitions to a rigid
        # subsidence - the surface sinks by ~the erupted thickness and the warped
        # grid stays monotonic
        D = QMagma.collapse_displacement(z; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=:hybrid)
        @test all(diff(z .+ D) .> 0)
        @test isapprox(D[end], -Erupt_thick; rtol=0.1)
        @test D[1] == 0.0
        T5, rocks6 = QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=:hybrid)
        @test all(T5[ind] .≈ 800.0)                    # walls meet at their own T
        @test minimum(T5) >= minimum(T) - 1e-8         # intensive transport: no dilution values
        @test all(rocks6 .>= -1e-9)
        @test sum(rocks6) <= sum(rocks_wide)           # grey conserved minus the erupted band

        # every closure reports the intruded magma it took out of the column, the term the
        # magma-volume budget debits (grey removed = erupted band + boundary outflow)
        for closure in (:caldera, :elastic, :hybrid)
            rock_in = copy(rocks_wide)
            _, rock_out, h_out = QMagma.erupt_melt!(T, rock_in, z; Erupt_z0=Erupt_z0,
                                                    Erupt_thick=Erupt_thick, method=closure)
            @test h_out ≈ (sum(rock_in) - sum(rock_out))*Δz_grid
            @test h_out > 0
        end
        # a band that misses the grid removes nothing
        @test QMagma.erupt_melt!(T, rocks_wide, z; Erupt_z0=1e4, Erupt_thick=Erupt_thick)[3] == 0.0

        @test_throws "unknown eruption collapse method" QMagma.erupt_melt!(T, rocks, z;
            Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=:typo)
        @test_throws "collapse_displacement supports only" QMagma.collapse_displacement(z;
            Erupt_z0=Erupt_z0, Erupt_thick=Erupt_thick, method=:caldera)
    end

    @testset "melt_thickness" begin
        z = collect(-10e3:100.0:0.0)
        ϕ = zeros(length(z))
        ind = findall(-6e3 .<= z .<= -4e3)
        ϕ[ind] .= 0.8
        @test QMagma.melt_thickness(ϕ, z, -6e3, -4e3) ≈ 0.8*2e3
        # empty interval
        @test QMagma.melt_thickness(ϕ, z, -2e3, -1e3) == 0.0
    end

    @testset "collapse_markers!/collapse_tracers!" begin
        Erupt_z0, Erupt_thick = -5e3, 1000.0

        # caldera: above drops by the full thickness, inside lands on the floor, below stays
        m = [-2e3, -5e3, -8e3]
        QMagma.collapse_markers!(m, Erupt_z0, Erupt_thick)
        @test m ≈ [-3e3, -5.5e3, -8e3]

        # elastic: inside snaps to the vent center, outside moves toward it by less
        # than the half-thickness
        m2 = [-2e3, -5e3, -8e3]
        QMagma.collapse_markers!(m2, Erupt_z0, Erupt_thick; method=:elastic)
        @test m2[2] == Erupt_z0
        @test -2.5e3 < m2[1] < -2e3
        @test -8e3 < m2[3] < -7.5e3

        # tracers follow the same displacement
        tracers = [QMagma.Tracer(-2e3, 600.0, 0.0, 0, Float64[], Float64[]),
                   QMagma.Tracer(-8e3, 600.0, 0.0, 0, Float64[], Float64[])]
        QMagma.collapse_tracers!(tracers, Erupt_z0, Erupt_thick)
        @test tracers[1].z ≈ -3e3
        @test tracers[2].z ≈ -8e3
    end

    @testset "extract_erupted_tracers!" begin
        z = collect(-6e3:500.0:-4e3)
        ϕ = [0.2, 0.4, 0.8, 0.6, 0.2]
        tracers = [
            QMagma.Tracer(-5e3, 1200.0, 1.0, 1, Float64[], Float64[]),
            QMagma.Tracer(-4.5e3, 1100.0, 0.7, 1, Float64[], Float64[]),
            QMagma.Tracer(-5.5e3, 900.0, 0.4, 0, Float64[], Float64[]),
            QMagma.Tracer(-6e3, 600.0, 0.0, 0, Float64[], Float64[]),
        ]

        erupted, h_cargo = QMagma.extract_erupted_tracers!(MersenneTwister(1), tracers,
            ϕ, z, -5.5e3, -4.5e3, 1e3; eligible_phase=1)

        @test length(erupted) == 2
        @test h_cargo ≈ 0.8*500 + 0.6*250
        @test length(tracers) == 2
        @test sort([tracer.phase for tracer in tracers]) == [0, 0]

        # Each cell's represented melt is independent of how many tracers were seeded.
        one = [QMagma.Tracer(-5e3, 1200.0, 0.8, 1, Float64[], Float64[])]
        many = [QMagma.Tracer(-5e3, 1200.0, 0.8, 1, Float64[], Float64[]) for _ in 1:4]
        _, h_one = QMagma.extract_erupted_tracers!(MersenneTwister(2), one, ϕ, z,
            -5.25e3, -4.75e3, 1e3; eligible_phase=1)
        _, h_many = QMagma.extract_erupted_tracers!(MersenneTwister(2), many, ϕ, z,
            -5.25e3, -4.75e3, 1e3; eligible_phase=1)
        @test h_one ≈ h_many
        @test h_one ≈ 0.8 * 500

        # All-phase cargo represents the full mush even when some melt-bearing cells
        # have no tracer. Sparse diagnostic sampling must not shrink physical withdrawal.
        sparse = [QMagma.Tracer(-5e3, 1200.0, 0.8, 1, Float64[], Float64[])]
        _, h_sparse = QMagma.extract_erupted_tracers!(MersenneTwister(3), sparse, ϕ, z,
            -5.5e3, -4.5e3, 1e3; eligible_phase=nothing)
        @test h_sparse ≈ 0.4*250 + 0.8*500 + 0.6*250

        partial = [
            QMagma.Tracer(-5.5e3, 900.0, 0.4, 1, Float64[], Float64[]),
            QMagma.Tracer(-5e3, 1200.0, 0.8, 1, Float64[], Float64[]),
            QMagma.Tracer(-4.5e3, 1100.0, 0.6, 1, Float64[], Float64[]),
        ]
        cargo, h_partial = QMagma.extract_erupted_tracers!(MersenneTwister(4), partial,
            ϕ, z, -5.5e3, -4.5e3, 350.0; eligible_phase=1)
        @test 1 <= length(cargo) <= 2
        @test length(cargo) + length(partial) == 3
        @test h_partial > 0

        @test_throws "h_erupt must be nonnegative" QMagma.extract_erupted_tracers!(
            MersenneTwister(3), QMagma.Tracer[], ϕ, z, -5.5e3, -4.5e3, -1.0;
            eligible_phase=nothing)
    end

    @testset "unified eruption event fails on split bookkeeping" begin
        z = collect(-1000.0:100.0:0.0)
        T = fill(1000.0, length(z))
        ϕ = zeros(length(z)); ϕ[4:8] .= 0.8
        Params, = QMagma.init_model(nz=length(z), L=1000.0, Geotherm=0.0,
            Ttop=1000.0, Tbot=1000.0, Δt=1.0)
        tracers = [QMagma.Tracer(zi, 1000.0, 0.8, 1, [0.0], [1000.0]) for zi in z[4:8]]

        Tnew, _, cargo, event = QMagma.realize_eruption!(MersenneTwister(7), T,
            zeros(length(z)), tracers, ϕ, z, Params.MatParam, Params.Phases;
            realization_time=1.0, h_requested=240.0, h_booked=240.0,
            z_lo=z[4], z_hi=z[8],
            trigger=:test, closure=:caldera, eligible_phase=1)

        @test event.requested == event.state_withdrawn == event.booked == 240.0
        @test event.cargo_represented ≈ 240.0
        @test event.cargo_count == length(cargo) == 3
        @test event.enthalpy_after == QMagma.column_enthalpy(Tnew, z,
            Params.MatParam, Params.Phases)
        @test event.magma_removed == 0.0        # no intruded magma in this column to remove

        # with intruded magma present the event carries the grey the closure withdrew,
        # independently of the four melt-thickness accounts
        rocks = zeros(length(z)); rocks[3:9] .= 1.0
        _, rocks_after, _, event_grey = QMagma.realize_eruption!(MersenneTwister(7), T,
            rocks, [QMagma.Tracer(zi, 1000.0, 0.8, 1, [0.0], [1000.0]) for zi in z[4:8]],
            ϕ, z, Params.MatParam, Params.Phases;
            realization_time=1.0, h_requested=240.0, h_booked=240.0,
            z_lo=z[4], z_hi=z[8],
            trigger=:test, closure=:caldera, eligible_phase=1)
        @test event_grey.magma_removed ≈ QMagma.integrated_content(rocks, z) -
            QMagma.integrated_content(rocks_after, z)
        @test event_grey.magma_removed > 0

        event_args = function (; state=10.0, cargo=10.0, booked=10.0)
            return (; trigger_time=1.0, realization_time=1.0, requested=10.0,
                state_withdrawn=state,
                cargo_represented=cargo, booked, z_lo=-1.0, z_hi=1.0,
                z_centroid=0.0, trigger=:test, closure=:caldera,
                enthalpy_before=100.0, enthalpy_after=90.0, erupted_enthalpy=10.0)
        end
        @test_throws "state-withdrawn=9.0" QMagma.EruptionEvent(;
            event_args(state=9.0)...)
        @test_throws "cargo-represented=8.0" QMagma.EruptionEvent(;
            event_args(cargo=8.0)..., cargo_atol=1.0)
        @test_throws "booked=11.0" QMagma.EruptionEvent(;
            event_args(booked=11.0)...)
    end

    @testset "overpressure trigger (D&H 3-phase)" begin
        ep = QMagma.EruptionParams(ΔP_crit=20e6, z_erupt_max=15e3)
        # higher P dissolves more water -> less gas -> denser
        ρlo, glo = QMagma.mixture_density(5e7, 1100.0, 0.7, ep; z_centroid=-5e3)
        ρhi, ghi = QMagma.mixture_density(3e8, 1100.0, 0.7, ep; z_centroid=-5e3)
        @test 0 <= glo <= 1 && 0 <= ghi <= 1
        @test ρhi > ρlo && ghi < glo

        z = collect(-30e3:100.0:0.0)
        ϕ = [(-15e3 <= zi <= -12e3) ? 0.8 : 0.1 for zi in z]
        ind, V_e, zc = QMagma.eruptible_mush(ϕ, z; ϕ_erupt=ep.ϕ_erupt)
        @test !isempty(ind) && V_e == 3e3 && zc ≈ -13.5e3

        # A second disconnected lens is a different pressure reservoir, not extra volume.
        ϕ[-5e3 .<= z .<= -4e3] .= 0.9
        ind2, V_e2, zc2 = QMagma.eruptible_mush(ϕ, z; ϕ_erupt=ep.ϕ_erupt)
        @test ind2 == ind
        @test V_e2 == V_e
        @test zc2 == zc

        st = QMagma.EruptionState(); QMagma.init_eruption!(st, ep, zc)
        @test st.P ≈ st.P_lith
        @test !QMagma.overpressure_erupts(st, ep, zc)   # at lithostatic: no eruption
        st.P = st.P_lith + ep.ΔP_crit + 1e6
        @test QMagma.overpressure_erupts(st, ep, zc)    # over threshold: erupts
        ep_deep = QMagma.EruptionParams(ΔP_crit=20e6, z_erupt_max=10e3)
        @test !QMagma.overpressure_erupts(st, ep_deep, zc)

        # second boiling (D&H's dominant trigger): crystallizing (ϕ_melt↓) must exsolve gas
        _, g_wet = QMagma.mixture_density(1e8, 1100.0, 0.8, ep; z_centroid=-5e3)
        _, g_dry = QMagma.mixture_density(1e8, 1100.0, 0.4, ep; z_centroid=-5e3)
        @test g_dry > g_wet

        # Recharge builds pressure without an accessible failure threshold.
        ep.η_r = 1e30
        ep.ΔP_crit = 1e15
        QMagma.init_eruption!(st, ep, zc)
        for _ in 1:200
            QMagma.step_overpressure!(st, ep, 1200.0+273.15, 0.8, V_e, 1e-9, 1e10;
                z_centroid=zc)
        end
        @test st.P - st.P_lith > 0
        @test st.h_erupt == 0.0

        # A reachable threshold drains no more melt than was recharged.
        ep.ΔP_crit = 20e6
        st2 = QMagma.EruptionState(); QMagma.init_eruption!(st2, ep, zc)
        ȧ_t, Δt_t, nstep = 1e-9, 1e11, 60
        QMagma.step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t;
            z_centroid=zc)
        erupted = 0.0
        for _ in 1:nstep
            QMagma.step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t;
                z_centroid=zc)
            erupted += st2.h_erupt
        end
        recharge = ȧ_t*Δt_t*nstep
        @test 0 < erupted <= 1.05recharge

        st2.P = st2.P_lith + 5ep.ΔP_crit
        QMagma.step_overpressure!(st2, ep, 1200.0+273.15, 0.8, V_e, ȧ_t, Δt_t;
            z_centroid=-30e3)
        @test st2.h_erupt == 0.0

        # A migrating chamber changes lithostatic pressure without jumping overpressure.
        st_m = QMagma.EruptionState()
        QMagma.update_lithostatic!(st_m, ep, -13e3)
        @test st_m.P ≈ st_m.P_lith
        st_m.P += 5e6
        QMagma.update_lithostatic!(st_m, ep, -14e3)
        @test st_m.P_lith ≈ ep.ρ_crust*ep.g*14e3
        @test st_m.P - st_m.P_lith ≈ 5e6
    end

    @testset "RK gas EOS (Huber 2010, item 1)" begin
        @test QMagma.rho_gas_RK(1e8, 1123.15) > QMagma.rho_gas_RK(1e8, 1173.15)  # hotter -> lighter
        @test QMagma.rho_gas_RK(3e8, 1123.15) > QMagma.rho_gas_RK(1e8, 1123.15)  # higher P -> denser
        @test QMagma.rho_gas_RK(3e8, 1123.15) > 0
        @test QMagma.rho_gas_RK(3e7, 1173.15) > 0                                 # positive over the box
        @test_throws DomainError QMagma.rho_gas_RK(2e7, 1100.0)
        @test_throws DomainError QMagma.rho_gas_RK(1e8, 1200.0)
    end

    @testset "Liu 2005 H₂O solubility (item 4)" begin
        # mass-fraction form (reference exsolve_silicic.m includes the 1e-2 wt%->fraction factor)
        meq(Pmpa) = 1e-2*((354.94*sqrt(Pmpa) + 9.623*Pmpa - 1.5223*Pmpa^1.5)/1200.0 + 1.2439e-3*Pmpa^1.5)
        @test 0.03 < meq(200.0) < 0.07     # ~5 wt% at 200 MPa / 1200 K
        @test meq(400.0) > meq(200.0)      # more soluble at higher P (monotone in range)
        # the ported Liu m_eq must be the one mixture_density actually uses
        ep = QMagma.EruptionParams()
        ρ, _ = QMagma.mixture_density(2e8, 1100.0, 1.0, ep; z_centroid=-5e3)
        @test isfinite(ρ) && ρ > 0
    end

    @testset "wall-T relaxation viscosity η_r (item 2a)" begin
        ep = QMagma.EruptionParams()
        @test QMagma.wall_relaxation_viscosity(ep, 500.0) > QMagma.wall_relaxation_viscosity(ep, 650.0)  # colder -> stiffer
        @test 1e17 <= QMagma.wall_relaxation_viscosity(ep, 500.0) <= 1e24                                 # in clamp range
        @test QMagma.wall_relaxation_viscosity(ep, 1200.0) == 1e17    # hot (mush-interior) wall clamps to the floor
        @test QMagma.wall_relaxation_viscosity(ep, 300.0)  == 1e24    # very cold wall clamps to the ceiling
    end

    @testset "H₂O speciation diagnostics on EruptionState" begin
        ep = QMagma.EruptionParams(m_w=0.05)
        # partition splits total water: dissolved + exsolved == m_w when saturated (X_g>0)
        m_diss, X_g, ρ_g, m_eq = QMagma.water_gas_partition(2e8, 1100.0, 0.7, ep;
            z_centroid=-5e3)
        @test m_diss ≈ m_eq*0.7
        @test isapprox(m_diss + X_g, ep.m_w; atol=1e-12)   # water conserved while gas-saturated
        @test ρ_g > 0
        # crystallizing (ϕ_melt↓) exsolves more gas -> X_g rises (second boiling)
        _, X_wet, _, _ = QMagma.water_gas_partition(2e8, 1100.0, 0.8, ep; z_centroid=-5e3)
        _, X_dry, _, _ = QMagma.water_gas_partition(2e8, 1100.0, 0.4, ep; z_centroid=-5e3)
        @test X_dry > X_wet
        # Shallow calls cannot extrapolate RK beyond its calibration box.
        @test_throws DomainError QMagma.water_gas_partition(1e9, 1100.0, 0.7, ep;
            z_centroid=-5e3)
        # The lower crust has no represented free-gas phase and never evaluates RK,
        # even for deliberately invalid RK pressure and temperature inputs.
        md, Xg, ρg, meq = QMagma.water_gas_partition(1e9, 1500.0, 0.7, ep;
            z_centroid=-11e3)
        @test (md, Xg, ρg) == (ep.m_w, 0.0, 0.0)
        @test isnan(meq)
        ρdeep, ϕgdeep = QMagma.mixture_density(1e9, 1500.0, 0.7, ep;
            z_centroid=-11e3)
        @test ρdeep ≈ 0.7*ep.ρ_melt + 0.3*ep.ρ_x
        @test ϕgdeep == 0.0
        # step_overpressure! must populate the diagnostics on the state. Use a shallow chamber
        # (~5 km, ~130 MPa) so the melt is water-saturated and X_g > 0; a deep chamber at high
        # lithostatic P keeps all water dissolved (X_g = 0), which is correct but trivial.
        st = QMagma.EruptionState(); QMagma.init_eruption!(st, ep, -5e3)
        QMagma.step_overpressure!(st, ep, 1100.0, 0.7, 1000.0, 1e-9, 1e10;
            z_centroid=-5e3)
        @test st.m_diss > 0 && st.ρ_gas > 0 && st.X_g > 0
        @test st.ϕ_mush == 0.7 && st.η_r == ep.η_r   # mush ϕ and wall viscosity recorded too
    end

    @testset "step_overpressure! nsub does not overflow Int64" begin
        # a soft (floored) η_r + big ΔP + large Δt makes the raw sub-step count ≫ typemax(Int64);
        # the clamp must happen in float space so ceil(Int, …) never sees the overflowing value
        ep = QMagma.EruptionParams(η_r=1e17, ΔP_crit=20e6)
        st = QMagma.EruptionState(); QMagma.init_eruption!(st, ep, -13e3)
        st.P = st.P_lith + 100*ep.ΔP_crit          # far over threshold -> huge dPdt0
        QMagma.step_overpressure!(st, ep, 1200.0+273.15, 0.8, 1000.0, 1e-9, 1e11;
            z_centroid=-13e3)  # init
        @test_nowarn QMagma.step_overpressure!(st, ep, 1200.0+273.15, 0.8, 1000.0, 1e-9, 1e11; z_centroid=-13e3)
    end

    @testset "D&H sub-grid withdrawals accumulate before booking" begin
        st = QMagma.EruptionState()
        @test QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time=1.0) == 0.0
        @test QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time=2.0) == 0.0
        h_realized = QMagma.pending_withdrawal!(st, 75.0, 1000.0, 100.0; time=3.0)
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
            st, 51.0)
    end

    @testset "V2: analytic overpressure relaxation" begin
        # constant properties, no exsolution (m_w=0) and no recharge (ȧ=0): the master ODE
        # reduces to dΔP/dt = -β_r·ΔP/η_r, i.e. exponential decay with τ = η_r/β_r.
        ep = QMagma.EruptionParams(m_w=0.0, β_r=1e10, η_r=1e18)
        τ  = ep.η_r/ep.β_r
        st = QMagma.EruptionState(P_lith=2e8, P=2e8)
        ΔP0 = 15e6; st.P = st.P_lith + ΔP0
        Δt = τ/2000; nstep = 2000
        QMagma.step_overpressure!(st, ep, 1200.0+273.15, 0.7, 100.0, 0.0, Δt;
            z_centroid=-13e3)  # first call inits
        for _ in 1:nstep
            QMagma.step_overpressure!(st, ep, 1200.0+273.15, 0.7, 100.0, 0.0, Δt;
                z_centroid=-13e3)
        end
        ΔP_num  = st.P - st.P_lith
        ΔP_exact = ΔP0*exp(-nstep*Δt/τ)                # t = nstep·Δt = τ
        @test isapprox(ΔP_num, ΔP_exact; rtol=2e-3)    # explicit Euler at Δt=τ/2000
    end

    @testset "column enthalpy drift across eruption closures (§2.1)" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=201, L=40e3, Geotherm=20.0,
                                                     Ttop=0.0, Tbot=800.0, Δt=200SecYear)
        # a hot mobile mush embedded in the geotherm
        T[(-17e3 .<= z .<= -13e3)] .= 1000.0
        rocks = zeros(length(z))
        MatParam = Params.MatParam
        H0 = QMagma.column_enthalpy(T, z, MatParam, Params.Phases)

        Tc, _ = QMagma.erupt_melt!(T, rocks, z; Erupt_z0=-15e3, Erupt_thick=800.0, method=:caldera)
        Te, _ = QMagma.erupt_melt!(T, rocks, z; Erupt_z0=-15e3, Erupt_thick=800.0, method=:elastic)
        Hc = QMagma.column_enthalpy(Tc, z, MatParam, Params.Phases)
        He = QMagma.column_enthalpy(Te, z, MatParam, Params.Phases)

        # both closures remove heat; caldera removes MORE (the full band leaves), elastic
        # re-covers the vent with stretched host rock at its own T -> smaller heat debit.
        # This quantifies the §2.1 energy-conservation gap of the intensive-T closures.
        @test H0 - Hc > 0
        @test H0 - He > 0
        @test H0 - Hc > H0 - He
    end

    @testset "cumulative enthalpy budget accounting" begin
        z = collect(-2.0:1.0:0.0)
        T = [2.0, 1.0, 0.0]
        @test QMagma.conductive_boundary_energy(T, [3.0, 3.0], z, 5.0) == 0.0
        @test QMagma.source_energy(fill(2.0, 3), z, 5.0) == 10.0

        budget = QMagma.EnthalpyBudget(100.0)
        QMagma.update_enthalpy_budget!(budget, 118.0;
            boundary=2.0, injected=10.0, source=10.0, erupted=4.0)
        snapshot = QMagma.enthalpy_budget_snapshot(budget)
        @test snapshot.storage_change == 18.0
        @test snapshot.residual == 0.0
        QMagma.update_enthalpy_budget!(budget, 120.0; boundary=2.0)
        @test budget.boundary == 4.0
        @test budget.residual == 0.0
    end

    @testset "magma and melt budget accounting" begin
        budget = QMagma.MassBudget(0.0, 100.0)

        # 200 m of magma in, 10 m of it displaced off the grid, all of it stored as grey:
        # the magma-volume budget closes. Only 60 m of it is still melt, so the other 130 m
        # The unresolved melt-content residual combines crystallization and host melting.
        QMagma.update_mass_budget!(budget, 190.0, 160.0; injected=200.0, withdrawn=10.0)
        snapshot = QMagma.mass_budget_snapshot(budget)
        @test snapshot.magma_change == 190.0
        @test snapshot.melt_change == 60.0
        @test snapshot.residual == 0.0
        @test snapshot.melt_residual == 140.0

        # an eruption debits both accounts, each with its own withdrawal: grey leaves with
        # the closure, melt leaves as the booked thickness
        QMagma.update_mass_budget!(budget, 150.0, 130.0; withdrawn=40.0, erupted=30.0)
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
        # §8.2 acceptance: with no eruptions the magma-volume budget closes to the
        # discretization of emplacing a sill on a fixed grid, while the melt budget must
        # stay strictly positive - it is the column's net crystallization.
        H, γ, Ttop = 40.0, 20.0, 0.0
        Δt = 200SecYear
        nz = 201
        Tsill, Sillthick, Sill_int_yr = 1200.0, 100.0, 500.0
        Silltop, Sillbot = 10.0, 20.0
        nt = 1500                                   # 300 kyr

        Params, BC, N, Δ, T, z = QMagma.init_model(nz=nz, L=H*1e3, Geotherm=γ, Ttop=Ttop,
                                                     Tbot=Ttop + H*γ, Δt=Δt)
        MatParam = Params.MatParam
        Params.Told .= T
        Δz = Δ[1]
        rocks = zero(T)

        Jac, colors = thermal_jacobian_workspace(nz)
        F = zero(T)

        compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
        budget = QMagma.MassBudget(QMagma.integrated_content(rocks, z),
            QMagma.melt_thickness(Params.ϕ, z, z[1], z[end]))

        ȧ = Sillthick/Sill_int_yr/SecYear
        time, A_inj = 0.0, 0.0
        rng = MersenneTwister(1234)
        for _ in 1:nt
            Δh = QMagma.injected_thickness(ȧ, time, Δt)
            n_injections = QMagma.sills_due(A_inj, Δh, Sillthick)
            A_inj += Δh
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params,
                                            MatParam=MatParam, verbose=false)
            injected_step, withdrawn_step = 0.0, 0.0
            for _ in 1:n_injections
                Sill_z0 = rand(rng, -Sillbot*1e3:1.0:-Silltop*1e3)
                T, rocks, h_out = QMagma.insert_sill(T, rocks, z; Sill_thick=Sillthick,
                                                      Sill_z0=Sill_z0, Sill_T=Tsill)
                injected_step += Sillthick
                withdrawn_step += h_out
                Params.Told .= T
            end
            Params.Told .= T
            compute_meltfraction!(Params.ϕ, MatParam, Params.Phases, (T = Params.Told .+ 273.15,))
            QMagma.update_mass_budget!(budget, QMagma.integrated_content(rocks, z),
                QMagma.melt_thickness(Params.ϕ, z, z[1], z[end]);
                injected=injected_step, withdrawn=withdrawn_step)
            time += Δt
        end

        @test budget.injected ≈ nt*Δt*ȧ
        @test budget.injected ≈ 600*Sillthick          # 300 kyr / 500 yr
        # Nothing reaches the surface at this injection depth.
        @test budget.withdrawn ≈ 0.0 atol=1e-6
        @test budget.magma ≈ budget.injected atol=1e-6
        @test budget.residual ≈ 0.0 atol=1e-6
        @test budget.melt_residual > 0
        @test budget.melt_residual < budget.injected
        @test budget.melt > 0
    end

    @testset "magma_heat_input" begin
        Params, = QMagma.init_model(nz=11, L=10e3, Ttop=0.0, Tbot=800.0, Δt=SecYear)
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
        Erupt_z0, Erupt_thick = -5e3, 1000.0
        half = Erupt_thick/2

        # :hybrid collapses the band onto the vent, like :elastic
        @test QMagma.erupt_displacement(Erupt_z0, Erupt_z0, Erupt_thick; method=:hybrid) == Erupt_z0

        # below the band both elastic laws decay toward the vent by less than a half-thickness
        below = Erupt_z0 - 3e3
        z_hyb = QMagma.erupt_displacement(below, Erupt_z0, Erupt_thick; method=:hybrid)
        @test below < z_hyb < below + half
        @test z_hyb ≈ QMagma.erupt_displacement(below, Erupt_z0, Erupt_thick; method=:elastic)

        # above the band :hybrid subsides more than :elastic but never more than the
        # rigid :caldera drop
        above = Erupt_z0 + 3e3
        z_hyb_up = QMagma.erupt_displacement(above, Erupt_z0, Erupt_thick; method=:hybrid)
        z_ela_up = QMagma.erupt_displacement(above, Erupt_z0, Erupt_thick; method=:elastic)
        @test z_hyb_up < z_ela_up
        @test z_hyb_up > above - Erupt_thick

        @test_throws "unknown eruption collapse method" QMagma.erupt_displacement(
            above, Erupt_z0, Erupt_thick; method=:nonsense)
    end

end
