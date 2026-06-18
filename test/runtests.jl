using QMagma
using GeoParams
using LinearAlgebra, SparseArrays, SparseDiffTools
using Test

const SecYear = 3600 * 24 * 365.25

@testset "QMagma.jl" begin

    @testset "av" begin
        @test QMagma.av([1.0, 2.0, 4.0]) ≈ [1.5, 3.0]
        @test QMagma.av([0.0, 0.0]) ≈ [0.0]
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

        # a custom MatParam tuple should be used as-is
        Params2, = QMagma.init_model(nz=5, L=1e3, Δt=1SecYear, MatParam=Params.MatParam)
        @test Params2.MatParam === Params.MatParam
    end

    @testset "nonlinear_solution: steady linear geotherm is (near) a fixed point" begin
        Params, BC, N, Δ, T, z = QMagma.init_model(nz=21, L=10e3, Geotherm=10.0,
                                                     Ttop=0.0, Tbot=100.0, Δt=100SecYear)
        Params.Told .= T

        nz = N[1]
        J1 = Tridiagonal(ones(nz - 1), ones(nz), ones(nz - 1))
        J1[1, 2] = 0; J1[2, 1] = 0; J1[nz-1, nz] = 0; J1[nz, nz-1] = 0
        Jac = sparse(Float64.(abs.(J1) .> 0))
        colors = matrix_colors(Jac)
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

        nz = N[1]
        J1 = Tridiagonal(ones(nz - 1), ones(nz), ones(nz - 1))
        J1[1, 2] = 0; J1[2, 1] = 0; J1[nz-1, nz] = 0; J1[nz, nz-1] = 0
        Jac = sparse(Float64.(abs.(J1) .> 0))
        colors = matrix_colors(Jac)
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

        # host-rock advection velocity w: zero inside the zone, elastic-style decay outside
        # (mirrors the decay law used by discrete sill injection, crack_perp_displacement)
        ind_below = findall(z .< -Sillbot*1e3)
        ind_above = findall(z .> -Silltop*1e3)
        @test all(Params.w[ind_zone] .== 0)         # no relative displacement inside the zone itself
        @test all(Params.w[ind_below] .< 0)         # pushed down below the zone
        @test all(Params.w[ind_above] .> 0)         # pushed up above the zone
        @test issorted(abs.(Params.w[ind_below]))            # |w| grows approaching the zone from below (z increasing)
        @test issorted(abs.(Params.w[ind_above]), rev=true)  # |w| decays moving away from the zone above (z increasing)

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
        nzN   = N[1]
        J1    = Tridiagonal(ones(nzN - 1), ones(nzN), ones(nzN - 1))
        J1[1, 2] = 0; J1[2, 1] = 0; J1[nzN-1, nzN] = 0; J1[nzN, nzN-1] = 0
        Jac    = sparse(Float64.(abs.(J1) .> 0))
        colors = matrix_colors(Jac)
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam, verbose=false)
            if mod(time/SecYear, Sill_int_yr) == 0 && t > 1
                Sill_z0 = rand(-Sillbot*1e3:1:-Silltop*1e3)
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                Params.Told .= T
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam, verbose=false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        rel_diff = abs(maximum(T) - maximum(T_Q))/maximum(T)
        @test rel_diff < 0.01    # the two methods agree to within 1% in the many-sill limit
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
        nzN   = N[1]
        J1    = Tridiagonal(ones(nzN - 1), ones(nzN), ones(nzN - 1))
        J1[1, 2] = 0; J1[2, 1] = 0; J1[nzN-1, nzN] = 0; J1[nzN, nzN-1] = 0
        Jac    = sparse(Float64.(abs.(J1) .> 0))
        colors = matrix_colors(Jac)
        F = zero(T); F_Q = zero(T_Q)
        time = 0.0

        for t in 1:nt
            T, = QMagma.nonlinear_solution(F, T, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params, MatParam=MatParam, verbose=false)
            if mod(time/SecYear, Sill_int_yr) == 0 && t > 1
                Sill_z0 = rand(-Sillbot*1e3:1:-Silltop*1e3)
                T, rocks = QMagma.insert_sill(T, rocks, z, Sill_thick=Sillthick, Sill_z0=Sill_z0, Sill_T=Tsill)
                Params.Told .= T
            end
            Params.Told .= T

            QMagma.compute_Q_magma!(Params_Q, MatParam, z; Tsill=Tsill, ȧ=ȧ, Silltop=Silltop, Sillbot=Sillbot)
            QMagma.advect_w!(Params_Q)
            T_Q, = QMagma.nonlinear_solution(F_Q, T_Q, Jac, colors; Δ=Δ, N=N, BC=BC, Params=Params_Q, MatParam=MatParam, verbose=false)
            Params_Q.Told .= T_Q

            time += Δt
        end

        rel_diff = abs(maximum(T) - maximum(T_Q))/maximum(T)
        @test rel_diff < 0.01    # advection term keeps the two methods in close agreement
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

        # :constant displaces by exactly ±Sill_thick outside the sill
        tracers2 = [QMagma.Tracer(-10e3, 0.0, 0.0, 0, Float64[], Float64[])]
        QMagma.advect_tracers_sill!(tracers2, Sill_z0, Sill_thick; SillType=:constant)
        @test tracers2[1].z ≈ -10e3 + Sill_thick
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
        @test all(T2[ind] .≈ Sill_T)
        @test all(rocks2[ind] .== 1)
        @test sum(rocks2) > 0
    end

end
