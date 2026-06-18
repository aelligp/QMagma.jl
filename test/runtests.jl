using QMagma
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
