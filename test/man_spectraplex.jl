using ConvexHullProjection
using Test
using LinearAlgebra
using Random

const CHP = ConvexHullProjection

@testset "SymmPosSemidefFixedRankUnitTrace" begin
    r, k = 5, 3
    M = SymmPosSemidefFixedRankUnitTrace{r, k}()

    @testset "vec_to_lowrankmat $i" for i in 1:10
        Random.seed!(1643 + i)
        Y = randn(r, k)
        Z = Y * Y'
        z = vec(Z)

        Ybuilt = CHP.vec_to_lowrankmat(M, z)
        @test norm(z - CHP.lowrankmat_to_vec(CHP.vec_to_lowrankmat(M, z))) < 5e-14
    end

    @testset "tangent space projection" for i in 1:10
        Random.seed!(1643 + i)
        Y = randn(r, k)
        V = randn(r, k)

        Z = CHP.project(M, Y, V)
        @test norm(Z' * Y - Y' * Z) < 5e-14
        @test tr(Y' * Z) < 5e-14
    end

    @testset "retraction" for i in 1:10
        Random.seed!(1643 + i)
        Y = randn(r, k)
        V = randn(r, k)

        Vtan = CHP.project(M, Y, V)
        Zret = CHP.retract(M, Y, Vtan)

        Zmat = Zret * Zret'
        Λ = eigvals(Zmat)
        @test norm(min.(Λ, 0)) < 5e-14
        @test sum(Λ) ≈ 1.
        @test rank(Zmat) == k
    end
end
