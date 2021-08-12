using JuMP
using OSQP
using Test
using LinearAlgebra
using Random

using ConvexHullProjection
const CHP = ConvexHullProjection

@testset "Convex Hull projection" begin
    n = 10
    k = 5

    @testset "Simplex projection" for i in 1:10
        Random.seed!(1423 + i)
        x = 2 .* randn(n)
        set = ConvexHull(rand(n, k))

        # Compute explicit simplex projection
        res = similar(x)
        CHP.prox_γg!(res, set, x)

        # Solve same probelm with QP
        model = Model(optimizer_with_attributes(OSQP.Optimizer, "polish"=>true, "verbose"=>false, "max_iter"=>1e8, "time_limit"=>2, "eps_abs"=>1e-14, "eps_rel"=>1e-14))
        α = @variable(model, α[1:n])
        @objective(model, Min, dot(α - x, α - x))
        @constraint(model, sum(α) == 1)
        @constraint(model, α .>= 0)

        optimize!(model)
        @test termination_status(model) ∈ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL])

        # Check explicit computation is correct
        @test res ≈ value.(α)
    end


    @testset "projection on convexhull" for i in 1:10
        Random.seed!(1423 + i)
        set = ConvexHull(rand(n, k))

        # Compute projection of zero on set explicitly
        res = projection_zero(set, zeros(k))

        # Compute projection of zero on set with JuMP
        res_JuMP = projection_zero_JuMP(set)

        @test res ≈ res_JuMP atol = 1e-5 norm=x->maximum(abs.(x))

    end
end
