using JuMP
using MosekTools

"""
    projection_zero_JuMP(set::ConvexHull{Float64})

Compute the projection of zero on the given convex hull by formulating a linear SOCP
problem in JuMP and solving with Mosek.
"""
function projection_zero_JuMP(set::ConvexHull{Float64})
    n, k = size(set.gs)
    QUIET = true
    model = Model(with_optimizer(Mosek.Optimizer;
        QUIET,
        MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-12,
        MSK_DPAR_INTPNT_CO_TOL_INFEAS=1e-12,
        MSK_DPAR_INTPNT_CO_TOL_MU_RED=1e-12,
        MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-12,
        MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-12))

    α = @variable(model, α[1:k])
    η = @variable(model, η)
    gconvhull = sum(α[i] .* set.gs[:, i] for i in 1:k)
    @objective(model, Min, η)
    socpctr = @constraint(model, vcat(η, gconvhull) in SecondOrderCone())

    @constraint(model, sum(α) == 1)
    cstr_pos = @constraint(model, α .>= 0)

    optimize!(model)

    if termination_status(model) ∉ Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
        @debug "projection_zero_JuMP: subproblem was not solved to optimality" termination_status(model) primal_status(model) dual_status(model)
    end

    return value.(α)
end

include("projection.jl")
