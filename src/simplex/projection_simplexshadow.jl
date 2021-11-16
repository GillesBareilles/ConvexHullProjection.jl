raw"""
    SimplexShadow{Tf}

Models the convex hull of vectors gs.
"""
struct SimplexShadow{Tf} <: StructuredSet{Tf}
    A::Matrix{Tf}
end


f(ch::SimplexShadow, α, ::AmbRepr) = 0.5 * norm(ch.A * α)^2

∇f!(res, ch::SimplexShadow, α, ::AmbRepr) = (res .= ch.A' * ch.A * α)

∇²f!(res, ch::SimplexShadow, x, d, ::AmbRepr) = (res .= ch.A' * ch.A * d)

g(::SimplexShadow, α) = sum(α) == 1 && sum(α .>= 0) == length(α)


"""
    prox_γg!(res, ch, α)

Computes the prox of the indicator of the simplex, which amounts to projecting onto the simplex.

"""
function prox_γg!(res, ::SimplexShadow{Tf}, α, ::AmbRepr) where Tf
    M = project_simplex!(res, α)
    return M
end

