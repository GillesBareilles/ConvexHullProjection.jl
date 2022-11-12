raw"""
    SimplexShadow{Tf}

Models the convex hull of vectors contained in `P`, with shift `a`.
Corresponds to minimizing $0.5 * |P * x|^2 + ⟨a, x⟩$ with $x$ in the simplex.
"""
struct SimplexShadow{Tf} <: StructuredSet{Tf}
    P::Matrix{Tf}
    a::Vector{Tf}
end


################################################################################
## Ambient representation
################################################################################
f(ch::SimplexShadow, α, ::AmbRepr) = 0.5 * norm(ch.P * α)^2 + dot(ch.a, α)

∇f!(res, ch::SimplexShadow, α, ::AmbRepr) = (res .= ch.P' * ch.P * α .+ ch.a)

∇²f!(res, ch::SimplexShadow, x, d, ::AmbRepr) = (res .= ch.P' * ch.P * d)

g(::SimplexShadow, α, ::AmbRepr) = sum(α) == 1 && sum(α .>= 0) == length(α)

function prox_γg!(res, ::SimplexShadow{Tf}, α, ::AmbRepr) where Tf
    M = project_simplex!(res, α)
    return M
end

