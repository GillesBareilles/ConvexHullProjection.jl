raw"""
    ConvexHull{Tf}

Models the convex hull of vectors gs.
"""
struct ConvexHull{Tf} <: StructuredSet{Tf}
    gs::Matrix{Tf}
end


f(ch::ConvexHull, α) = 0.5 * norm(ch.gs * α)^2
∇f!(res, ch::ConvexHull, α) = (res .= ch.gs' * ch.gs * α)


g(::ConvexHull, α) = sum(α) == 1 && sum( α .>= 0) == length(α)

function ∇f(ch::ConvexHull, α)
    res = similar(α)
    ∇f!(res, ch, α)
    return res
end

∇f(ch::ConvexHull, α, repr) = ∇f(ch, α)
∇²f(ch::ConvexHull, x, d, repr) = (ch.gs' * ch.gs * d)

"""
    prox_γg!(res, ch, α)

Implement the prox of the indicator of the simplex, which amounts to projecting onto the simplex.

"""
function prox_γg!(res, ::ConvexHull{Tf}, α) where Tf
    M = project_simplex!(res, α)
    return M
end

function form_projection(set::ConvexHull, x)
    return set.gs * x
end

function get_activities(::ConvexHull, x)
    return filter(i -> x[i] == 0, 1:length(x))
end


# """
#     identification_Newtonaccel!(x, set::ConvexHull)

# Identifies the structure (a manifold) of `x` and performs a linesearch along the Riemannian Newton step on that structure.

# *Notes:*
# - that exact identification is possible only since `x` is the output of an *exact* proximity operator, here the projection on the simplex.
# - this function is valid for arbitrary real types, including `BigFloat`.
# """
# function identification_Newtonaccel!(x, set::ConvexHull{Tf}, structure, ::AmbRepr) where Tf
#     # Identify active manifold
#     minusgradfx = -∇f(set, x)

#     k = size(set.gs, 2)
#     nnnzentries = length(structure)
#     nnzentries = zeros(Bool, k)
#     for i in structure
#         nnzentries[i] = true
#     end

#     function projtan!(res, v, α, β::T) where T
#         sumvalsv = sum(v[nnzentries])
#         if β != zero(T)
#             res[nnzentries] .= α .* (v[nnzentries] .- sumvalsv / nnnzentries) .+ β .* v[nnzentries]
#             res[.!(nnzentries)] .= β .* v
#         else
#             res[nnzentries] .= α .* (v[nnzentries] .- sumvalsv / nnnzentries)
#             res[.!(nnzentries)] .= zero(T)
#         end
#     end

#     tangentproj = LinearOperator(Tf, k, k, true, true, projtan!)
#     ∇²f = LinearOperator(Tf, k, k, true, true,
#                          (res, v, α, β) -> ∇²f!(res, set, v, α, β))

#     # Compute Riemannian gradient and hessian of f
#     ∇f!(minusgradfx, set, x)
#     minusgradfx .*= -1
#     projtan!(minusgradfx, minusgradfx, one(Tf), zero(Tf))
#     Hessfx = tangentproj * ∇²f * tangentproj

#     # Solve Newton's equation
#     dᴺ, stats = Krylov.lsmr(Hessfx, minusgradfx)

#     # Linesearch step
#     ls = BackTracking()
#     γ(t) = x .+ t .* dᴺ
#     φ(t) = f(set, γ(t)) + g(set, γ(t))

#     φ_0 = φ(0.0)
#     dφ_0 = -dot(minusgradfx, dᴺ)
#     α, fx = ls(φ, 1.0, φ_0, dφ_0)

#     x .+= α .* dᴺ

#     @debug "ConvexHull projection - Newton acceleration step" nnnzentries norm(Hessfx * dᴺ - minusgradfx) norm(dᴺ) dot(dᴺ, minusgradfx) / (norm(dᴺ)*norm(minusgradfx)) α
#     return inds = (;)
# end
