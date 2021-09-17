"""
    $(SIGNATURES)

The manifold of symmetric positive semidefinite matrices of size `r`, rank `k`
and unit trace.

# Representation:
- Manifold points are represented as vectors of size `rk`
- Tangent vectors are represented as vectors of size `rk`.

# Reference:
- Journ\'ee, M., Bach, F., Absil, P., & Sepulchre, R. (2010). Low-rank
  optimization for semidefinite convex problems. SIAM Journal on Optimization,
  20(5), 2327–2351. http://dx.doi.org/10.1137/080731359
"""
struct SymmPosSemidefFixedRankUnitTrace{r, k} end



function vec_to_lowrankmat(::SymmPosSemidefFixedRankUnitTrace{r, k}, z::AbstractVector{Tf}) where {r, k, Tf}
    Z = reshape(z, (r, r))
    λs, E = eigen(Symmetric(Z))

    Y = zeros(Tf, r, k)
    for i in 1:k
        Y[:, i] .= sqrt(λs[end-i+1]) * E[:, end-i+1]
    end
    return Y
end
function lowrankmat_to_vec(Z)
    return vec(Z * Z')
end



@doc raw"""
    $(SIGNATURES)

Orthogonally project `d` on the tangent space of the current manifold at point `p`
and stores the result in `res`.

````math
\operatorname{proj}_{Y}(Z) = Z - Y\Omega - \frac{\langle Y, Z \rangle}{\langle Y, Y\rangle}
````
where $\Omega$ is a skew-symmetric matrix solving the following Lyapunov
equation:
````math
\Omega Y^\top Y + Y^\top Y \Omega = Y^\top Z - Z^\top Y.
````
"""
function project!(::SymmPosSemidefFixedRankUnitTrace{r, k}, res, Y, Z) where {r, k}
    res .= Z .- Y * lyap(Y' * Y, Z' * Y - Y' * Z) .- (tr(Z' * Y) / tr(Y' * Y)) .* Y
    return res
end

function project(M::SymmPosSemidefFixedRankUnitTrace{r, k}, p, v) where {r, k}
    res = similar(p)
    project!(M, res, p, v)
    return res
end



@doc raw"""
    $(SIGNATURES)

Compute the projection retraction of vector `d` at point `p`
"""
function retract(M::SymmPosSemidefFixedRankUnitTrace{r, k}, P, D) where {r, k}
    # P = vec_to_lowrankmat(M, p)
    # D = vec_to_lowrankmat(M, d)

    # display(P)
    # display(D)

    retrD = P .+ D
    retrD ./= sqrt(tr(retrD' * retrD))
    return retrD
end



function build_tangentprojoperator(manifold::SymmPosSemidefFixedRankUnitTrace{r, k}, z) where {r, k}
    Tf = typeof(first(z))

    # NOTE: a @closure might do some good here
    function mul!(res, v, α, β::T) where T
        if β == zero(T)
            res .= α .* project(manifold, z, v)
        else
            res .= α .* project(manifold, z, v) .+ β .* res
        end
    end

    return LinearOperator(Tf, r^2, r^2, true, true, mul!)
end
