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


function manifold_dimension(::SymmPosSemidefFixedRankUnitTrace{r, k}) where {r, k}
    # NOTE: not sure about this...
    return r*k - 1
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
    if size(Y) != (r, k) || size(Z) != (r, k)
        throw(error("Input point and vector sizes should be ($k, $r) but are $(size(Y)), $(size(Z))."))
    end
    Ω = zeros(k, k)
    try
        Ω = lyap(Y' * Y, Z' * Y - Y' * Z)
    catch e
        @warn "Failed to solve lyapunov system" e r k
        display(Y)
    end

    res .= Z .- Y * Ω .- (tr(Z' * Y) / tr(Y' * Y)) .* Y
    return res
end




@doc raw"""
    $(SIGNATURES)

Compute the projection retraction of vector `d` at point `p`
"""
function retract!(::SymmPosSemidefFixedRankUnitTrace{r, k}, res, P, D) where {r, k}
    if size(P) != (r, k) || size(D) != (r, k)
        throw(error("Input point and vector sizes are incompatible with quotient representation."))
    end
    res .= P .+ D
    res ./= sqrt(tr(res' * res))
    return res
end

function ehess2rhess!(M::SymmPosSemidefFixedRankUnitTrace{r, k}, res, x, ∇fₓ, ∇²fₓξ, ξ) where {r, k}
    if size(x) != (r, k) || size(ξ) != (r, k)
        throw(error("Input point and vector sizes are incompatible with quotient representation."))
    end

    Ω = lyap(x'*x, x'*∇fₓ-∇fₓ'*x)

    rhs = ξ'*∇fₓ-∇fₓ'*ξ + x'*∇²fₓξ-∇²fₓξ'*x - Ω*(x'*ξ+ξ'*x)-(x'*ξ+ξ'*x)*Ω
    Ωᴰ = lyap(x'*x, rhs)

    res .= ∇²fₓξ -ξ*Ω-x*Ωᴰ-tr(ξ'*∇fₓ)*x-tr(x'*∇²fₓξ)*x-tr(x'*∇fₓ)*ξ

    project!(M, res, x, res)
    return res
end


################################################################################
### Utilities
################################################################################

function mat_to_lowrankmat(::SymmPosSemidefFixedRankUnitTrace{r, k}, Z::AbstractMatrix{Tf}) where {r, k, Tf}
end




function amb_to_manifold_repr(::SymmPosSemidefFixedRankUnitTrace{r, k}, x::AbstractMatrix{Tf}) where {r, k, Tf}
    λs, E = eigen(Symmetric(x))
    λs .= max.(λs, 0)

    Y = zeros(Tf, r, k)
    for i in 1:k
        Y[:, i] .= sqrt(λs[end-i+1]) * E[:, end-i+1]
    end
    return Y
end
function manifold_to_amb_repr(::SymmPosSemidefFixedRankUnitTrace, x)
    return x * x'
end
manifold_repr(::SymmPosSemidefFixedRankUnitTrace) = QuotRepr()


"""
    $(SIGNATURES)

Computes the solution `X` to the continuous Lyapunov equation `AX + XA' + C = 0`
for arbitraty type, based on the `IterativeSolvers.idrs` routine.
"""
function lyap(A::AbstractMatrix{Tf}, C::AbstractMatrix{Tf}) where {Tf <: Real}
    r = size(A, 1)
    if size(A) != (r, r) || size(C) != (r, r)
        throw(error())
    end

    function lyapop(res::Vector, x::Vector)
        X = reshape(x, (r, r))
        res .= vec(A*X)
        res .+= vec(X*A')
        return
    end

    O = LinearMap{Tf}(lyapop, r^2, r^2, issymmetric=true)
    return reshape(idrs(O, -vec(C)), (r, r))
end
