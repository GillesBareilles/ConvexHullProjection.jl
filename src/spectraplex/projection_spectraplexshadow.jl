raw"""
    SpectraplexShadow{Tf}

Models the problem of projecting zero on a shadow of the spectraplex as
\min 0.5 * || A Z ||^2, s.t. Z \in \mathcal S_r, Z \succeq 0, tr(Z) = 1,
using the Frobenius norm.
"""
struct SpectraplexShadow{Tf, Tm} <: StructuredSet{Tf}
    As::Vector{Tm}
    r::Int64
    function SpectraplexShadow(Tf, As::Vector{Tm}, r) where Tm
        if typeof(first(As[1])) != Tf
            throw(error("Floating precision type inconsistent with shadow vectors"))
        end
        return new{Tf, Tm}(As, r)
    end
end

SpectraplexShadow(As::Vector, r) = SpectraplexShadow(typeof(first(As[1])), As, r)




################################################################################
## Ambient representation
################################################################################
function f(set::SpectraplexShadow, Z, ::AmbRepr)
    return 0.5 * sum(dot(A, Z)^2 for A in set.As)
end

function ∇f!(res, sh::SpectraplexShadow, Z, ::AmbRepr)
    res .= 0
    for A in sh.As
        res .+= dot(A, Z) .* A
    end
    return res
end

"""
    $(SIGNATURES)

Indicator function of the spectraplex set.
"""
function g(::SpectraplexShadow{Tf}, X, ::AmbRepr) where Tf
    tol = 50 * eps(Tf)

    Λ = eigvals(Symmetric(X*X'))

    norm(sum(Λ) - Tf(1)) > tol && return Tf(Inf)
    norm(min.(Λ, 0)) > tol && return Tf(Inf)

    return Tf(0)
end

function prox_γg!(res, sh::SpectraplexShadow{Tf}, Z, ::AmbRepr) where Tf
    Λ, E = eigen(Symmetric(Z))

    Λ_prox = zeros(Tf, size(Λ))
    M = project_simplex!(Λ_prox, Λ)
    rank = sum(M.sparsitymask)

    res .= E * Diagonal(Λ_prox) * E'
    return SymmPosSemidefFixedRankUnitTrace{sh.r, rank}()
end


################################################################################
## Quotient representation
################################################################################
function f(sh::SpectraplexShadow, X, ::QuotRepr)
    return 0.5 * sum(dot(A, X * X')^2 for A in sh.As)
end

function ∇f!(res, sh::SpectraplexShadow, X, ::QuotRepr)
    res .= 0
    for A in sh.As
        res .+= 2*dot(A, X*X') .* A*X
    end
    return res
end

function ∇²f!(res, sh::SpectraplexShadow, X, D, ::QuotRepr)
    res .= 0
    for A in sh.As
        res .+= 4*dot(A, X*D') .* A*X + 2*dot(A, X*X') .* A*D
    end
    return res
end

function g(::SpectraplexShadow{Tf}, X, ::QuotRepr) where Tf
    tol = 50 * eps(Tf)

    σ = svdvals(X)
    norm(sum(σ.^2) - Tf(1)) > tol && return Tf(Inf)

    return Tf(0)
end
function prox_γg!(res, sh::SpectraplexShadow{Tf}, X, ::QuotRepr) where Tf
    res .= X / norm(X)
    return SymmPosSemidefFixedRankUnitTrace{sh.r, 5}()
end
