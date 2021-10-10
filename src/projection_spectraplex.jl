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
    norm(sum(σ) - Tf(1)) > tol && return Tf(Inf)

    return Tf(0)
end
function prox_γg!(res, sh::SpectraplexShadow{Tf}, X, ::QuotRepr) where Tf
    res .= X / norm(X)
    return SymmPosSemidefFixedRankUnitTrace{sh.r, 5}()
end



################################################################################
## Fallbacks
################################################################################
f(set, X, ::AmbRepr) = f(set, X)
∇f!(res, set, X, ::AmbRepr) = ∇f!(res, set, X)
∇²f!(res, set, X, D, ::AmbRepr) = ∇²f!(res, set, X, D)
g(set, X, ::AmbRepr) = g(set, X)
prox_γg!(res, set, X, ::AmbRepr) = prox_γg!(res, set, X)

∇f(sh::SpectraplexShadow, X) = ∇f(sh, X, AmbRepr())

function ∇f(sh::SpectraplexShadow, Z, repr)
    res = similar(Z)
    return ∇f!(res, sh, Z, repr)
end
function ∇²f(sh::SpectraplexShadow, Z, D, repr)
    res = similar(Z)
    return ∇²f!(res, sh, Z, D, repr)
end


gradᴹnorm(set, M, x) = gradᴹnorm(set, M, x, AmbRepr)

# function prox_γg!(res, sh::SpectraplexShadow{Tf}, X) where Tf
#     Λ, E = eigen(Symmetric(X*X'))

#     Λ_prox = zeros(Tf, size(Λ))
#     nnzentries = project_simplex!(Λ_prox, Λ)
#     # @show Λ

#     k = length(nnzentries)
#     manifold = SymmPosSemidefFixedRankUnitTrace{sh.r, k}()

#     for i in 1:k
#         res[:, i] .= sqrt(Λ_prox[end-i+1]) .* E[:, end-i+1]
#     end
#     res[:, k+1:end] .= 0

#     return manifold
# end


# function get_svd(X, k, Tf)
#     r = size(X, 1)
#     if norm(X) == 0.0
#         return Matrix{Tf}(I, r, r), zeros(Tf, r), Matrix{Tf}(I, r, r), r
#     end
#     Xnz = @view X[:, 1:k]

#     F = svd(Xnz)
#     @show k, norm(F.U'*F.U - I), norm(F.Vt*F.Vt' - I)
#     @show norm(F.U'*F.U - I) > 100*eps(Tf)
#     @show norm(F.Vt*F.Vt' - I) > 100*eps(Tf)
#     while norm(F.U'*F.U - I) > 100*eps(Tf) || norm(F.Vt*F.Vt' - I) > 100*eps(Tf)
#         k -= 1
#         Xnz = @view X[:, 1:k]
#         F = svd(Xnz)
#         @show k, norm(F.U'*F.U - I), norm(F.Vt*F.Vt' - I)
#         @infiltrate
#     end
#     @infiltrate

#     return F.U, F.S, F.Vt, k
# end




# function prox_γg!(res, sh::SpectraplexShadow{Tf}, X, ::SymmPosSemidefFixedRankUnitTrace{r, kin}) where {Tf, r, kin}
#     # U, S, Vt, kpract = get_svd(X, kin, Tf)
#     # printstyled("$kpract \n", color=:blue)
#     # Xnz = @view X[:, 1:kpract]
#     Xnz = @view X[:, 1:kin]
#     F = svd(Xnz)
#     U = F.U
#     S = F.S
#     Vt = F.Vt
#     # @show S

#     if norm(F.U'*F.U - I) > 200*eps(Tf) || norm(F.Vt*F.Vt' - I) > 200*eps(Tf)
#         if norm(F.U) < 200*eps(Tf) && norm(F.Vt - I) < 200*eps(Tf)
#             U = F.Vt
#         else
#             @error "Inconsistent svd" norm(F.U'*F.U - I) norm(F.Vt*F.Vt' - I)
#             # U, S, Vt, k = get_svd(X, kin, Tf)
#             # @infiltrate
#         end
#     end

#     S_prox = similar(S)
#     # nnzentries = project_simplex!(S_prox, map(t -> t > eps(Tf) ? t : Tf(0), S.^2))
#     nnzentries = project_simplex!(S_prox, S.^2)

#     # prune near zero entries of the thresholded singular values
#     # S_proxthresh = filter(t -> t > sqrt(eps(Tf)), S_prox)
#     S_proxthresh = S_prox
#     # @show nnzentries, S_prox

#     k = length(nnzentries)
#     k = min(length(nnzentries), length(S_proxthresh))
#     manifold = SymmPosSemidefFixedRankUnitTrace{sh.r, k}()

#     for i in 1:k
#         res[:, i] .= U[:, i] .* sqrt(S_proxthresh[i])
#     end
#     res[:, k+1:end] .= 0


#     # @show norm(U * Diagonal(S) * Vt - Xnz)
#     # @show norm(U * Diagonal(S.^2) * U' - X * X')
#     @infiltrate (Tf == BigFloat && norm(U * Diagonal(S.^2) * U' - X * X') > 1e-50)
#     # @debug "reconstruction error" norm(E*Diagonal(Λ_prox)*E' - res * res')
#     return manifold
# end



# function gradᴹnorm(set::SpectraplexShadow, M::SymmPosSemidefFixedRankUnitTrace{r, k}, x, ::AmbRepr) where {r, k}
#     x_man = mat_to_lowrankmat(M, x)
#     ∇fₓ = ∇f(set, x_man, QuotRepr())
#     gradFₓ = project(M, x_man, ∇fₓ)
#     return norm(gradFₓ)
# end




# function identification_Newtonaccel!(x, set::SpectraplexShadow{Tf}, M::SymmPosSemidefFixedRankUnitTrace{r, k}, ::AmbRepr) where {Tf, r, k}
#     x_man = mat_to_lowrankmat(M, x)

#     # Compute Riemannian gradient and hessian of f
#     ∇fₓ = ∇f(set, x_man, QuotRepr())

#     function HessFₓη!(res, d)
#         dₜ = project(M, x_man, d)
#         res .= ehess2rhess(M, x_man, ∇fₓ, ∇²f(set, x_man, dₜ, QuotRepr()), dₜ)
#         return res
#     end

#     gradFₓ = project(M, x_man, ∇fₓ)
#     gradnorm = norm(gradFₓ)

#     # Solve Newton's equation
#     tol = max(1e-3 * min(0.5, (gradnorm)^0.5) * gradnorm, eps(Tf))
#     # dᴺvec, stats = Krylov.lsmr(HessFₓop, -vec(gradFₓ), verbose=0, atol=tol, rtol=tol)
#     # dᴺ = dᴺvec
#     dᴺ, CGstats = solve_tCG(gradFₓ, HessFₓη!; ν=1e-15, ϵ_residual = tol, maxiter=2*r^2, printlev=0)

#     res = zeros(size(gradFₓ))
#     HessFₓη!(res, dᴺ)
#     res .+= gradFₓ


#     # Linesearch step
#     ls = BackTracking()
#     γ(t) = retract(M, x_man, t .* dᴺ)

#     function φ(t)
#         return f(set, γ(t), QuotRepr()) #+ g(set, γ(t), QuotRepr())
#     end

#     φ_0 = φ(0.0)
#     dφ_0 = dot(gradFₓ, dᴺ)
#     α, fx = ls(φ, 1.0, φ_0, dφ_0)

#     inds = (;
#             Symbol("|dᴺ|") => norm(dᴺ),
#             CGit = CGstats.iter,
#             α,
#             Symbol("CG") => CGstats.d_type,
#             Symbol("|Hessdᴺ+grad|") => norm(res),
#             GCtol = tol,
#             Symbol("θ") => dot(dᴺ, -gradFₓ) / (norm(dᴺ)*norm(gradFₓ) + eps(Tf)),
#     )

#     x_man .= retract(M, x_man, α .* dᴺ)
#     x .= x_man * x_man'

#     return inds
# end
