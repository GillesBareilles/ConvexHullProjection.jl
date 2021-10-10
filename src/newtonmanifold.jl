function amb_to_manifold_repr(M, x)
    return x
end
function manifold_to_amb_repr(M, x)
    return x
end

manifold_repr(M) = AmbRepr()


function newton_manifold!(x, set::StructuredSet{Tf}, M, ::AmbRepr) where {Tf}
    x_man = amb_to_manifold_repr(M, x)
    manrepr = manifold_repr(M)

    # Compute Riemannian gradient and hessian of f
    ∇fₓ = ∇f(set, x_man, manrepr)

    function HessFₓη!(res, d)
        dₜ = project(M, x_man, d)
        ehess2rhess!(M, res, x_man, ∇fₓ, ∇²f(set, x_man, dₜ, manrepr), dₜ)
        return res
    end

    gradFₓ = project(M, x_man, ∇fₓ)
    gradnorm = norm(gradFₓ)

    # Solve Newton's equation
    tol = max(1e-3 * min(0.5, (gradnorm)^0.5) * gradnorm, eps(Tf))
    dᴺ, CGstats = solve_tCG(gradFₓ, HessFₓη!; ν=1e-15, ϵ_residual = tol, maxiter=2*manifold_dimension(M), printlev=0)

    res = zeros(size(gradFₓ))
    HessFₓη!(res, dᴺ)
    res .+= gradFₓ

    # Linesearch step
    ls = BackTracking()
    γ(t) = retract(M, x_man, t .* dᴺ)

    function φ(t)
        return f(set, γ(t), manrepr) #+ g(set, γ(t), manrepr)
    end

    φ_0 = φ(0.0)
    dφ_0 = dot(gradFₓ, dᴺ)
    α, fx = ls(φ, 1.0, φ_0, dφ_0)

    inds = (;
            Symbol("|dᴺ|") => norm(dᴺ),
            CGit = CGstats.iter,
            α,
            Symbol("CG") => CGstats.d_type,
            Symbol("|Hessdᴺ+grad|") => norm(res),
            GCtol = tol,
            Symbol("θ") => dot(dᴺ, -gradFₓ) / (norm(dᴺ)*norm(gradFₓ) + eps(Tf)),
    )

    x_man .= retract(M, x_man, α .* dᴺ)
    x .= manifold_to_amb_repr(M, x_man)
    return inds
end
