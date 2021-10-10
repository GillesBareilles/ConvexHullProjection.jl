"""
    projection_zero(set, x0)

Compute the projection of zero on the given `set` starting from point `x0` in arbitrary type. Solved using a projected gradient accelerated by Riemannian Newton steps on the identified manifolds.

### Reference:
- Newton acceleration on manifolds identified by proximal-gradient methods
"""
function projection_zero(set::StructuredSet{Tf}, x0;
                         newtonaccel = true,
                         maxiter = 100,
                         showtrace = false,
                         printls = false,
                         ) where {Tf}
    x = similar(x0)
    repr = AmbRepr()

    M = prox_γg!(x, set, x0, repr)
    x_old = similar(x0)
    x_old .= x
    ∇fx = similar(x0)
    u = similar(x0)

    converged = false
    stopped = false
    L = Tf(1e-5)

    showtrace && @printf "     %.16e\n" f(set, x, repr)
    it = 0
    while !converged && !stopped
        ∇f!(∇fx, set, x, repr)

        ## backtracking procedure for proxgrad step
        M, L, it_btbeck = backtracked_proxgrad!(x, set, L, x_old, ∇fx, u, repr; printls)

        ## Logging step
        step = norm(x_old - x)
        normgradFₓ = gradᴹnorm(set, M, x, repr)
        showtrace && display_proxgradlog(set, x, repr, step, normgradFₓ, it, L, M)
        x_old .= x

        if newtonaccel
            ## Add Newton acceleration step
            # inds = identification_Newtonaccel!(x, set, M, repr)
            inds = newton_manifold!(x, set, M, repr)

            ##Logging
            step = norm(x_old - x)
            normgradᴹFx = gradᴹnorm(set, M, x, repr)
            showtrace && display_newtonsteplog(set, x, repr, step, normgradᴹFx, inds)
        end

        stopped = it > maxiter
        converged = normgradᴹFx < 1e2 * eps(Tf) || norm(x-x_old) < 1e2 * eps(Tf)

        x_old .= x
        it += 1
    end

    showtrace && display_finalpointlog(set, x, M, it, repr, converged, stopped)
    @debug "projection:" f(set, x, repr) gradᴹnorm(set, M, x, repr) M
    return x, M
end
