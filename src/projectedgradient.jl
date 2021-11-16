"""
    $(SIGNATURES)

Perform a backtacked proximal gradient step, following procedure `B1`, section
10.3 from Beck's book.

## Reference:
- Beck, A. (2017). First-Order Methods in Optimization. Philadelphia, PA:
  Society for Industrial and Applied Mathematics.
"""
function backtracked_proxgrad!(x, set, L, x_old, ∇fx, u, repr; maxiter = 100, showls = false)
    Tf = eltype(x)

    s = 1e-10
    η = 2
    γᵇ = 1e-4

    u .= x .- (1/L) .* ∇fx
    structure = prox_γg!(x, set, u, repr)
    it_btbeck = 0

    fxold = f(set, x_old, repr)
    fx = f(set, x, repr)
    Δf = fxold - fx
    Δx = norm(x_old - x)

    showls && @printf "*it   f(x)            f(Tᴸ(x))           f(x)-f(Tᴸ(x)) γL|Tᴸ(x)-x|²      L          |Tᴸ(x)-x|²  M\n"
    showls && @printf "*%3i  %.8e  %.8e        % .3e %.3e         %.3e  %.3e   %s\n" it_btbeck fxold fx Δf γᵇ * L * Δx^2 L Δx^2 structure
    while Δf < γᵇ * L * Δx^2 - 10*eps(eltype(x))
        # the above eltype is meant to stop the linesearch when the values are equal up to machine precision.
        # This happens generally when they are equal to zero.
        L *= η
        u .= x_old .- (1/L) .* ∇fx
        structure = prox_γg!(x, set, u, repr)

        fx = f(set, x, repr)
        Δf = fxold - fx
        Δx = norm(x_old - x)

        it_btbeck += 1
        showls && @printf "*%3i  %.8e  %.8e        % .3e %.3e         %.3e  %.3e   %s\n" it_btbeck fxold fx Δf γᵇ * L * Δx^2 L Δx^2 structure

        if it_btbeck > maxiter
            @warn "Many backtracking steps here"
            break
        end
    end

    return structure, L, it_btbeck
end
