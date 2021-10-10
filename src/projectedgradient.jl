"""
    $(SIGNATURES)

Perform a backtacked proximal gradient step, following procedure `B1`, section
10.3 from Beck's book.

## Reference:
- Beck, A. (2017). First-Order Methods in Optimization. Philadelphia, PA:
  Society for Industrial and Applied Mathematics.
"""
function backtracked_proxgrad!(x, set, L, x_old, ∇fx, u, repr; maxiter = 100, printls = false)
    s = 1e-10
    η = 1.2
    γᵇ = 1e-4

    u .= x .- (1/L) .* ∇fx
    structure = prox_γg!(x, set, u, repr)
    it_btbeck = 0
    printls && @printf "*it   f(x)            f(Tᴸ(x))           f(x)-f(Tᴸ(x)) γL|Tᴸ(x)-x|²      L          |Tᴸ(x)-x|²  M\n"
    printls && @printf "*%3i  %.8e  %.8e        % .3e %.3e         %.3e  %.3e   %s\n" it_btbeck f(set, x_old, repr) f(set, x, repr) f(set, x_old, repr) - f(set, x, repr) γᵇ * L * norm(x_old - x)^2 L norm(x_old - x)^2 structure
    while f(set, x_old, repr) - f(set, x, repr) ≤ γᵇ * L * norm(x_old - x)^2
        L *= η
        u .= x_old .- (1/L) .* ∇fx
        structure = prox_γg!(x, set, u, repr)
        it_btbeck += 1
        printls && @printf "*%3i  %.8e  %.8e        % .3e %.3e         %.3e  %.3e   %s\n" it_btbeck f(set, x_old, repr) f(set, x, repr) f(set, x_old, repr) - f(set, x, repr) γᵇ * L * norm(x_old - x)^2 L norm(x_old - x)^2 structure
        it_btbeck > maxiter && @assert false
    end

    return structure, L, it_btbeck
end
