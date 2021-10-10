"""
    $(TYPEDSIGNATURES)

Projects α on the unit simplex.

### Reference:
- Fast Projection onto the Simplex and the ℓ1 Ball, L. Condat, alg. 1
"""
function project_simplex!(res::Vector{Tf}, α::Vector{Tf}) where Tf
    N = length(α)

    # 1. Sorting
    u = sort(α, rev=true)

    # 2.
    k = 1
    sum_u_1k = u[1]
    while (k < N) && (sum_u_1k + u[k+1] - 1) / (k+1) < u[k+1]
        k += 1
        sum_u_1k += u[k]
    end

    # 3.
    τ = (sum_u_1k - 1) / k

    res .= @. max(α - τ, Tf(0))
    return SimplexFace(res .!= Tf(0))
end
