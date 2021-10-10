"""
    $(SIGNATURES)

The set of points in ℝ^`n` which entries sum to one and are nonnull exactly when
the corresponding entry of `sparsitymask` is 1.
"""
struct SimplexFace
    sparsitymask::BitVector
end

manifold_dimension(M::SimplexFace) = sum(M.sparsitymask) - 1

"""
    $(SIGNATURES)

Orthogonally projects `d` on the tangent space to manifold `M` at point `x`.
"""
function project!(M::SimplexFace, res, x, d)
    res .= d .* M.sparsitymask
    resnz = @view res[M.sparsitymask]
    resnz .-= sum(resnz) / length(resnz)
    return res
end

function retract!(::SimplexFace, res, x, d)
    res .= x .+ d
    project_simplex!(res, res)
    return res
end

function ehess2rhess!(M::SimplexFace, res, x, ∇fₓ, ∇²fₓ, dₜ)
    project!(M, res, x, ∇²fₓ)
end
