function gradᴹnorm(set, M, x, ::AmbRepr)
    x_man = amb_to_manifold_repr(M, x)
    manrepr = manifold_repr(M)

    ∇fₓ = ∇f(set, x_man, manrepr)
    gradFₓ = project(M, x_man, ∇fₓ)
    return norm(gradFₓ)
end


function retract(M, x, d)
    res = similar(x)
    retract!(M, res, x, d)
    return res
end

function project(M, x, d)
    res = similar(d)
    project!(M, res, x, d)
    return res
end


function ∇f(set, x, kwargs...)
    res = similar(x)
    ∇f!(res, set, x, kwargs...)
    return res
end

function ∇²f(set, x, d, kwargs...)
    res = similar(d)
    ∇²f!(res, set, x, d, kwargs...)
    return res
end
