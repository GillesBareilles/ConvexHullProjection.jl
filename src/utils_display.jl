function display_proxgradlog(set, x, repr, step, normgradFₓ, it, L, structure, it_bt)
    @printf "%3i  %.16e %.3e  |gradf|: %.3e  it_bt: %.3i " it f(set, x, repr) step normgradFₓ it_bt
    s = @sprintf " L = %.1e  %s   \n" L structure
    printstyled(s, color=:green)
    return
end

function display_newtonsteplog(set, x, repr, step, normgradᴹFx, inds)
    @printf "     %.16e %.3e  |gradf|: %.3e" f(set, x, repr) step normgradᴹFx
    for (k, v) in pairs(inds)
        k in Set([:gradFnorm]) && continue
        @printf " %s: " k
        if isa(v, Integer)
            @printf "%i" v
        elseif isa(v, Real)
            @printf "%.1e" v
        else
            print(v)
        end
    end
    println()
    return
end

function display_finalpointlog(set, x, structure, it, repr, converged, stopped)
    println("Iterations     : ", it)
    println("Structure      : ", structure)
    println("f(x̄)           : ", f(set, x, repr))
    println("|gradᴹ f(x̄)|   : ", gradᴹnorm(set, structure, x, repr))
    println("converged      : ", converged)
    println("stopped        : ", stopped)
    return
end
