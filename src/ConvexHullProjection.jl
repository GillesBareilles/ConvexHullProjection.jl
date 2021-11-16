module ConvexHullProjection

using Infiltrator

using LinearAlgebra
using GenericLinearAlgebra
using GenericSchur
using LinearOperators
using LinearMaps
using IterativeSolvers
using Krylov
using LineSearches
using DocStringExtensions
using Manifolds

using Printf
using ConjugateGradient

using Zygote

import LinearAlgebra.lyap

const CHP = ConvexHullProjection

abstract type StructuredSet{Tf} end

abstract type PointRepr end
struct AmbRepr <: PointRepr end
struct QuotRepr <: PointRepr end


include("simplex/utils_simplexprojection.jl")

# Manifolds
include("manifold_generic.jl")
include("simplex/manifold_simplexface.jl")
include("spectraplex/manifold_spectraplexface.jl")

# Optim problems
include("simplex/projection_simplexshadow.jl")
include("spectraplex/projection_spectraplexshadow.jl")

# Optimization routines
include("utils_display.jl")
include("projectedgradient.jl")
include("newtonmanifold.jl")
include("optimize.jl")


export SimplexShadow, SimplexFace
export SpectraplexShadow, SymmPosSemidefFixedRankUnitTrace
export optimize

export CHP

end # module
