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

abstract type StructuredSet{Tf} end

abstract type PointRepr end
struct AmbRepr <: PointRepr end
struct QuotRepr <: PointRepr end


include("utils_simplexprojection.jl")
include("manifold_generic.jl")

include("utils_display.jl")
include("projectedgradient.jl")
include("newtonmanifold.jl")

include("manifold_simplexface.jl")
include("projection_convexhull.jl")

include("manifold_spectraplexface.jl")
include("projection_spectraplex.jl")

include("projection.jl")

export ConvexHull
export SimplexFace
export SpectraplexShadow, SymmPosSemidefFixedRankUnitTrace
export projection_zero

end # module
