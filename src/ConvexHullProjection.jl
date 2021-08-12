module ConvexHullProjection

using LinearAlgebra
using LinearOperators
using Krylov
using LineSearches

greet() = print("Hello World!")

include("projection_convexhull.jl")
include("projection.jl")

export ConvexHull
export projection_zero

end # module
