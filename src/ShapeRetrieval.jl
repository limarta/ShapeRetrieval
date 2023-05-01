module ShapeRetrieval

using LinearAlgebra
using SparseArrays
using Arpack

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x::Matrix; dims=1) = sqrt.(sum(x.^2, dims=dims))

include("mesh.jl")
include("load_obj.jl")
include("utils.jl")
include("geom.jl")
include("heat.jl")
include("viz.jl")

end # module ShapeRetrieval
