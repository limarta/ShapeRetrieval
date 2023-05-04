module ShapeRetrieval

using LinearAlgebra
using SparseArrays
using Arpack
using Flux

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x::Matrix; dims=1) = sqrt.(sum(x.^2, dims=dims))
normalize_vectors(x::Matrix; dims=1) = x ./ vec(norm(x, dims=dims))'

include("mesh.jl")
include("geom.jl")
include("heat.jl")
include("diffusion/diffusion.jl")
include("utils.jl")
include("viz.jl")

end # module ShapeRetrieval
