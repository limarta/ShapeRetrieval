module ShapeRetrieval

using LinearAlgebra
using SparseArrays
using Arpack
using DataStructures
using LogExpFunctions

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x::Matrix; dims=1) = sqrt.(sum(x.^2, dims=dims))
normalize_vectors(x::Matrix; dims=1) = x ./ vec(norm(x, dims=dims))'
sigmoid(x) = 1 / (1+exp(-x))
entropy(P) = -sum(P .* log2.(P))
mean(A) = sum(A)/ length(A)

include("mesh.jl")
include("geom.jl")
include("heat.jl")
include("diffusion/diffusion.jl")
include("utils.jl")
include("dataloader/dataloader.jl")
include("functional_maps.jl")
include("sampler.jl")
include("viz.jl")

end # module ShapeRetrieval
