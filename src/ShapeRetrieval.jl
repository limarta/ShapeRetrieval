module ShapeRetrieval

using LinearAlgebra
using SparseArrays
using Arpack
using Flux

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x::Matrix; dims=1) = sqrt.(sum(x.^2, dims=dims))
function normalize_vectors(x::Matrix; dims=1)
    X = x ./ vec(norm(x, dims=dims))'
    X
end

include("mesh.jl")
include("load_obj.jl")
include("geom.jl")
include("heat.jl")
include("viz.jl")
include("utils.jl")
include("diffusion/diffusion.jl")

end # module ShapeRetrieval
