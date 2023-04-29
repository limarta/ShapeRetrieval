module ShapeRetrieval

using LinearAlgebra
using SparseArrays

# TODO: Standardize all operations to do column-major manipulations
mutable struct Mesh
    V::Matrix{Float64}
    F::Matrix{Int}
    normals::Matrix{Float64}
end

include("geom.jl")
include("heat.jl")
include("load_obj.jl")
include("viz.jl")


export load_obj
export to_mesh_jl

end # module ShapeRetrieval
