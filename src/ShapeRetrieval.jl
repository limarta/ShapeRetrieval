module ShapeRetrieval

using LinearAlgebra
using SparseArrays
using Arpack

# TODO: Standardize all operations to do column-major manipulations

include("mesh.jl")
include("load_obj.jl")
include("utils.jl")
include("geom.jl")
include("heat.jl")
include("viz.jl")


export to_mesh_jl

end # module ShapeRetrieval
