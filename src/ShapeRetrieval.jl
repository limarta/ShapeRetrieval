module ShapeRetrieval

using LinearAlgebra
using SparseArrays

# TODO: Standardize all operations to do column-major manipulations

include("mesh.jl")
include("geom.jl")
include("heat.jl")
include("load_obj.jl")
include("utils.jl")
include("viz.jl")


export load_obj
export to_mesh_jl

end # module ShapeRetrieval
