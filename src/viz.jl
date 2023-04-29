import Meshes: Meshes, connect, Triangle
using Meshes:SimpleMesh

# function connect(m::Vector{Int})
#     connect(tuple(m...), Triangle)
# end
# function to_mesh_jl(m::Mesh)
#     connections = vec(mapslices(connect, m.F, dims=[2]))
#     vertices =  [tuple(m.V[:,i]...) for i in 1:size(m.V,2)]
#     SimpleMesh(vertices, connections)
# end