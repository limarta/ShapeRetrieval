using WGLMakie
export meshviz

# function connect(m::Vector{Int})
#     connect(tuple(m...), Triangle)
# end
# function to_mesh_jl(m::Mesh)
#     connections = vec(mapslices(connect, m.F, dims=[2]))
#     vertices =  [tuple(m.V[:,i]...) for i in 1:size(m.V,2)]
#     SimpleMesh(vertices, connections)
# end
function meshviz(mesh::Mesh; args...)
    V = mesh.V
    F = mesh.F
    N = mesh.normals
    X = face_centroids(mesh)
    println(size(N))
    fig = Figure(resolution = (1000,1000))
    ax = Axis3(fig[1,1], aspect=:data, elevation = 0.0, azimuth = -π/2)
    mesh!(ax, V', F'; args...)
    arrows!(X[1,:],X[2,:],X[3,:], N[1,:], N[2,:], N[3,:], linewidth=0.001, linecolor=:red, arrowcolor=:red, arrowsize=0.01, lengthscale=0.01)
    # TODO: Fix axis orientation
    # scene = Scene(fig.scene)
    # Makie.rotate!(scene, Makie.Vec3f(0,0,1), π/4)
	# ϕ = -π/4
	# v = Makie.Vec3f(300, 300, 0)
	# for p in scene.plots
	#     Makie.rotate!(p, Makie.Vec3f(0,0,1)3, ϕ)
	#     translate!(p, v)
	# end0
    fig
end
