using WGLMakie
# export meshviz

# TODO: Fix axis orientation
# scene = Scene(fig.scene)
# Makie.rotate!(scene, Makie.Vec3f(0,0,1), π/4)
# ϕ = -π/4
# v = Makie.Vec3f(300, 300, 0)
# for p in scene.plots
#     Makie.rotate!(p, Makie.Vec3f(0,0,1)3, ϕ)
#     translate!(p, v)
# end
function meshviz(mesh::Mesh; args...)
    V = mesh.V
    F = mesh.F
    fig = Figure(resolution = (2000,2000))
    ax = Axis3(fig[1,1], aspect=:data, elevation = 0.0, azimuth = -π/2)
    mesh!(ax, V', F'; args...)
    if get(args,:viz_field, false)
        if !haskey(args, :field)
            field = mesh.normals
            field_type = :face
        else
            field = args[:field]
            field_type = args[:field_type]
        end
        viz_field!(mesh, field, type=field_type)
    end
    fig
end

function viz_field!(mesh::Mesh, field; type::Symbol)
    # Face-based vector field
    if type == :face
        X = face_centroids(mesh)
    elseif type == :vertex
        X = mesh.V
    end
    arrows!(X[1,:],X[2,:],X[3,:], field[1,:], field[2,:], field[3,:], linewidth=0.001, linecolor=:red, arrowcolor=:red, arrowsize=0.01, lengthscale=0.01)
end