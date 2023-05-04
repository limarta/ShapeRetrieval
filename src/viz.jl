using WGLMakie

function meshviz(mesh::Mesh; args...)
    V = mesh.V
    F = mesh.F
    fig = Figure(resolution = (900,900))
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

function viz_field!(mesh::Mesh, field; field_type::Symbol, kwargs...)
    # Face-based vector field
    if field_type == :face
        X = face_centroids(mesh)
    elseif field_type == :vertex
        X = mesh.V
    end
    arrow_args = Dict{Symbol, Any}()
    arrow_args[:linewidth] = get(kwargs, :linewidth, 0.01)
    arrow_args[:arrowsize] = get(kwargs, :arrowsize, 0.01)
    arrow_args[:lengthscale] = get(kwargs, :lengthscale, 0.01)
    arrow_args[:arrowcolor] = get(kwargs, :color, :red) 
    arrow_args[:linecolor] = get(kwargs, :color, :red) 

    arrows!(X[1,:],X[2,:],X[3,:], field[1,:], field[2,:], field[3,:]; arrow_args...)
end