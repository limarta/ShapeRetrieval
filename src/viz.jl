using WGLMakie

function meshviz(V,F; args...)
    fig = Figure(resolution = (900,900))
    ax = Axis3(fig[1,1], aspect=:data, elevation = 0.0, azimuth = -π/2)
    mesh!(ax, V', F'; args...)
    fig
end
meshviz(mesh::Mesh; args...)  = meshviz(mesh.V, mesh.F; args...)

function viz_field!(base, field;  kwargs...)
    arrow_args = Dict{Symbol, Any}()
    arrow_args[:linewidth] = get(kwargs, :linewidth, 0.01)
    arrow_args[:arrowsize] = get(kwargs, :arrowsize, 0.01)
    arrow_args[:lengthscale] = get(kwargs, :lengthscale, 0.01)
    arrow_args[:arrowcolor] = get(kwargs, :color, :red) 
    arrow_args[:linecolor] = get(kwargs, :color, :red) 

    arrows!(base[1,:],base[2,:],base[3,:], field[1,:], field[2,:], field[3,:]; arrow_args...)
end
function viz_field!(mesh::Mesh,field=nothing, field_type=:face; kwargs...)
    if field===nothing
        field = mesh.normals
    end

    if field_type == :face
        base = face_centroids(mesh)
    else
        base = mesh.V
    end
    viz_field!(base, field; kwargs...)
end

function viz_grid(V,F, data; kwargs...)
    # data - |V|×N

    N = size(data)[2]
    # n = ceil(Int, sqrt(N))
    # m = trunc(Int, N/n)
    # while n * m != N
    #     n -= 1
    #     m = trunc(Int, N/n)
    # end
    fig = Figure(resolution=(900,200))
    for i=1:N
        ax = Axis3(fig[1,i], aspect=:data, elevation = 0.0, azimuth = -π/2)
        hidedecorations!(ax)
        mesh!(ax, V', F'; kwargs...)
    end
    fig
end