using WGLMakie

wgl_coords(V) = [V[1,:] -V[3,:]  V[2,:]]'

function meshviz(V,F; resolution=(900,900), shift_coordinates=false, args...)
    fig = Figure(resolution = resolution)
    ax = Axis3(fig[1,1], aspect=:data, elevation = 0.0, azimuth = -π/2)
    if shift_coordinates
        V_new = wgl_coords(V)
    else
        V_new = V
    end
    mesh!(ax, V_new', F'; args...)
    fig 
end
meshviz(mesh::Mesh; args...)  = meshviz(mesh.V, mesh.F; args...)
meshviz(mesh::Shell; args...) = meshviz(mesh.X_k, mesh.F; args...)

function meshviz(meshes::Vector; shift_coordinates=false, args...) 
    N = length(meshes)
    resolution = (600*N, 300* N)
    fig = Figure(resolution=resolution)
    for i=1:N
        ax = Axis3(fig[1,i], aspect=:data, elevation=0.0, azimuth=-π/2)
        hidedecorations!(ax)
        hidespines!(ax)
        if shift_coordinates
            V_new = wgl_coords(meshes[i].V)
        else
            V_new = meshes[i].V
        end
        mesh!(ax, V_new', meshes[i].F'; args...)
    end
    fig
end

function viz_field!(base, field;  shift_coordinates=false, kwargs...)
    arrow_args = Dict{Symbol, Any}()
    arrow_args[:linewidth] = get(kwargs, :linewidth, 0.01)
    arrow_args[:arrowsize] = get(kwargs, :arrowsize, 0.01)
    arrow_args[:lengthscale] = get(kwargs, :lengthscale, 0.01)
    arrow_args[:arrowcolor] = get(kwargs, :color, :red) 
    arrow_args[:linecolor] = get(kwargs, :color, :red) 
    if shift_coordinates
    end

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
viz_field!(mesh::Shell; kwargs...) = viz_field!(mesh.X_k, mesh.n_k; kwargs...)

function viz_point!(mesh::Mesh, id::Int; kwargs...)
    xyz = mesh.V[:,id]
    scatter!(xyz...,; kwargs...)
end

function viz_grid(V,F, data; shift_coordinates=false, kwargs...)
    # data - |V|×N
    if shift_coordinates
        V = wgl_coords(V)
    end

    N = size(data)[2]
    if N > 5
        if haskey(kwargs, :dims)
            n = kwargs[:dims][1]
            m = kwargs[:dims][2]
        else
            n = ceil(Int, sqrt(N))
            m = trunc(Int, N/n)
            while n * m != N
                n -= 1
                m = trunc(Int, N/n)
            end
        end
        fig = Figure(resolution=(500*n,300*m))
        for i=1:n
            for j=1:m
                ax = Axis3(fig[i,j],  aspect=:data, elevation = 0.0, azimuth = -π/2)
                hidedecorations!(ax)
                hidespines!(ax)
                v = mesh!(ax, V', F'; color=data[:,(i-1)*m+j], kwargs...)
            end
        end
    else
        fig = Figure(resolution=(1000,200*N))
        for i=1:N
            ax = Axis3(fig[1,i],  aspect=:data, elevation = 0.0, azimuth = -π/2)
            hidedecorations!(ax)
            hidespines!(ax)
            v = mesh!(ax, V', F'; color=data[:,i], kwargs...)
        end
    end
    fig
end
