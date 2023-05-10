using WGLMakie

wgl_coords(V) = [V[:,1] -V[:,3]  V[:,2]]
function meshviz(V,F; args...)
    fig = Figure(resolution = (900,900))
    ax = Axis3(fig[1,1], aspect=:data, elevation = 0.0, azimuth = -π/2)
    V_new = wgl_coords(V')
    mesh!(ax, V_new, F'; args...)
    # fig,lscene,z = mesh(V', F'; args...)
    # cam = cameracontrols(lscene)
    # update_cam!(lscene.scene, (1,0.4,0.0), (0.0, 0.0, 0.0))
    # rotate_cam!(lscene.scene, 0, π/4, -π)
    # fig
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
        fig = Figure(resolution=(1000,200*m))
        for i=1:n
            for j=1:m
                ax = Axis3(fig[i,j],  aspect=:data, elevation = 0.0, azimuth = -π/2)
                hidedecorations!(ax)
                hidespines!(ax)
                v = mesh!(ax, wgl_coords(V'), F'; color=data[:,(i-1)*m+j], kwargs...)
            end
        end
    else
        fig = Figure(resolution=(1000,200))
        for i=1:N
            ax = Axis3(fig[1,i],  aspect=:data, elevation = 0.0, azimuth = -π/2)
            hidedecorations!(ax)
            hidespines!(ax)
            v = mesh!(ax, wgl_coords(V'), F'; color=data[:,i], kwargs...)
        end
    end
    fig
end