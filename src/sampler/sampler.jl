struct MeshSampler
    mesh::Mesh
    G
end

function MeshSampler(mesh::Mesh)
    # Create 
end

function relabel_mesh_from_mask(mesh::Mesh, V_mask, F_mask)
    V_new = mesh.V[:, V_mask .==1]
    F_new = mesh.F[:, F_mask .==1]
    old_ids = findall(x->x==1, V_mask)
    relabels = zeros(Int, mesh.nv)
    relabels[old_ids] = 1:length(old_ids)
    F_new = replace(x -> relabels[x], F_new)
    Mesh(V_new, F_new), relabels
end

function get_in_sphere(mesh::Mesh, point, radius)
    V = mesh.V
    F = mesh.F
    center = V[:,point]
    dist = vec(norm(V .- center, dims=1))
    V_sampled = dist.<radius
    ind = findall(<(radius), dist)
    i = F[1,:] .∈ Ref(ind)
    j = F[2,:] .∈ Ref(ind)
    k = F[3,:] .∈ Ref(ind)
    F_sampled = i .& j .& k
    V_sampled, F_sampled
end

function connected_mesh_in_sphere(mesh::Mesh, point, radius)
    V_sampled, F_sampled = get_in_sphere(mesh, point, radius)

    start_vertex = point
        
    visited_vertices = .!V_sampled
    stack = Stack{Int}()
    push!(stack, start_vertex)
    visited_vertices[start_vertex] = true
    visited_faces = .!F_sampled

    # DFS for reachability
    while !isempty(stack)
        vert = pop!(stack)
        
        faces = findall(==(vert), mesh.F)
        for F in faces
            f = F[2]
            if visited_faces[f]
                continue
            end
            visited_faces[f] = true

            for v in mesh.F[:, f]
                if visited_vertices[v]
                    continue
                end
                
                push!(stack, v)
                visited_vertices[v] = true
            end
        end
    end
    V_sampled .& visited_vertices, F_sampled .& visited_faces
end
