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
    @time Mesh(V_new, F_new)
end
