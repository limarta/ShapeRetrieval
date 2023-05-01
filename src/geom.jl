export normalize_mesh
function normalize_mesh(mesh::Mesh)
    V = mesh.V
    Z = maximum(vec(norm(V,dims=1)))
    V ./= Z
    Mesh(V, mesh.F, mesh.normals)
end