mutable struct Mesh
    V::Matrix  # 3×|V|
    F::Matrix{Int} # 3×|F|
    normals::Matrix # 3×|F|
    face_normals::Matrix
    vertex_normals::Matrix
    face_area::Vector
    vertex_area::Vector
    FtoV
    ∇_x
    ∇_y
    cot_laplacian
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F,N)
    A = face_area(V,F)
    vert_A = vertex_area(V,F)
    # ftov = FtoV(V,F,A)
    ftov = zeros(1,1)
    face_N = normals(V,F)
    vert_N = vertex_normals(V,F)
    ∇_x, ∇_y = vertex_grad(V,F,vert_N)
    L = cot_laplacian(V,F)
    Mesh(V,F, N, face_N, vert_N, A, vert_A, ftov, ∇_x, ∇_y, L, size(V)[2], size(F)[2])
end
Mesh(V, F) = Mesh(V, F, normals(V,F))

function Base.copy(mesh::Mesh)
    Mesh(copy(mesh.V), copy(mesh.F), copy(mesh.normals), copy(mesh.face_normals), copy(mesh.vertex_normals),
        copy(mesh.face_area), copy(mesh.vertex_area), copy(mesh.FtoV), 
        copy(mesh.∇_x), copy(mesh.∇_y), copy(mesh.cot_laplacian), mesh.nv, mesh.nf) 
end
