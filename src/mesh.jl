mutable struct Mesh
    V::Matrix{Float64}  # 3×|V|
    F::Matrix{Int} # 3×|F|
    normals::Matrix{Float64} # 3×|F|
    face_normals::Matrix{Float64}
    vertex_normals::Matrix{Float64}
    face_area::Vector{Float64}
    vertex_area::Vector{Float64}
    FtoV
    cot_laplacian
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F,N)
    A = face_area(V,F)
    vert_A = vertex_area(V,F)
    ftov = FtoV(V,F,A)
    face_N = normals(V,F)
    vert_N = vertex_normals(V,F,A)
    L = cot_laplacian(V,F)
    Mesh(V,F, N, face_N, vert_N, A, vert_A, ftov, L, size(V)[2], size(F)[2])
end
Mesh(V, F) = Mesh(V, F, normals(V,F))

function add_vertices(mesh::Mesh, V) end
function add_faces(mesh::Mesh, F) end

