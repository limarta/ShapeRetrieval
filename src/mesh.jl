mutable struct Mesh
    V::Matrix{Float64}  # 3×|V|
    F::Matrix{Int} # 3×|F|
    normals::Matrix{Float64} # 3×|F|
    vertex_normals::Matrix{Float64}
    FtoV
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F,N)
    A = face_area(V,F)
    ftov = FtoV(V,F,A)
    vert_N, vertex_normals(V,F)
    Mesh(V,F, N, vert_N, ftov, size(V)[2], size(F)[2])
end
Mesh(V, F) = Mesh(V, F, normals(V,F))

function add_vertices(mesh::Mesh, V) end
function add_faces(mesh::Mesh, F) end

