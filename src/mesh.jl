mutable struct Mesh
    V::Matrix{Float64}  # 3×|V|
    F::Matrix{Int} # 3×|F|
    normals::Matrix{Float64} # 3×|F|
    nv::Int # vertex count
    nf::Int # face count
end

function Mesh(V,F,normals)
    Mesh(V,F, normals, size(V)[2], size(F)[2])
end
Mesh(V, F) = Mesh(V, F, normals(V,F))

function add_vertices(mesh::Mesh, V) end
function add_faces(mesh::Mesh, F) end

