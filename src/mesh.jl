using DataStructures

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

function normalized_area_mesh(mesh::Mesh)
    total_area = sum(mesh.face_area)
    return Mesh(mesh.V/sqrt(total_area), mesh.F, mesh.normals)
end

function spectral_decomposition(mesh::Mesh, ϕ)
    c = ϕ'*(mesh.vertex_area .* mesh.V')
    real.(ϕ * c)
end

function smooth_spectral_decomposition(mesh::Mesh, K::Int, ϕ)
    c = ϕ'*(mesh.vertex_area .* mesh.V')
    decaying = [sigmoid(K-k) for k=1:size(ϕ)[2]]
    damped_c = decaying .* c
    real.(ϕ * damped_c)
end

