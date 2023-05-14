function area_normals(V,F)
    T = V[:,F]
    u = T[:,2,:] - T[:,1,:]
    v = T[:,3,:] - T[:,1,:]
    A = 0.5 * multicross(u,v) # Correct direction??
end

function cot_laplacian(V,F)
    nv = size(V)[2]
    nf = size(F)[2]
    #For all 3 shifts of the roles of triangle vertices
    #to compute different cotangent weights
    cots = zeros(nf, 3)
    for perm in [(1,2,3), (2,3,1), (3,1,2)]
        i, j, k = perm
        u = V[:,F[i,:]] - V[:, F[k,:]]
        v = V[:, F[j,:]] - V[:, F[k,:]]
        cotAlpha = vec(abs.(vdot(u,v; dims=1)) ./ norm(multicross(u,v); dims=1))
        cots[:,i] = cotAlpha

    end
    I = F[1,:]; J = F[2,:]; K = F[3,:];

    L = sparse([I;J;K], [J;K;I], [cots[:,1];cots[:,2];cots[:,3]], nv, nv)
    L = L + L'
    rowsums = vec(sum(L,dims=2))
    L = spdiagm(0 => rowsums) - L
    return 0.5 * L
end

face_area(V,F) = vec(norm(area_normals(V,F); dims=1))

face_centroids(V,F) = dropdims(sum(V[:,F], dims=2) ./ 3; dims=2)

function FtoV(V,F, face_area)
    A = face_area
    nf = size(F)[2]
    nv = size(V)[2]
    G = sparse(vec(F), vec(repeat(1:nf, 1, 3)'), vec(repeat(A, 1, 3)'), nv, nf)
    G = (G' ./ A)'
end

function normals(V,F)
    A = area_normals(V,F)
    faceNormals = A ./ (norm(A, dims=1))
end

function normalize_mesh(V,F)
    Z = maximum(vec(norm(V,dims=1)))
    V ./= Z
    Mesh(V, F) # ???
end

function vertex_area(V,F)
    B = zeros(size(V)[2])
    for f in eachcol(F)
        T = V[:,f]
        x = T[:,1] - T[:,2]
        y = T[:,3] - T[:,2]
        A = 0.5*(sum(cross(x,y).^2)^0.5)
        B[f] .+= A
    end
    B ./= 3
    return B
end

function vertex_normals(V,F, face_area)
    N = normals(V,F)
    ftov = FtoV(V,F, face_area)
    (ftov * N')'
end
vertex_normals(V,F) = vertex_normals(V,F, face_area(V,F))



function face_grad(mesh::Mesh)
    cot(v1, v2) = dot(v1, v2) / norm(cross(v1, v2))
    V = mesh.V
    F = mesh.F
    ∇ = spzeros(3 * mesh.nf, mesh.nv)
    A = face_area(mesh)
    N = mesh.normals
    for f=1:mesh.nf
        u, v, w = F[:,f]
        vw = V[:,w] - V[:,v]
        wu = V[:,u] - V[:,w]
        uv = V[:,v] - V[:,u]
        J = 3f .+ (-2:0)
        ∇[J, u], ∇[J, v], ∇[J, w] = map(e->cross(N[:,f], e)/2A[f], [vw, wu, uv])
    end
    return ∇
end

function vertex_grad(V,F,N)
    nv = size(V)[2]
    frames = tangent_basis(V,F,N)
    ∇ = zeros(2,nv,nv)
    for i=1:nv # Naive
        # P(∇f) = Df -> Solve via least squares -> ∇ = (P'P)^{-1}P'D
        neighbors = [f for f in eachcol(F) if i∈f]
        neighbors = Set(vcat(neighbors...))
        delete!(neighbors, i)
        neighbors = collect(neighbors)
        edges = V[:,neighbors]
        center = V[:,i]
        edges = edges .- center
        proj_edges = embed_in_plane(frames[:,i,:], edges)'
        D = zeros(size(neighbors)[1], nv)
        D[:,i] .= -1
        ind = CartesianIndex.(1:size(neighbors)[1], neighbors)
        D[ind] .= 1
        grad_i = proj_edges \ D
        ∇[:,:,i] = grad_i
    end
    # A = mesh.vertex_area
    sparse(∇[1,:,:]'),  sparse(∇[2,:,:]')
end

function embed_in_plane(frame, edges)
    # Project the edges onto the tangent plane defined by frame
    e_1 = frame[:,1]
    e_2 = frame[:,2]
    c_1 = sum(e_1 .* edges; dims=1)
    c_2 = sum(e_2 .* edges; dims=1)
    embedding = [c_1; c_2]
end

function project_to_plane(normal::AbstractMatrix, u::AbstractVector)
    # Project u onto the tangent plane defined by vertex normals
    u .- vdot(normal, u;dims=1) ./ sum(normal.^2; dims=1) .* normal
end

function tangent_basis(V,F,N)
    # frame 3×|V|×2
    # N - vertex normals
    nv = size(V)[2]
    e_1 = [1.0, 0, 0]
    e_2 = [0, 1.0, 0]
    t_1 = project_to_plane(N, e_1)
    t_1 = t_1 ./ norm(t_1; dims=1)
    t_2 = project_to_plane(N, e_2)
    t_2 = t_2 ./ norm(t_2; dims=1)
    frame = zeros(3,nv,2)
    frame[:,:,1] = t_1
    frame[:,:,2] = t_2
    frame
end

function world_coordinates(mesh::Mesh, gradients)
    frame = tangent_basis(mesh)
    x_1 = gradients[1,:]' .* frame[:,:,1]
    x_2 = gradients[2,:]' .* frame[:,:,2] 
    x_1 + x_2
end

######################
# Convenience Methods
######################

area_normals(mesh::Mesh) = area_normals(mesh.V, mesh.F)
cot_laplacian(mesh::Mesh) = cot_laplacian(mesh.V, mesh.F)
face_area(mesh::Mesh) = face_area(mesh.V, mesh.F)
face_centroids(mesh::Mesh) = face_centroids(mesh.V, mesh.F)
FtoV(mesh::Mesh) = FtoV(mesh.V, mesh.F, face_area(mesh))
normals(mesh::Mesh) = normals(mesh.V, mesh.F)
normalize_mesh(mesh::Mesh) = normalize_mesh(mesh.V, mesh.F)
vertex_area(mesh::Mesh) = vertex_area(mesh.V, mesh.F)
vertex_normals(mesh::Mesh) = vertex_normals(mesh.V,mesh.F, mesh.face_area)
vertex_grad(mesh::Mesh) = vertex_grad(mesh.V, mesh.F, mesh.vertex_normals)
tangent_basis(mesh::Mesh) =  tangent_basis(mesh.V, mesh.F, mesh.vertex_normals)

function normalize_area(mesh::Mesh)
    total_area = sum(mesh.face_area)
    return Mesh(mesh.V/sqrt(total_area), mesh.F, mesh.normals)
end
function get_operators(mesh::Mesh; k=200)
    λ, ϕ = get_spectrum(mesh, k=k)
    grad_x, grad_y = mesh.∇_x, mesh.∇_y
    grad_x = convert.(Float32, grad_x)
    grad_y = convert.(Float32, grad_y)
    mesh.cot_laplacian, convert.(Float32, mesh.vertex_area), λ, ϕ, grad_x, grad_y
end



export cot_laplacian