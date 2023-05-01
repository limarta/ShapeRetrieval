export face_centroids, normalize_mesh, vertex_area, face_area, area_normals, normals, cot_laplacian 
function normalize_mesh(mesh::Mesh)
    V = mesh.V
    Z = maximum(vec(norm(V,dims=1)))
    V ./= Z
    Mesh(V, mesh.F, mesh.normals)
end

function face_centroids(mesh::Mesh)
    T = mesh.V[:,mesh.F]
    dropdims(sum(T, dims=2) ./ 3; dims=2)
end

function area_normals(V,F)
    T = V[:,F]
    u = T[:,2,:] - T[:,1,:]
    v = T[:,3,:] - T[:,1,:]
    A = 0.5 * multicross(u,v) # Correct direction??
end
area_normals(mesh::Mesh) = area_normals(mesh.V, mesh.F)
function normals(V,F)
    A = area_normals(V,F)
    faceNormals = A ./ (norm(A, dims=1))
end
normals(mesh::Mesh) = normals(mesh.V, mesh.F)

face_area(mesh::Mesh) = vec(norm(area_normals(mesh); dims=1))

function vertex_area(mesh::Mesh)
    V = mesh.V
    F = mesh.F
    B = zeros(mesh.nv)
    for f in eachcol(F)
        T = V[:,f]
        x = T[:,1] - T[:,2]
        y = T[:,3] - T[:,2]
        A = 0.5*(sum(cross(x,y).^2)^0.5)
        B[f] .+= A
    end
    B ./= 3
    # println(sum(B))
    return B
end

function cot_laplacian(mesh::Mesh)
    V = mesh.V
    F = mesh.F
    nv = mesh.nv
    nf = mesh.nf
    L = spzeros(nv,nv)
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
    return -.5 * L
end

function FtoV(mesh::Mesh) end
function VtoF(mesh::Mesh) end
# mesh.FtoV = mesh.massMatrix \ sparse(vec(T), vec(repeat(1:mesh.nf, 1, 3)), vec(repeat(mesh.faceAreas ./ 3, 1, 3)), mesh.nv, mesh.nf)
# mesh.VtoF = sparse(vec(repeat(1:mesh.nf, 1, 3)), vec(T), fill(1/3, length(T)), mesh.nf, mesh.nv)
function face_grad(mesh::Mesh)
    ∇ = spzeros(3 * mesh.nf, mesh.nf)
    # cot(v1, v2) = dot(v1, v2) / norm(cross(v1, v2))
    # V = mesh.V
    # F = mesh.F
    # for f=1:mesh.nf
    #     u, v, w = F[:,f]...
    #     # A =area 
    #     A = 1
    #     vw = V[w, :] - V[v, :]
    #     wu = V[u, :] - V[w, :]
    #     uv = V[v, :] - V[u, :]
    #     vu, uw, wv = -uv, -wu, -vw
    #     J = 3f .+ (-2:0)
    #     ∇[J, u], ∇[J, v], ∇[J, w] = map(e->cross(data.faceNormals[f, :], e)/2A,
    #                                              [vw, wu, uv])
    # end
    return ∇
end
function vertex_grad(mesh::Mesh)
    ∇ = face_grad(mesh)  # 3|F| × V
    P * ∇
end
# function compute_operators(mesh::Mesh)
#     # https://github.com/nmwsharp/diffusion-net/blob/55931bcbec8b27810f2401dd6553a975e2c8cb7e/src/diffusion_net/geometry.py#L276
# end