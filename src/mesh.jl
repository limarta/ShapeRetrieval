export vertex_area, face_area, area_normals, cot_laplacian

mutable struct Mesh
    V::Matrix{Float64}  # 3*V
    F::Matrix{Int} # 3 * F
    normals::Matrix{Float64} # 3*F
    nv::Int # vertex count
    nf::Int # face count
end

Mesh(V, F, normals) = Mesh(V, F, normals, size(V)[2], size(F)[2])

vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x::Matrix; dims=1) = sqrt.(sum(x.^2, dims=dims))

function area_normals(mesh::Mesh)
    # Returns area-weighted face normals
    V = mesh.V
    F = mesh.F
    T = V[:,F]
    u = T[:,2,:] - T[:,1,:]
    v = T[:,3,:] - T[:,1,:]
    A = 0.5 * multicross(u,v) # Correct direction??
end

face_area(mesh::Mesh) = vec(norm(area_normals(mesh); dims=1))
# function vertex_area(mesh::Mesh)
#     F = mesh.F
#     A = face_area(mesh)
#     vertAreas = zeros(mesh.nv)
#     for f in eachcol(F)
#         for i=1:3
#             vertAreas[f[i]] += A[f[i]]
#         end
#     end
#     vertAreas /= 3
#     return vertAreas
# end
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
