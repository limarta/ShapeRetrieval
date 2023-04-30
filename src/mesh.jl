mutable struct Mesh
    V::Matrix{Float64}  # 3*V
    F::Matrix{Int} # 3 * F
    normals::Matrix{Float64} # 3*F
end
vdot(x,y; dims=1) = sum(x .* y, dims=dims)
multicross(x,y) = reduce(hcat, cross.(eachcol(x), eachcol(y)))
norm(x; dims=1) = sqrt.(sum(x.^2, dims=dims))

function area_normals(mesh::Mesh)
end
function area(mesh::Mesh)
end

function cot_laplacian(mesh::Mesh)
    V = mesh.V
    F = mesh.F
    nv = size(V)[2]
    nf = size(F)[2]
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

# """
#     cotLaplacian(X, T)
# """
# function cotLaplacian(mesh::Mesh)
#     X = mesh.V'
#     T = mesh.F'
#     nv = size(X,1)
#     nt = size(T,1)

#     I = T[:,1]; J = T[:,2]; K = T[:,3];
#     crossprods = zeros(nt,3)
#     for t=1:nt
#         crossprods[t,:] = cross(X[J[t],:]-X[I[t],:], X[K[t],:]-X[I[t],:])
#     end
#     areas = .5*norm(crossprods,dims=1)

#     cots = zeros(nt,3)
#     for i=0:2
#         I = T[:,i+1]; J = T[:,(i+1)%3+1]; K = T[:,(i+2)%3+1];
#         e1 = X[J,:]-X[I,:]
#         e2 = X[K,:]-X[I,:]
#         cots[:,i+1] = sum(e1.*e2,dims=2)./(2*areas)
#     end

#     I = T[:,1]; J = T[:,2]; K = T[:,3];
#     L = sparse([I;J;K], [J;K;I], [cots[:,3];cots[:,1];cots[:,2]], nv, nv)
#     L = L + L'
#     rowsums = vec(sum(L,dims=2))
#     L = L - spdiagm(0 => rowsums)
#     return -.5 * L
# end

export area, area_normals, cot_laplacian