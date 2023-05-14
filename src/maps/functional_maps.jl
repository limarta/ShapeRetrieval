function compute_correspondence(f,g, λ_1, λ_2, ϕ_1, ϕ_2)
    # f, g are |V|×|C| 
    # Note: No regularization of entries
    A = ϕ_1' * f
    B = ϕ_2' * g
    K = size(λ_1)[1]
    C = zeros(K,K)
    for k=1:K
        bb = B[k,:]
        c = A' \ bb
        C[k,:] = c
    end
    C
end

function smooth_deformation(V_1, V_2, F_1, F_2, N_1, N_2, ϕ_1, ϕ_2, C, D=nothing, T=nothing)
    nv1 = size(V_1)[2]
    nv2 = size(V_2)[2]
    if D === nothing
        D = rand(Float32, nv2, nv1)
    end
    if T === nothing
        # T = rand(Float32, )
    end
    # Computes the deformation+translation between vertices V_1 and V_2 given a correspondence matrix
end
function smooth_correspondence(V_1, V_2, F_1, F_2, N_1, N_2, ϕ_1, ϕ_2, D, T, C=nothing) 
    nv1 = size(V_1)[2]
    nv2 = size(V_2)[2]
    if C === nothing
        C = rand(Float32, nv2,nv1)
    end
    
    # Perform gradient descent on D,T
end
function smooth_correspondence_score(f_1, f_2, V_1, V_2, F_1, F_2, N_1, N_2, ϕ_1, ϕ_2, C, D, T)
    # Assumes V_1 and V_2 are K-smooth and ϕ_1 and ϕ_2 are |V|×K
    extrinsic_deformation = V_1 + (ϕ_1*T)'
    spectral_deformation = ϕ_1 * C'
    normal_deformation = 0 # compute normals from V_1

    y_1 = P * extrinsic_deformation
    y_2 = P * spectral_deformation
    y_3 = P * normal_deformation

    error = sum((V_2 - y_1).^2 + (ϕ_2 - y_2).^2 + (N_2 - y_3).^2)

    feat_error = sum((C*ϕ_1'*f_1 - ϕ_2' * f_2).^2)
    arap_error = 0
    error + feat_error + arap_error

end

# Convenience methods
function smooth_correspondence_score(mesh_1, mesh_2, ϕ_1, ϕ_2, K, C, D, T)
    1
end