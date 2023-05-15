struct Shell
    V
    ϕ_k
    n_k
    F
    K::Int
end

function compute_fm(f,g, λ_1, λ_2, ϕ_1, ϕ_2)
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

# P: Y->X; P^T:X->Y
function optimize_deformation(S_1, S_2, f_1, f_2, P; C=nothing, T=nothing)
    # Optimizes C (functional correspondence) and T (extrinsic shift)
    # correspondence_score(S_1, S_2, f_1, f_2, P, C, T)
    k_1 = S_1.K
    k_2 = S_2.K
end

function compute_tau(S_1, S_2, P)
    res = [P*S_2.V' - S_1.V'; S_2.V' - P'*S_1.V' ]
    A = [S_1.ϕ_k; P' * S_1.ϕ_k]
    A \ res
end

function apply_tau(S, T)
    V = (S.V' + S.ϕ_k * T)'
    V, S.F
end

function optimize_bijection(S_1, S_2, f_1, f_2, C, T; P) 
    # Optimizes P (1-1 correspondence)
    nv1 = size(S_1.V)[2]
    nv2 = size(S_2.V)[2]
    if P === nothing
        P = rand(Float32, nv2, nv1)
    end
    # Perform sinkhorn iterations
end

function correspondence_score(S_1, S_2, f_1, f_2, P, C, T, a)
    # P - 1-to-1 correspondence
    # C - functional map correspondence
    # T - coordinate deformation
    # a - entropic regularizer

    X_k_new = S_1.V + (S_1.ϕ_k * T)'
    ϕ_k_new = C * S_1.ϕ_k'
    n_k_new = vertex_normals(X_k_new, S_1.F)

    # Y_k = (P * X_k_new')'
    # n_y_k = (P * n_k_new')'
	
    error = sum((S_2.V - Y_k).^2) + sum((S_2.ϕ_k - ϕ_k_new).^2) + sum((S_2.n_k - n_y_k).^2) + a * entropy(P)
end

function compute_mean_error(S_1, S_2, P, T)
    X_k_new = S_1.V + (S_1.ϕ_k * T)'
    sum((X_k_new - P* S_2.V).^2) + sum((P' * X_k_new - S_2.V).^2)
end


function spectral_decomposition(V,K,ϕ)
    ϕ = ϕ[:,1:K]
    # c = ϕ'*(mesh.vertex_area .* mesh.V')
    c = ϕ'*V'
    real.(ϕ * c)
end


function smooth_spectral_decomposition(V, K, ϕ)
    # c = ϕ' * (mesh.A .* V')
    c = ϕ' * V'
    decaying = [sigmoid(K-k) for k=1:size(ϕ)[2]]
    damped_c = decaying .* c
    real.(ϕ * damped_c)'
end

function shell_coordinates(V, F, K, ϕ)
    X_k = smooth_spectral_decomposition(V, K, ϕ)
    ϕ_k = ϕ[:,1:K]
    n_k = vertex_normals(X_k, F)

    X_k, ϕ_k, n_k
    Shell(X_k, ϕ_k, n_k, F,K)
end

######################
# Convenience Methods
######################
shell_coordinates(mesh::Mesh, K, ϕ) = shell_coordinates(mesh.V, mesh.F, K, ϕ)
smooth_spectral_decomposition(mesh::Mesh, K::Int, ϕ) = smooth_spectral_decomposition(mesh.V, K, ϕ)
spectral_decomposition(mesh::Mesh, K::Int, ϕ) = spectral_decomposition(mesh.V, K, ϕ)