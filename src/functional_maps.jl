struct Shell
    V
    ϕ_k
    n_k
    F
    K::Int
end
function Shell(S, C, T)
    X_k_new = S.V + (S_1.ϕ_k * T)'
    ϕ_k_new = C * S_1.ϕ_k'
    n_k_new = vertex_normals(X_k_new, S_1.F)
    Shell(X_k_new, ϕ_k_new, n_k_new, S.F, S.K)
end 
function apply_tau(S,T)
    X_k_new = S.V + (S.ϕ_k * T)'
    n_k_new = vertex_normals(X_k_new, S.F)
    Shell(X_k_new, S.ϕ_k , n_k_new, S.F, S.K)
end

# P: Y->X; P^T:X->Y
function compute_correspondence(S_1, S_2, f_1, f_2, P)
    # Optimizes C (functional correspondence) and T (extrinsic shift)
    # correspondence_score(S_1, S_2, f_1, f_2, P, C, T)
    # compute_fm()
    T = compute_tau(S_1, S_2, P)
end

function compute_fm(f,g, S_1, S_2)
    # f, g are |V|×|C| 
    # Note: No regularization of entries
    A = S_1.ϕ_k' * f
    B = S_2.ϕ_k' * g
    K = size(S_1)[1]
    C = zeros(K,K)
    for k=1:K
        bb = B[k,:]
        c = A' \ bb
        C[k,:] = c
    end
    C
end

function compute_tau(S_1, S_2, P)
    res = [P*S_2.V' - S_1.V'; S_2.V' - P'*S_1.V' ]
    A = [S_1.ϕ_k; P' * S_1.ϕ_k]
    A \ res
end

function feat_correspondence(S_1, S_2, sigma=0.3, N=10)
    X = [S_1.V' S_1.n_k']
    Y = [S_2.V' S_2.n_k']
    d = dist_mat(X, Y)
    sinkhorn(d, sigma, N)
end

function sinkhorn(d, sigma, N)
    # normalize distances by average
    # ϵ = 2*sigma^2 
    d = d / mean(d)
    log_p = -d ./ (2*sigma^2) # Initialize
    for t=1:N
        log_p = log_p .- LogExpFunctions.logsumexp(log_p, dims=1)
        log_p = log_p .- LogExpFunctions.logsumexp(log_p, dims=2)
    end
    log_p = log_p .- LogExpFunctions.logsumexp(log_p, dims=1)
    p = exp.(log_p)
    log_p = log_p .- LogExpFunctions.logsumexp(log_p, dims=2)
    p_adj = exp.(log_p)'
    p, p_adj
end

function dist_mat(x, y)
    # X and Y are 3× |V| matrices
    # d[x,y] for x and y
    d = -2 *x'*y
    v_x = sum(x .^2; dims=1)
    v_y = sum(y .^2; dims=1)
    return d .+ v_x' .+ v_y
end
dist_mat(x::Shell, y::Shell) = dist_mat(x.V, y.V)

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