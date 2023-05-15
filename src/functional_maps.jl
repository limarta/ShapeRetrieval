struct Shell
    V
    ϕ_k
    n_k
    F
    K::Int
end

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

function optimize_deformation(S_1, S_2, f_1, f_2, P; C=nothing, T=nothing)
    # Optimizes C (functional correspondence) and T (extrinsic shift)
    if C === nothing
        C = rand(Float32, S_2.K, S_1.K)
    end
    if T === nothing
        T = rand(Float32, S_1.K, 3)
    end
    for i=1:100
        grad = gradient(correspondence_score, S_1, S_2, f_1, f_2, P, C, T)
    end
    # correspondence_score(S_1, S_2, f_1, f_2, P, C, T)
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

function correspondence_score(S_1, S_2, f_1, f_2, P, C, T)
    # Assumes V_1 and V_2 are K-smooth and ϕ_1 and ϕ_2 are |V|×K
    X_k_new = S_1.V + (S_1.ϕ_k * T)'
    ϕ_k_new = S_1.ϕ_k * C'
    n_k_new = vertex_normals(X_k_new, S_1.F) # compute normals from V_1

    Y_k = (P * X_k_new')'
    n_y_k = (P * n_k_new')'
	
    error = sum((S_2.V - Y_k).^2) + sum((S_2.ϕ_k - ϕ_k_new).^2) + sum((S_2.n_k - n_y_k).^2) 
end

function correspondence_score(S_1, S_2, f_1, f_2, P, C, T, V::Symbol)
    X_k_new = S_1.V + (S_1.ϕ_k * T)'
    ϕ_k_new = S_1.ϕ_k * C'
    n_k_new = vertex_normals(X_k_new, S_1.F) # compute normals from V_1

    Y_k = (P * X_k_new')'
    n_y_k = (P * n_k_new')'
    # println("ϕ_k")
    # display(S_1.ϕ_k)
    # println("ϕ_k_new")
    # display(ϕ_k_new)
    # println("y ϕ_k")
    # display(S_2.ϕ_k)
	
    e_1 = sum((S_2.V - Y_k).^2) 
	e_2 = sum((S_2.ϕ_k - ϕ_k_new).^2)
	e_3 = sum((S_2.n_k - n_y_k).^2)
    feat_error = sum((C*S_1.ϕ_k'*f_1 - S_2.ϕ_k' * f_2).^2)
    arap_error = 0
    Dict{Symbol, Float32}(:vertex =>e_1, :spectral=>e_2, :normal=>e_3, :feature=>feat_error, :arap=>arap_error)
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
    n_k = vertex_normals(X_k, F) # TODO: Incorrect

    X_k, ϕ_k, n_k
    Shell(X_k, ϕ_k, n_k, F,K)
end

######################
# Convenience Methods
######################
shell_coordinates(mesh::Mesh, K, ϕ) = shell_coordinates(mesh.V, mesh.F, K, ϕ)
smooth_spectral_decomposition(mesh::Mesh, K::Int, ϕ) = smooth_spectral_decomposition(mesh.V, K, ϕ)
spectral_decomposition(mesh::Mesh, K::Int, ϕ) = spectral_decomposition(mesh.V, K, ϕ)