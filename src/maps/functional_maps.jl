function compute_correspondence(f,g, λ_1, λ_2, ϕ_1, ϕ_2)
    # f, g are |V|×|C| 
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