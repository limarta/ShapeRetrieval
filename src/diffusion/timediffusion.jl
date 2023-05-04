export LearnedTimeDiffusionBlock

# Diffusion block = diffusion + inner product + dense layer
struct LearnedTimeDiffusionBlock
    C_inout::Int
    diffusion_time::Vector{Float64}
    method::Symbol
    function LearnedTimeDiffusionBlock(C_inout::Int, diffusion_time::Vector{Float64}, method::Symbol)
        if method == :spectral || method == :implicit
            new(C_inout, diffusion_time, method)
        else
            throw("Invalid diffusion method $(method)")
        end
    end
    LearnedTimeDiffusionBlock(C_inout::Int, method::Symbol) = LearnedTimeDiffusionBlock(C_inout, rand(C_inout), method)
end

Flux.@functor LearnedTimeDiffusionBlock
Flux.trainable(m::LearnedTimeDiffusionBlock) = (m.diffusion_time,)

function (model::LearnedTimeDiffusionBlock)(x, L, M, A::Vector{Float64})
    # LM - diffusion operator M+dt*L
    # A - vertex areas

    dt = max(model.diffusion_time[1],0)
    D = Matrix(M + dt * L) # Need to make dense :/
    F = cholesky(D)
    heat = x
    for t=1:4
        heat = F \ (heat.* A)
    end
end

function (model::LearnedTimeDiffusionBlock)(x, λ::Vector{ComplexF64}, ϕ::Matrix{ComplexF64}, A::Vector{Float64})
    # x - features |V|×|C|
    # λ, ϕ - eigvals, eigvecs
    # A - vertex area
    time = max.(model.diffusion_time,1e-8)
    x_diffused = heat_diffusion(λ, ϕ, A, x, time)
end
