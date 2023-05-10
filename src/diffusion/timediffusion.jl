export LearnedTimeDiffusionBlock

abstract type DiffusionMode end

struct Spectral <: DiffusionMode end
struct Implicit <: DiffusionMode end

struct LearnedTimeDiffusionBlock
    C_inout::Int
    diffusion_time::Vector{Float64}
end
function LearnedTimeDiffusionBlock(C_inout::Int)
    LearnedTimeDiffusionBlock(C_inout, rand(C_inout))
end

Flux.@functor LearnedTimeDiffusionBlock
Flux.trainable(m::LearnedTimeDiffusionBlock) = (diffusion_time = m.diffusion_time,)

function (model::LearnedTimeDiffusionBlock)(x, L, A)
    # LM - diffusion operator M+dt*L
    # A - vertex areas

    M = diagm(A)

    T = length(model.diffusion_time)
    heat_buf = Zygote.Buffer(x, size(x)[1], T)
    for t=1:size(model.diffusion_time)[1]
        dt = max(model.diffusion_time[t],1e-8)
        F = M + dt * L # Need to make dense :/
        # F = cholesky(D)
        heat = F \ (x[:,t].* A)
        heat_buf[:,t] = heat
    end
    copy(heat_buf)
end

function (model::LearnedTimeDiffusionBlock)(x, λ::Vector, ϕ::Matrix, A::Vector{Float64})
    # x - features |V| or |V|×|C| or |V|×|C|×|B|
    # λ, ϕ - eigvals, eigvecs |V|×|K|
    # A - vertex area |V|×|B|
    if size(x)[end] != model.C_inout
        throw("Input channels do not match $(size(x)[end]) C_inout=$(model.C_inout)")
    end
    time = max.(model.diffusion_time,1e-8)
    x_diffused = heat_diffusion(λ, ϕ, A, x, time)
end
