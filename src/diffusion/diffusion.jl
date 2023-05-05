include("timediffusion.jl")
include("minimlp.jl")
include("diffusionblock.jl")

struct DiffusionNet
    C_in::Int # Input dimension
    C_out::Int 
    C_width::Int # dimension of internal block
    N_block::Int 
    last_activation::Bool
    outputs_at::Symbol # vertex/face
    mlp_hidden_dims::Vector{Int} 
    dropout::Bool
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    diffusion_method::Symbol
    first::Flux.Dense
    last::Flux.Dense
    blocks::DiffusionNetBlock

    function DiffusionNet(C_in, C_out, C_width, N_block, last_activation=true, outputs_at=:vertex, mlp_hidden_dims=nothing, dropout=true, 
        with_gradient_features=true, with_gradient_rotations=true, diffusion_method=:spectral)
        if mlp_hidden_dims === nothing
            mlp_hidden_dims = [C_width,]
        end
        first = Dense(C_in=>C_width)
        last = Dense(C_width=>C_out)
        b1 = DiffusionNetBlock(C_width, mlp_hidden_dims; diffusion_method=diffusion_method,
            dropout=dropout, with_gradient_features=with_gradient_features, with_gradient_rotations=with_gradient_rotations)
        blocks = b1 # Fix to make multiple blocks
        new(C_in, C_out, C_width, N_block, last_activation, outputs_at, mlp_hidden_dims, dropout, with_gradient_features, with_gradient_rotations,
        diffusion_method, first, last, blocks)
    end
end

@Flux.functor DiffusionNet
Flux.trainable(net::DiffusionNet) = (first=net.first, blocks=net.blocks, last=net.last)

function (model::DiffusionNet)(x, L, M, A, λ, ϕ, ∇)
    x_in = model.first(x')'
    x_imm = model.blocks(x_in, L, M, A, λ, ϕ, ∇)
    x_out = model.last(x_imm)
end