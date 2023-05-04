include("timediffusion.jl")
include("spectralgradient.jl")
include("minimlp.jl")

struct DiffusionNetBlock
    C_width::Int
    diffusion_method::Symbol
    diffuse_block::LearnedTimeDiffusionBlock
    spatial_gradient::Flux.Bilinear
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    mlp
end
function DiffusionNetBlock(C_width::Int, mlp_hidden_dims::Vector{Int}; dropout::Bool=True, 
    diffusion_method::Symbol, with_gradient_features::Bool=True, with_gradient_rotations::Bool=True)
    diffuse_block = LearnedTimeDiffusion(C_width, diffusion_method) 
    spatial_gradient = Flux.Bilinear((1,1)=>2)
    mlp = MLP(mlp_hidden_dims, dropout)
    # DiffusionNetBlock(C_width, diffuse_block, spatial_gradient, with_gradient_features, with_gradient_rotations, mlp)
end

function (model::DiffusionNetBlock)(x, L, M, A, λ, ϕ, ∇_x, ∇_y)
    # run diffusion_method
    x_diffused = diffusion_block(x, λ, ϕ, A)
    if model.with_gradient_features
    end
    x_out = model.mlp(x_diffused)
end