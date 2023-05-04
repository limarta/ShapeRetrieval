include("timediffusion.jl")
include("minimlp.jl")

struct DiffusionNetBlock
    C_width::Int
    diffusion_method::Symbol
    diffuse_block::LearnedTimeDiffusionBlock
    spatial_gradient::Flux.Bilinear
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    mlp::Flux.Chain
end
function DiffusionNetBlock(C_width::Int64, mlp_hidden_dims; diffusion_method::Symbol = :spectral, dropout=true, 
    with_gradient_features=true, with_gradient_rotations=true)
    diffuse_block = LearnedTimeDiffusionBlock(C_width, diffusion_method) 
    if with_gradient_features
        spatial_gradient = Flux.Bilinear((0,0)=>0)
    else
        spatial_gradient = Flux.Bilinear((0,0)=>0)
    end
    # spatial_gradient = Flux.Bilinear((1,1)=>2)
    mlp_layers = vcat(C_width, mlp_hidden_dims, C_width)
    mlp = MLP(mlp_layers, dropout)
    DiffusionNetBlock(C_width, diffusion_method, diffuse_block, spatial_gradient, with_gradient_features, with_gradient_rotations, mlp)
end

function (model::DiffusionNetBlock)(x, L, M, A, λ, ϕ, ∇_x, ∇_y)
    # run diffusion_method
    if diffusion_method == :spectral
        x_diffused = diffusion_block(x, λ, ϕ, A)
    else
        x_diffused = diffusion_block(x, λ, ϕ, A)
    end
    x_out = model.mlp(x_diffused)
end