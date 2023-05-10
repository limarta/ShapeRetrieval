abstract type DiffusionMode end

struct Spectral <: DiffusionMode end
struct Implicit <: DiffusionMode end

struct DiffusionNetBlock{T<:DiffusionMode}
    C_width::Int
    diffusion_mode::T
    diffusion_block::LearnedTimeDiffusionBlock
    spatial_gradient::SpatialGradientBlock
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    mlp::Flux.Chain
end
@Flux.functor DiffusionNetBlock
Flux.trainable(m::DiffusionNetBlock) = (diffusion_block = m.diffusion_block, mlp = m.mlp)

function DiffusionNetBlock(C_width::Int, mlp_hidden_dims; diffusion_mode=Spectral(), dropout=true, 
    with_gradient_features=true, with_gradient_rotations=true)
    diffusion_block = LearnedTimeDiffusionBlock(C_width) 
    spatial_gradient = SpatialGradientBlock(with_gradient_rotations, C_width)
    mlp_layers = vcat(C_width, mlp_hidden_dims, C_width)
    mlp = MLP(mlp_layers, dropout)
    DiffusionNetBlock(C_width, diffusion_mode, diffusion_block, spatial_gradient, with_gradient_features, with_gradient_rotations, mlp)
end

# x - |V|×|C|×|B|

(model::DiffusionNetBlock{Implicit})(x, L, A::Vector, ∇_x, ∇_y) = 
    x_out = process_gradient_field(model, model.diffusion_block(x, L, A), ∇_x, ∇_y)

function (model::DiffusionNetBlock{Spectral})(x, λ, ϕ, A::Vector, ∇_x, ∇_y)
    x_diffused = 10*model.diffusion_block(x, λ, ϕ, A)
    x_out = process_gradient_field(model, x_diffused, ∇_x, ∇_y)
end

function process_gradient_field(model::DiffusionNetBlock, x_diffused, ∇_x, ∇_y)
    if model.with_gradient_features
        grad_x = (∇_x * x_diffused)'
        grad_y = (∇_y * x_diffused)'
        x_intermediate = model.spatial_gradient(grad_x, grad_y)
    else
        x_intermediate = x_diffused'
    end
    x_out = model.mlp(x_intermediate)
end
