abstract type DiffusionMode end

struct Spectral <: DiffusionMode end
struct Implicit <: DiffusionMode end

struct DiffusionNetBlock{M<:DiffusionMode, D<:LearnedTimeDiffusionBlock, G<:SpatialGradientBlock ,S,U}
    C_width::Int
    diffusion_mode::M
    diffusion_block::D
    spatial_gradient::G
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    mlp::MLP{S,U}
end
@Flux.functor DiffusionNetBlock
Flux.trainable(m::DiffusionNetBlock) = (diffusion_block = m.diffusion_block, spatial_gradient=m.spatial_gradient, mlp = m.mlp)

function DiffusionNetBlock(C_width::Int, mlp_hidden_dims; diffusion_mode=Spectral(), dropout=true, 
    with_gradient_features=true, with_gradient_rotations=true)
    diffusion_block = LearnedTimeDiffusionBlock(C_width) 
    spatial_gradient = SpatialGradientBlock(with_gradient_rotations, C_width)
    mlp_layers = vcat(2*C_width, mlp_hidden_dims, C_width)
    mlp = MLP(mlp_layers, dropout)
    DiffusionNetBlock(C_width, diffusion_mode, diffusion_block, spatial_gradient, with_gradient_features, with_gradient_rotations, mlp)
end

# x - |V|×|C|×|B|

(model::DiffusionNetBlock{Implicit})(x, L, A, ∇_x, ∇_y) = 
    x_out = process_gradient_field(model, model.diffusion_block(x, L, A), ∇_x, ∇_y)

function (model::DiffusionNetBlock{Spectral})(x, λ, ϕ, A, ∇_x, ∇_y)
    x_diffused = model.diffusion_block(x, λ, ϕ, A)
    # x_out = process_gradient_field(model, x_diffused, ∇_x, ∇_y)
    if model.with_gradient_features
        grad_x = (∇_x * x_diffused)'
        grad_y = (∇_y * x_diffused)'
        x_intermediate = model.spatial_gradient(grad_x, grad_y)
    else
        x_intermediate = x_diffused'
    end
    x_intermediate = vcat(x_diffused', x_intermediate)
    x_out = model.mlp(x_intermediate)
end

function process_gradient_field(model::DiffusionNetBlock, x_diffused, ∇_x, ∇_y)
    if model.with_gradient_features
        grad_x = (∇_x * x_diffused)'
        grad_y = (∇_y * x_diffused)'
        x_intermediate = model.spatial_gradient(grad_x, grad_y)
    else
        x_intermediate = x_diffused'
    end
    x_intermediate = vcat(x_diffused', x_intermediate)
    x_out = model.mlp(x_intermediate)
end
