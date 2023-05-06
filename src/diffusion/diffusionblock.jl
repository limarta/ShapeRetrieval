struct DiffusionNetBlock
    C_width::Int
    diffusion_method::Symbol
    diffusion_block::LearnedTimeDiffusionBlock
    spatial_gradient::Flux.Bilinear
    with_gradient_features::Bool
    with_gradient_rotations::Bool
    mlp::Flux.Chain
end
@Flux.functor DiffusionNetBlock
Flux.trainable(m::DiffusionNetBlock) = (diffusion_block = m.diffusion_block, mlp = m.mlp)

function DiffusionNetBlock(C_width::Int, mlp_hidden_dims; diffusion_method = :spectral, dropout=true, 
    with_gradient_features=true, with_gradient_rotations=true)
    diffusion_block = LearnedTimeDiffusionBlock(C_width, diffusion_method) 

    if with_gradient_features
        if with_gradient_rotations
            spatial_gradient = Flux.Bilinear((C_width,C_width)=>1)
        else
            spatial_gradient = Flux.Bilinear((C_width, C_width)=>1)
        end
    else
        spatial_gradient = Flux.Bilinear((0,0)=>0)
    end
    mlp_layers = vcat(C_width, mlp_hidden_dims, C_width)
    mlp = MLP(mlp_layers, dropout)
    DiffusionNetBlock(C_width, diffusion_method, diffusion_block, spatial_gradient, with_gradient_features, with_gradient_rotations, mlp)
end

function (model::DiffusionNetBlock)(x, L, M, A, λ, ϕ, ∇_x, ∇_y)
    # x - |V|×|C|×|B|

    # Diffusion
    if model.diffusion_method == :spectral
        x_diffused = model.diffusion_block(x, λ, ϕ, A)
    elseif model.diffusion_method == :implicit
        x_diffused = model.diffusion_block(x, L, M, A)
    end
    # Compute gradients and stack by feature
    if model.with_gradient_features
        grad_x = (∇_x * x_diffused)'
        grad_y = (∇_y * x_diffused)'
        metric = model.spatial_gradient(grad_x, grad_x)
        y = Float32.(metric)
        println("mlp ", model.mlp)
        println(size(y))
        x_out = model.mlp(y')
        return x_out
    else
        y = Float32.(x_diffused)
        x_out = model.mlp(y')
    end

end