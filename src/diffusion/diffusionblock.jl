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

function (model::DiffusionNetBlock)(x, L, M, A, λ, ϕ)
    # x - |V|×|C|×|B|

    # Diffusion
    if model.diffusion_method == :spectral
        x_diffused = model.diffusion_block(x, λ, ϕ, A)
    elseif model.diffusion_method == :implicit
        x_diffused = model.diffusion_block(x, L, M, A)
    end
    # Compute gradients and stack by feature
    # x_grad = reshape(reshape(∇ * x_diffused, 2, :, model.C_width), model.C_width, :, 2)
    # model.spatial_gradient(x_grad[:,:,1], x_grad[:,:,1])
    # Combine by feature

    y = Float32.(x_diffused)
    x_out = model.mlp(y')
end