struct SpatialGradientBlock{T<:AbstractVecOrMat}
    is_rotation::Bool
    C_in::Int
    A::T
    B::T
end

@Flux.functor SpatialGradientBlock
Flux.trainable(net::SpatialGradientBlock) = (A=net.A, B=net.B)

SpatialGradientBlock(is_rotation::Bool, C_in::Int) = SpatialGradientBlock(is_rotation, C_in, rand(Float32, C_in,C_in).-0.5f0, rand(Float32,C_in, C_in).-0.5f0)

function (model::SpatialGradientBlock)(grad_x, grad_y)
    if model.is_rotation 
        re = model.A * grad_x
        im = model.B * grad_y
    else
        re = model.A * grad_x
        im = model.A * grad_y
    end
    dots = grad_x .* re + grad_y .* im
    relu.(dots)
end