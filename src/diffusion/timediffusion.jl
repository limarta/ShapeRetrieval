using Flux
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
    function LearnedTimeDiffusionBlock(C_inout::Int, method::Symbol)
        diffusion_time = rand(C_inout)
        LearnedTimeDiffusionBlock(C_inout, diffusion_time, method)
    end
end

Flux.@functor LearnedTimeDiffusionBlock
function (model::LearnedTimeDiffusionBlock)(x, L, M, A::Vector{Float64})
    # LM - diffusion operator M+dt*L
    # A - vertex areas

    dt = max(model.diffusion_time[1],0)
    D = Matrix(M + dt * L) # Need to make dense :/
    F = cholesky(D)
    heat = x
    for t=1:10
        heat = F \ (heat.* A)
    end
    sum(heat)
end
function (model::LearnedTimeDiffusionBlock)(x, λ::Vector{ComplexF64}, ϕ::Matrix{ComplexF64}, A::Vector{Float64})
    # x - feature 
    # λ, ϕ - eigvals, eigvecs
    # A - vertex area
    time = max(model.diffusion_time[1],1e-8)
    x_diffused = heat_diffusion(λ, ϕ, A, x, time)
    sum(x_diffused)
end

# struct SpatialGradientFeatures
#     C_inout::Int32
#     A::Matrix{Float64}
# end

# function (model::SpatialGradientFeatures)(vectors)
#     vectorsA = vectors # (V,C)

#     # if self.with_gradient_rotations:
#     #     vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
#     #     vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
#     # else:
#     vectorsBreal = self.A(vectors[...,0])
#     vectorsBimag = self.A(vectors[...,1])

#     dots = vectorsA[...,1] * vectorsBreal + vectorsA[...,2] * vectorsBimag

#     return tanh(dots)
# end

# struct MiniMLP

# end

# struct DiffusionNetBlock
#     C_width::Int
#     diffusion_method::Symbol
#     time_diffusion::LearnedTimeDiffusionBlock
#     spatial_gradient::SpatialGradientFeatures
#     mlp
#     function DiffusionNetBlock(C_width::Int, diffusion_method::Symbol)
#         td = LearnedTimeDiffusionBlock()
#         sg = SpatialGradientFeatures()
#         mlp = 
#         new(C_width, diffusion_method, td, sg, mlp)
#     end
# end

# struct DiffusionNet
#     C_in::Int
#     C_out::Int
#     C_width::Int
#     N_block::Int
#     diffusion_method::Symbol
#     blocks::Vector{DiffusionNetBlock}
#     first::Dense
#     last::Dense
#     function DiffusionNetBlock(C_in::Int, C_out::Int, C_width::Int, N_block::Int, diffusion_method::Symbol)
#         blocks = Vector{DiffusionNetBlock}[]
#         for i=1:N_block
#             b = DiffusionNetBlock()
#             push!(blocks, b)
#         end
#         first = Dense(C_in=>C_width)
#         last = Dense(C_width=>C_out)
#         new(C_in, C_out, C_width, N_block, diffusion_method, blocks, first, last)
#     end
# end

# function (net::DiffusionNet)(input, mesh::Mesh)
#     #  x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None
# end

# Utility methods to pass meshes into network
