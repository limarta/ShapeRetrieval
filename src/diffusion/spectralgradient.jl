export SpatialGradientFeatures

# struct SpatialGradientFeatures
#     C_inout::Int32
#     A::Matrix{Float64}
#     with_gradient_rotations::Bool
# end
# SpatialGradientFeatures(C_inout::Int32, with_gradient_rotations::Bool) = SpatialGradientFeatures(C_inout, rand(C_inout, C_inout), with_gradient_rotations)

# function (model::SpatialGradientFeatures)(x_diffused)
#     # x_diffused |2|×|V|×|C|

#     if model.with_gradient_rotations
#         # vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
#         # vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
#     else
#         # vectorsBreal = model.A(vectors[...,0])
#         # vectorsBimag = model.A(vectors[...,1])
#     end

#     # dots = vectorsA[...,1] * vectorsBreal + vectorsA[...,2] * vectorsBimag

#     # return tanh(dots)
# end

# Consider using Bilinear