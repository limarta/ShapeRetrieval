using Flux

# Diffusion block = diffusion + inner product + dense layer
# Diffusion net = Diffusion block * n

# Learnable parameters -> [t (heat integrator), A (equation 5), dense layer] for each block

using Flux
using heat
using LinearAlgebra

# Diffusion block = diffusion + inner product + dense layer
# Diffusion net = Diffusion block * n

# Learnable parameters -> [t (heat integrator), A (equation 5), dense layer] for each block

struct LearnedTimeDiffusionBlock
    C_inout::Int32
    diffusion_time::Vector{Float64}
end

Flux.@functor LearnedTimeDiffusionBlock

function (model::LearnedTimeDiffusionBlock)(x, L, M)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    V = size(x, 2)

    # Form the dense matrices (M + tL) with dims (B,C,V,V)
    # L.to_dense().unsqueeze(2).expand(-1, model.C_inout, V, V)
    mat_dense = L.to_dense().unsqueeze(2).expand(-1, model.C_inout, V, V)
    mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mat_dense += torch.diag_embed(mass).unsqueeze(1)

    # Factor the system
    cholesky_factors = cholesky(mat_dense)
    
    # Solve the system
    rhs = x * mass.unsqueeze(-1)
    rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
    sols = torch.cholesky_solve(rhsT, cholesky_factors)
    x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)
    return x_diffuse
end

struct SpatialGradientFeatures:
    A::Matrix{Float64}
end

function (model::SpatialGradientFeatures)(vectors)
    vectorsA = vectors # (V,C)

    # if self.with_gradient_rotations:
    #     vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
    #     vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
    # else:
    vectorsBreal = self.A(vectors[...,0])
    vectorsBimag = self.A(vectors[...,1])

    dots = vectorsA[...,1] * vectorsBreal + vectorsA[...,2] * vectorsBimag

    return tanh(dots)
end

struct MiniMLP

end

struct DiffusionNetBlock
end

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out