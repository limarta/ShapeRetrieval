using Flux

# Diffusion block = diffusion + inner product + dense layer
# Diffusion net = Diffusion block * n

# Learnable parameters -> [t (heat integrator), A (equation 5), dense layer] for each block
