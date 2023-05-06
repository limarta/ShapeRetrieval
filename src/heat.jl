# export heat_integrator, heat_diffusion
function heat_integrator(L, A, signal; dt=0.001, steps=100)
    M = spdiagm(A)
    D = lu(M+dt*L)
    heat = signal
    for t=1:steps
        heat = D \ (heat .* A)
    end
    return heat
end

heat_integrator(mesh::Mesh, signal; dt=0.001, steps=100) = heat_integrator(mesh.cot_laplacian, mesh.vertex_area, signal, dt=dt, steps=steps) 

function heat_diffusion(L, A, init, t; k=200)
    λ, ϕ = eigs(L ./ A, nev=k, sigma=1e-8)
    heat_diffusion(λ, ϕ, A, init, t)
end

function heat_diffusion(λ::Vector{T}, ϕ::Matrix{T}, A::Vector{Float64}, init, t) where T <: Union{ComplexF64, Float64}
    # init - |V| or |V|×|C|
    # note that init may be a single vector or length(t) vectors. If it is a single vector, then heat is diffused for each time t. If it is
    # multiple vectors, then vector init[i] is diffused for time t[i]
    c = ϕ'*(A .* init) .* exp.(-λ * t')
    heat = abs.(ϕ * c)
end

heat_diffusion(mesh::Mesh, init, t::Float64, k=200) = heat_diffusion(mesh.cot_laplacian, mesh.vertex_area, init, t, k=k)

function get_spectrum(mesh::Mesh; k=200)
    L = mesh.cot_laplacian ./ mesh.vertex_area
    λ, ϕ = eigs(L, nev=k, sigma=1e-8)
end