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

function heat_diffusion(mesh::Mesh, L, A, signal; t=1.0, k=200)
    L = L ./ A # Brittle? Consider Cholesky decomposition for precomputation? Or just use eigen?
    heat = zeros(mesh.nv)
    λ, ϕ = eigs(L, nev=k, sigma=1e-8)
    c = ϕ'*(signal .* A) .*exp.(-t* λ)
    heat = abs.(ϕ * c)
end
function heat_diffusion(λ::Vector{ComplexF64}, ϕ::Matrix{ComplexF64}, A::Vector{Float64}, init, t)
    # init - |V| or |V|×|C|
    c = ϕ'*(A .* init) .* exp.(-λ * t')
    heat = abs.(ϕ * c)
end

# function heat_diffusion(mesh::Mesh, L, U, A, signal, t, k) end # LU Decomposition
heat_diffusion(mesh::Mesh, signal; t=1, k=200) = heat_diffusion(mesh, cot_laplacian(mesh), vertex_area(mesh), signal, t=t, k=k)

function get_spectrum(mesh::Mesh; k=200)
    L = mesh.cot_laplacian ./ mesh.vertex_area
    λ, ϕ = eigs(L, nev=k, sigma=1e-8)
end