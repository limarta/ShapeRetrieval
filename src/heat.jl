# export heat_integrator, heat_diffusion
function heat_integrator(mesh, L, A, signal; dt=0.001, steps=100)
    M = spdiagm(A)
    D = lu(M-dt*L)
    heat = signal
    for t in 1:steps
        heat = D \ (heat .* A)
    end
    return heat
end
heat_integrator(mesh::Mesh, signal; dt=0.001, steps=100) = heat_integrator(mesh, cot_laplacian(mesh), vertex_area(mesh), signal, dt=dt, steps=steps) 

function heat_diffusion(mesh::Mesh, L, A, signal; t=1.0, k=200)
    M = spdiagm(A)
    L = L ./ A # Brittle? Consider Cholesky decomposition for precomputation? Or just use eigen?
    heat = zeros(mesh.nv)
    λ, ϕ = eigs(L, nev=k, sigma=1e-8)
    c = ϕ'*(signal .* A) .*exp.(-t* λ)
    heat = ϕ * c
    heat = abs.(heat)
end
# function heat_diffusion(mesh::Mesh, L, U, A, signal, t, k) end # LU Decomposition
heat_diffusion(mesh::Mesh, signal; t=1, k=200) = heat_diffusion(mesh, cot_laplacian(mesh), vertex_area(mesh), signal, t=t, k=k)
unit_heat_diffusion(mesh::Mesh, v::Int; t=1, k=200) = heat_diffusion(mesh, [], t, k)