using Revise
Revise.revise()
import ShapeRetrieval: ShapeRetrieval as SR
using LinearAlgebra
using SparseArrays
using Flux
using CUDA

bunny = SR.load_obj("./meshes/gourd.obj")
bunny = SR.normalize_mesh(bunny)
println("Vertices $(bunny.nv) Faces $(bunny.nf)")
L, A, λ, ϕ, ∇_x, ∇_y = SR.get_operators(bunny) |> gpu
println("Done with operators")

function diffusion_loss(model, x, ϕ, λ, A, y) 
    norm(model(x,ϕ,λ,A) - y)
end
heat_init = zeros(Float32, bunny.nv,3)
heat_init[1,1] = 1.0 #  can be|V|×|C| input
heat_init[200,2] = 1.0
heat_init[140,3] = 1.0; heat_init[300,3] = 1.0
heat_init = heat_init |> gpu

m = SR.LearnedTimeDiffusionBlock(3) |> gpu
@time y_true = m(heat_init, λ,ϕ, A);
println("t_true: ", m.diffusion_time)
m = SR.LearnedTimeDiffusionBlock(3) |> gpu
println("t_start: ", m.diffusion_time)
println("start cost ", diffusion_loss(m, heat_init, λ, ϕ, A, y_true))

# opt_state = Flux.setup(Adam(), m)
# @time for i=1:10000
#     grad = gradient(diffusion_loss, m, heat_init, λ, ϕ, A, y_true)
#     Flux.update!(opt_state, m, grad[1])
# end
# println("final cost : ",diffusion_loss(m,heat_init, λ, ϕ, A, y_true))
# println("t_final: ", m.diffusion_time)
# # predicted_heat = m(heat_init, λ, ϕ, A)
# # heat_viz = [y_true predicted_heat]
# # SR.viz_grid(bunny.V, bunny.F, heat_viz; dims=(2,3))