import ShapeRetrieval: ShapeRetrieval as SR
using LinearAlgebra
using SparseArrays
using Flux
using CUDA

bunny = SR.load_obj("./meshes/bunny.obj")
bunny = SR.normalize_mesh(bunny)
println("Vertices $(bunny.nv) Faces $(bunny.nf)")
L, A, λ, ϕ, ∇_x, ∇_y = SR.get_operators(bunny) |> gpu
println("Done with operators")

println("Create the data")
xyz = convert.(Float32,bunny.V)' |> gpu
y = copy(bunny.V)
y .-= minimum.(eachrow(y))
y ./= maximum.(eachrow(y))
y = convert.(Float32, (y.-0.5)) |> gpu

function diffusion_loss(model, x, λ, ϕ, A,∇_x, ∇_y, y)
    y_pred = model(x, λ, ϕ, A, ∇_x, ∇_y)
    norm(y - y_pred)
end

dnb = SR.DiffusionNetBlock(3,[3], with_gradient_features=true) |> gpu
# # # opt_state = Flux.setup(Adam(), dnb)
# # # println(dnb)
# println("init cost: ", diffusion_loss(dnb, xyz, λ,ϕ, A,∇_x, ∇_y, y))
# # # @code_warntype dnb(xyz, λ,ϕ, A,∇_x, ∇_y)
# # # @time for i=1:1
# # #     grad = gradient(diffusion_loss, dnb, xyz, λ, ϕ, A, ∇_x, ∇_y, y)
# # #     if i % 1000 == 0
# # #         println(diffusion_loss(dnb, xyz, λ, ϕ, A, ∇_x, ∇_y,y))
# # #     end
# # #     Flux.update!(opt_state, dnb, grad[1])
# # # end

# # println("final cost : ",diffusion_loss(dnb, xyz, λ, ϕ, A, ∇_x, ∇_y,y))

y_pred = dnb(xyz, λ, ϕ, A, ∇_x, ∇_y)
# # # @code_warntype dnb(xyz, λ, ϕ, A, ∇_x, ∇_y)
# # # fig = SR.meshviz(bunny, color=y_pred[3,:], shift_coordinates=true, resolution=(1500,1500))
# # # y_0, y_1, y_2 = SR.diffusion_explain(dnb, xyz, λ, ϕ, A, ∇_x, ∇_y)
# # # println(dnb)

# # # fig = SR.viz_grid(bunny.V, bunny.F, [y; y_0'; y_2]', shift_coordinates=false, dims=(3,3))

# # x_fake = cu(rand())