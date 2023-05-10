# struct MLP
#     dense::Vector{Flux.Dense}
#     drop::Vector{Flux.Dropout}
# end
# @Flux.functor MLP

# function MLP(layer_dims::Vector{Int}, dropout::Bool=true, activation=leakyrelu)
#     dense = Flux.Dense[]
#     drop = Flux.Dropout[]
#     for i=1:length(layer_dims)-1
#         if dropout && i > 1
#             push!(drop, Flux.Dropout(0.5))
#         end

#         if i < length(layer_dims)
#             push!(dense, Flux.Dense(layer_dims[i]=>layer_dims[i+1], activation))
#         else
#             push!(dense, Flux.Dense(layer[i] => layer[i+1]))
#         end
#     end
#     MLP(dense, drop)
# end

# function (mlp::MLP)(x)
#     temp = x
#     for i=1:length(mlp.drop)  
#         temp = mlp.drop[i](mlp.dense[i](temp))
#     end
#     mlp.dense[end](temp)
# end

function MLP(layer_dims::Vector{Int}, dropout::Bool=false, activation=tanh)
    layers = Union{Flux.Dropout, Flux.Dense}[]
    for i=1:length(layer_dims)-1
        if dropout && i > 0
            push!(layers, Flux.Dropout(0.5))
        end
        if i < length(layer_dims)
            push!(layers, Flux.Dense(layer_dims[i] => layer_dims[i+1],activation))
        else
            push!(layers, Flux.Dense(layer[i]=>layer[i+1]))
        end
    end
    Flux.Chain(layers...)
end