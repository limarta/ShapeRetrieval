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