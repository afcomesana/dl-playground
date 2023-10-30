# NEURAL NETWORK STRUCTURE
struct NeuralNetwork
    layers::Vector{Layer}
end

# Use multiple dispatch to customize get_output function for this custom type
function get_output(input::Union{Float64, Vector{Float64}}, nn::NeuralNetwork, layer_index::Int64 = 1)::Union{Float64, Vector{Float64}}
    # We just computed all the layers in the network:
    if ( layer_index > length(nn.layers) )
        return input
    end

    output = get_output(input, nn.layers[layer_index])
    layer_index += 1

    return get_output( output, nn, layer_index )
end