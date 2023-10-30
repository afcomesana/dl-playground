using Distributions
using Plots

include("Activation.jl")
include("Unit.jl")
include("Layer.jl")
include("Network.jl")

# Set up data:
gauss_0 = Normal(1.0, 3.0)
gauss_1 = Normal(5.0, 2.0)

class_0_data = rand(gauss_0, (100, 2))
class_0_data = [class_0_data zeros(size(class_0_data, 1))]

class_1_data = rand(gauss_1, (100, 2))
class_1_data = [class_1_data ones(size(class_1_data, 1))]

data = [class_0_data; class_1_data]

# Implement manual neural neutwork:

# Simple neural network
hidden = Layer(4, relu) # hidden layer
output = Layer(1, σ)    # output layer

nn = NeuralNetwork([hidden, output])

# println(get_output( input, nn ))