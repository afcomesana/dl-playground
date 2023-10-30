using LinearAlgebra

# Rectified linear activation function:
function relu(x::Number)::Number
    return max(0, x)
end

# Sigmoid function:
function σ(x::Number)::Float64
    return 1/(1+exp(-x))
end