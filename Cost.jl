# Single values cross entropy
function cross_entropy(predicted::Float64, target::Float64)::Float64
    # Prevent domain error:
    if ( iszero(predicted) ); predicted += eps(); end


    return -(target*log(predicted))
end

# Vector values cross entropy
function cross_entropy(predicted::Vector{Float64}, target::Vector{Float64})::Vector{Float64}
    return map(cross_entropy, predicted, target)
end

# Single values cross entropy
function binary_cross_entropy(predicted::Float64, target::Float64)::Float64
    expected_dist  = [1.0 - target, target]
    predicted_dist = [1.0 - predicted, predicted]

    return sum(cross_entropy(predicted_dist, expected_dist))
end

# Vector values cross entropy
function binary_cross_entropy(predicted::Vector{Float64}, target::Vector{Float64})::Vector{Float64}
    return map(binary_cross_entropy, predicted, target)
end

function ∂_binary_cross_entropy_∂_predicted(predicted::Float64, target::Float64)::Float64
    # Prevent domain errors:
    den_1 = isone(predicted)  ? eps() : 1 - predicted
    den_2 = iszero(predicted) ? eps() : predicted

    return ((1 - target)/den_1) - (target/den_2)
end