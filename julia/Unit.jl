# UNIT STRUCTURE
# The weights and bias are going to be updated, so it must a mutable struct
mutable struct Unit
    const activation::Function # default value identity function
    weights::Vector{Float64}   # default value small positive numbers
    bias::Float64              # default value small positive number

    # Constructor, apply default values
    function Unit(activation::Function = identity, weights::Vector{Float64} = Vector{Float64}(), bias = rand())
        new(activation, weights, bias)
    end
end

# Use multiple dispatch to customize get_output function for this custom type
function get_output(input::Vector{Float64}, unit::Unit)::Float64
    unit = check_computability(input, unit)

    # Compute result for current unit:
    return unit.activation( dot( input, unit.weights ) + unit.bias )
end

# The first time the network runs, unless provided, there will be no weights
# (since we do not know how many weights the unit needs, because there is no input)
function initialize(input::Vector{Float64}, unit::Unit)
    unit.weights = rand(length(input))
    return unit
end

# 
function check_computability(input::Vector{Float64}, unit::Unit)
    # It is the first time that the unit is used:
    if ( isempty( unit.weights ) )
        return initialize(input, unit)
    end

    if ( length(unit.weights) !== length(input) )
        Base.error("Dimension mismatch: input vector and unit weights vector are have not the same length.")
    end
end