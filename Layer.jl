# LAYER STRUCTURE
struct Layer
    units_number::Int
    activation::Function
    units::Vector{Unit}

    function Layer( units_number::Int = 1, activation::Function = identity, units::Vector{Unit} = Vector{Unit}() )
        units = [Unit(activation) for i in 1:units_number]
        new(units_number, activation, units)
    end
end

# Use multiple dispatch to customize get_output function for this custom type
function get_output(input::Vector{Float64}, layer::Layer)::Union{Float64, Vector{Float64}}
    if ( length(layer.units) === 1 )
        return get_output(input, layer.units[1])
    end

    return [get_output(input, unit) for unit in layer.units]
end