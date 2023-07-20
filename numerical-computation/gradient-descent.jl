import LinearAlgebra
import Zygote
f(x) = x^2 + x + 1

function gradient_descent(f::Function, initial_points)
    step = 0.1
    tolerance = 0.0001

    gradient = Zygote.gradient(f, initial_points)

    while LinearAlgebra.norm(gradient) > tolerance
        initial_points = initial_points .- step.*gradient
        println(initial_points)
        gradient = Zygote.gradient(f, initial_points...)
    end

    return initial_points
    
end

gradient_descent(f, 3)