import LinearAlgebra
import MultivariateStats
import RDatasets
import Statistics

iris = RDatasets.dataset("datasets", "iris")

Xtr = Matrix(iris[1:2:end, 1:4])'
Xtr_labels = Vector(iris[1:2:end, 5])

Xte = Matrix(iris[2:2:end, 1:4])'
Xte_labels = Vector(iris[2:2:end, 5])


M = MultivariateStats.fit(MultivariateStats.PCA, Xtr; maxoutdim=3)
println("Library principal vars:")
println(LinearAlgebra.eigvecs(M::MultivariateStats.PCA))
println("\n\n\n")
function pca(matrix::AbstractMatrix{<:Number}, dims::Int = -1, percentage = -1.0)
    rows, cols = size(matrix)
    if ( dims > cols )
        throw("Output dimensions must be equal or lower than input dimensions.")
    end

    if ( dims < 1 )
        dims = cols
    end
    eigmatrix = Statistics.cov(matrix)
    evals, evecs = LinearAlgebra.eigen(eigmatrix)
    for evec in eachrow(evecs)
        println(evec)
    end

    sorted_evals_index = sortperm(abs.(evals), rev=true)[1:dims]
    return evecs[:, sorted_evals_index]
    
end
println("Custom principal vars:")
pca(Xtr', 3)
println("\n\n\n")
# evecs'*row