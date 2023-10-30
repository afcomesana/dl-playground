import numpy as np

# 1. Center each variable to have mean = 0.
# 2. Compute covariance matrix.
# 3. Get the eigenvectors (the eigenvector with the highest eigenvalue will be the
# first component, the second highest the second component, and so on).
# 4. Select the desired amount of components (each eigenvector will have the same amount of
# items as colum    ns -variables- has the data matrix).
# 5. Normalize each vector so that it has norm = 1.
# 6. Perform matrix multiplication:
# - X (the data): originally m x n - m samples with n variables
# - W (weights of the transformation): d x n - d selected components and n variables
# So it must be X * W' = X_reduced
def pca(data = np.array([])):
    if len(data) == 0: return np.array([])

    data = data.astype("float")
    
    rows, cols = data.shape
    
    # Center each variable to have mean 0
    for col in range(cols):
        data[:, col] -= np.mean(data[:, col])
        
    data = np.cov(data, rowvar=False)
    
    print(np.linalg.eig(data))
    
    
    return data
    
    