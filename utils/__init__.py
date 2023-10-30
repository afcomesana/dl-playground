import numpy as np
from functools import reduce

def center(data):
    if len(data) == 0: return data
    
    data = data.astype("float")
    
    for col in range(data.shape[1]):
        data[:, col] -= np.mean(data[:, col])
        
    return data
    

def standarize(data = np.array([])):
    if len(data) == 0: return np.array([])
    
    rows, cols = data.shape
    
    # Center each variable to have mean 0 and standard deviation 1
    for col in range(cols):
        col_mean = np.mean(data[:, col])
        col_std  = np.std(data[:, col])
        data[:, col] -= col_mean
        data[:, col] /= col_std
        
    return data


# Theorical approach:
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
# 
# In practice, we compute the pca component analysis by using the singular value decomposition,
# since it is computationally more efficient.
class PCA:
    def __init__(self, ndims=3):
        self.ndims = ndims
    
    def fit(self, data):

        transformed_data = center(data)
        svd = np.linalg.svd(transformed_data)
        
        # Store only the columns and values we are going to use:
        self.U = svd.U[:, :self.ndims] 
        self.singular_values = svd.S[:self.ndims]
        
        # Calculate explained variance:
        eigenvalues                   = np.array(list(map(lambda sg: sg**2/(len(data) - 1), svd.S)))
        total_variance                = sum(eigenvalues)
        self.explained_variance_ratio = np.array(list(map(lambda ev: ev/total_variance * 100, eigenvalues[:self.ndims])))
        
    
    def transform(self):
        return np.matmul(self.U, np.diag(self.singular_values))

def cov(data):
    if len(data) == 0:
        return np.array([])
    
    data = data.astype("float")
    
    rows, cols = data.shape
    
    for col in range(cols):
        data[:, col] -= np.mean(data[:, col])

    cov_matrix = np.matmul(data.T, data)
    cov_matrix /= rows-1
    
    return cov_matrix
            
            
            