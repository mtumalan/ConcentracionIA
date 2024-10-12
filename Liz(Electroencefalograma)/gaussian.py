import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Augment the matrix A with the vector b
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Make the diagonal contain all 1's
        Ab[i] = Ab[i] / Ab[i, i]
        
        # Make the elements below the pivot positions zero
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])
    
    return x

# Example usage
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)
solution = gaussian_elimination(A, b)
print("Solution:", solution)