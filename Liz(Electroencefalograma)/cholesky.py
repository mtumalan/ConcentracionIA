import numpy as np

def cholesky(P):
    # Get the number of rows
    n = P.shape[0]
    
    # Initialize the lower triangular matrix S with zeros
    S = np.zeros_like(P)
    
    # Perform the decomposition
    for i in range(n):
        # Calculate S[i, i]
        S[i, i] = np.sqrt(P[i, i] - np.sum(S[i, :i] ** 2))
        
        # Calculate S[j, i] for j > i
        for j in range(i + 1, n):
            S[j, i] = (P[j, i] - np.sum(S[j, :i] * S[i, :i])) / S[i, i]
    
    return S

# Example use case:
# A = np.array([[36, 18, 12, 6],
#               [18, 25, 10, 15],
#               [12, 10, 16,  4],
#               [6,  15,  4, 20]])

P = np.array([[2, 1, 0, 0],
              [1, 2, 1, 0],
              [0, 1, 2, 1],
              [0, 0, 1, 2]])

L = cholesky(P)
print("Lower Triangular Matrix L:\n", L)
print("Reconstructed Matrix A (L * L.T):\n", np.dot(L, L.T))