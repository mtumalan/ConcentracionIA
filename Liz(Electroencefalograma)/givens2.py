import numpy as np

def givens(F, S, Q):
    m, n = F.shape
    U = np.eye(m)
    
    # Apply scaling if S is provided (S must be conformable with F)
    F = S @ F  # Apply scaling matrix S to F
    
    for i in range(n):
        for j in range(i + 1, m):
            a = F[i, i]
            b = F[j, i]
            if b == 0:
                c = 1
                s = 0
            else:
                if np.abs(b) > np.abs(a):
                    r = a / b
                    s = 1 / np.sqrt(1 + r**2)
                    c = s * r
                else:
                    r = b / a
                    c = 1 / np.sqrt(1 + r**2)
                    s = c * r
            # Create Givens rotation matrix
            G = np.eye(m)
            G[i, i] = c
            G[i, j] = s
            G[j, i] = -s
            G[j, j] = c

            # Apply the Givens rotation to F
            F = G @ F
            # Accumulate the rotations in U
            U = U @ G.T

    # After applying Givens rotations, we can directly manipulate R if needed
    # Here we return U and R
    return U, F  # F should now be upper triangular

if __name__ == "__main__":
    F = np.array([[1, 0], [1, 1]], dtype=float)  # Matrix F
    S = np.array([[1, 0], [0, 1]], dtype=float)  # Identity scaling matrix
    Q = np.array([[0, 0], [0, 2]], dtype=float)  # Matrix Q to apply (not directly used)

    U, R = givens(F, S, Q)
    
    print(" ")
    print(R)  # This should output the modified upper triangular matrix
    print(" ")