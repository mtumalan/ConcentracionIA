import numpy as np

def givens(F, Q, S):
    m, n = F.shape
    U = np.eye(m)
    for i in range(m):
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
            B = np.array([[c, s], [-s, c]])
            F[[i, j]] = B @ F[[i, j]]
            U[[i, j]] = B.T @ U[[i, j]]
    return U

if __name__ == "__main__":
    F = np.array([[1, 0], [1, 1]]) 
    Q = np.array([[0, 0], [0, 2]])
    S = np.array([[1, 0], [0, 1]])
    U = givens(F, Q, S)
    print(U)