
# LU Decomposition using Doolittle Method
# Application: Solving Linear Equations Ax = b
import numpy as np
def lu_decomposition(A):
    n = len(A)
    # Initialize L and U matrices
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        # Upper Triangular Matrix
        for k in range(i, n):
            sum_upper = 0
            for j in range(i):
                sum_upper += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - sum_upper
        # Lower Triangular Matrix
        L[i][i] = 1  # Diagonal as 1
        for k in range(i + 1, n):
            sum_lower = 0
            for j in range(i):
                sum_lower += L[k][j] * U[j][i]
            L[k][i] = (A[k][i] - sum_lower) / U[i][i]
    return L, U
def forward_substitution(L, b):
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        sum_value = 0
        for j in range(i):
            sum_value += L[i][j] * y[j]
        y[i] = b[i] - sum_value
    return y
def backward_substitution(U, y):
    n = len(U)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_value = 0
        for j in range(i + 1, n):
            sum_value += U[i][j] * x[j]
        x[i] = (y[i] - sum_value) / U[i][i]
    return x
# Input Matrix A
A = np.array([
    [2, -1, -2],
    [-4, 6, 3],
    [-4, -2, 8]
], dtype=float)
# Constant Matrix b
b = np.array([-2, 9, -5], dtype=float)
# Perform LU Decomposition
L, U = lu_decomposition(A)
# Solve Ly = b
y = forward_substitution(L, b)
# Solve Ux = y
x = backward_substitution(U, y)
# Display Results
print("Matrix A:")
print(A)
print("\nLower Triangular Matrix L:")
print(L)
print("\nUpper Triangular Matrix U:")
print(U)
print("\nIntermediate Solution y:")
print(y)
print("\nFinal Solution x:")
print(x)

# QR Decomposition using Gram-Schmidt Method
# Application: Solving Linear System Ax = b
import numpy as np
def qr_decomposition(A):
    n = A.shape[0]
    # Initialize Q and R matrices
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R
def solve_qr(Q, R, b):
    # Compute Qᵀb
    y = np.dot(Q.T, b)
    # Backward substitution to solve Rx = y
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_value = 0
        for j in range(i + 1, n):
            sum_value += R[i, j] * x[j]

        x[i] = (y[i] - sum_value) / R[i, i]
    return x
# Input Matrix A
A = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)
# Constant Matrix b
b = np.array([4, 5, 6], dtype=float)
# Perform QR Decomposition
Q, R = qr_decomposition(A)
# Solve system using QR decomposition
x = solve_qr(Q, R, b)
# Display Results
print("Matrix A:")
print(A)
print("\nOrthogonal Matrix Q:")
print(Q)
print("\nUpper Triangular Matrix R:")
print(R)
print("\nSolution Vector x:")
print(x)