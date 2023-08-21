import numpy as np

def matmul(A, B):
    n = A.shape[0]
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += A[i][k] * B[k][j]
    return res

def matvecmul(A, b): 
    C = []
    for row in A:
        rowsum = 0
        for j, val in enumerate(row):
            rowsum += val * b[j]
        C.append(rowsum)
    return np.array(C)

def dot(a, b):
    sum = 0
    for i, val in enumerate(a):
        sum += val * b[i]
    return sum

def norm(v): # L2-Norm of a Vector
    """
    L-2 norm of a vector.

    Parameters:
    v (NDArray): Vector.

    Returns:
    float: L-2 norm.
    """
    sum = 0
    for val in v:
        sum += val*val
    return np.sqrt(sum)

def matrixnorm(A, x):
    """
    Calculates the matrix norm of a vector.

    Parameters:
    A (NDArray): Square matrix.
    x (NDArray): Vector to calculate the norm for.

    Returns:
    float: Matrix-norm.
    """
    return np.sqrt(dot(matvecmul(A, x), x))

def back_sub(U, y): # Solves vector x for linear system Ux = y. Where U is an upper triangular matrix.
    """
    Solves vector x for linear system Ux = y. Where U is an upper triangular matrix.

    Parameters:
    U (NDArray): Upper triangular matrix.
    y (NDArray): Right hand side vector.

    Returns:
    NDArray: Solution vector x.
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sub = y[i]
        for j in range(i+1, n):
            sub -= U[i,j] * x[j]
        x[i] = sub / U[i,i]
    return x

def forth_sub_inv(L): # Inverts lower triangular matrix.
    """
    Inverts a lower triangular matrix.

    Parameters:
    L (NDArray): Lower triangular matrix.

    Returns:
    NDArray: Inverse of the input.
    """
    n = L.shape[0]
    L_inv = np.zeros((n,n))
    I = np.eye(n)
    for i in range(n):
        for j in range(i+1):
             L_inv[i,j] = (I[i,j] - dot(L[i,:i], L_inv[:i,j])) / L[i,i]
    return L_inv

def back_sub_inv(U): # Inverts upper triangular matrix.
    """
    Inverts an upper triangular matrix.

    Parameters:
    U (NDArray): Upper triangular matrix.

    Returns:
    NDArray: Inverse of the input.
    """
    n = U.shape[0] 
    U_inv = np.zeros((n,n)) 
    I = np.eye(n) 
    for i in range(n-1, -1, -1):
        for j in range(n-1, i-1, -1):
            U_inv[i,j] = (I[i,j] - dot(U[i,i:], U_inv[i:,j])) / U[i,i]
    return U_inv

def ILU0(A): # Performs ILU0 decomposition on matrix A.
    """
    Incomplete decomposition a square matrix to lower and upper triangular matrices.

    Parameters:
    A (NDArray): Square matrix.

    Returns:
    NDArray: Lower triangular matrix part.
    NDArray: Upper triangular matrix part.
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for k in range(n):
        L[k, k] = 1
        for j in range(k+1, n):
            L[j, k] = A[j, k] / A[k, k]
            for i in range(k+1, n):
                A[j, i] -= L[j, k] * A[k, i]
        for i in range(k, n):
            U[k, i] = A[k, i]
    return L, U
