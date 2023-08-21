import numpy as np
from utils import *

def read_msr(filepath = "test/gmres_test_msr.txt"):
    """
    Reads a text file containing JM and VM of a sparse square matrix in MSR format.

    Example file is given with the project.

    Parameters:
    filepath (string): The filepath.

    Returns:
    tuple: JM array, VM array, and square matrix dimension.
    """
    msr_file = open(filepath, 'r')
    lines = msr_file.readlines()
    msr_file.close()
    flag = lines[0].strip()
    dim, size = lines[1].strip().split()
    dim = int(dim)
    size = int(size)

    JM = np.zeros(size)
    VM = np.zeros(size)
    for i,line in enumerate(lines[2:]):
        JM[i] = line.split()[0]
        VM[i] = line.split()[1]
    JM = JM.astype(int) - 1
    return (JM, VM, dim)

def msrmul(JM, VM, x, symmetric = False): # A * x, where A is a matrix in MSR format, and x is a vector.
    """
    Multiplies a matrix in MSR format with a vector.

    Parameters:
    JM (NDArray): Index/Column vector.
    VM (NDArray): Value vector.
    x (NDArray): Vector to be multiplied.
    symmetric (bool): Whether the compressed matrix is originally symmetric or not.

    Returns:
    NDArray: The resulting vector.
    """
    dim = x.size
    y = matvecmul(np.diag(VM[0:dim]), x)
    if (symmetric):
        # y = Dx + Ux + Lx for symmetrical MSR
        for i in np.arange(dim):
            i1 = JM[i]
            i2 = JM[i+1]
            if (i1 == i2): continue
            y[i] +=  dot(VM[i1:i2], x[JM[i1:i2]])
            y[JM[i1:i2]] += VM[i1:i2] * x[i]
    else:
        # y = Ax for non-symmetrical MSR
        for i in np.arange(dim):
            i1 = JM[i]
            i2 = JM[i+1]
            if (i1 == i2): continue
            y[i] += dot(VM[i1:i2], x[JM[i1:i2]])
    return y

def to_msr(A, symmetric = False): # Encodes matrix A into MSR format.
    """
    Compresses a sparse square matrix into MSR format.

    Parameters:
    A (NDArray): Sparse square matrix to be compressed.
    symmetric (bool): Whether the sparse square matrix is symmetric or not.

    Returns:
    NDArray: The resulting JM vector.
    NDArray: The resulting VM vector.
    """
    VM = [A[i,i] for i in np.arange(A.shape[0])]
    VM.append(0)
    IM = [A.shape[0] + 1]
    JM = []
    for i, row in enumerate(A):
        found_values = 0
        for j, val in enumerate(row):
            if (i != j):
                if (val != 0):
                    found_values += 1
                    VM.append(val)
                    JM.append(j)
        IM.append(IM[-1] + found_values)
    if (symmetric):
        repeat_ind = int(len(JM)/2) 
        JM = JM[:repeat_ind]
        VM = VM[:-repeat_ind]
    IM.extend(JM)
    return np.array(IM), np.array(VM) # IM: Index Matrix, as JM finally.

def from_msr(JM, VM, dim):
    """
    Unpacks a sparse square matrix compressed in MSR format.

    Parameters:
    JM (NDArray): Index/Column vector.
    VM (NDArray): Value vector.

    Returns:
    NDArray: The unpacked sparse square matrix.
    """
    dim = np.where(VM == 0)[0][0]
    A = np.zeros((dim,dim))
    I = np.eye(dim)
    for j in range(dim):
        A[:,j] = msrmul(JM, VM, I[:,j])
    return A

def msrnorm(JM, VM, x, symmetric = False): # A norm of vector x in MSR format.
    """
    Calculates the matrix norm of a vector.

    The matrix must be in MSR format.

    Parameters:
    JM (NDArray): Index/Column vector.
    VM (NDArray): Value vector.
    x (NDArray): Vector to calculate the norm for.

    Returns:
    float: Matrix-norm.
    """
    return np.sqrt(dot(msrmul(JM, VM, x, symmetric), x))