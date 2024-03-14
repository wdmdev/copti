from numpy.typing import NDArray
import numpy as np
import scipy

def LU_solver(H: NDArray[np.float64], g: NDArray[np.float64],
                A: NDArray[np.float64], b: NDArray[np.float64]) \
                 -> NDArray[np.float64]:
    """Solve the system using LU factorization.

    Args:
    ----------
    H : NDArray[np.float64]
        The Hessian matrix of the quadratic problem.
    g : NDArray[np.float64]
        The gradient of the quadratic problem.
    A : NDArray[np.float64]
        The constraint matrix A of the system.
    b : NDArray[np.float64]
        The right hand side constraint vector b of the system.

    Returns:
    ----------
    NDArray[np.float64]
        The solution of the system.
    """
    d = - np.concatenate((g, b), axis=None)
    K = np.block([[H, -A.T], 
                  [-A, np.zeros((A.shape[0], A.shape[0]))]])

    # Perform LU decomposition
    P, L, U = scipy.linalg.lu(K, permute_l=False) #type: ignore

    # Rearrange d using the permutation matrix P
    d = np.dot(P, d)

    # Solve the system of linear equations
    z = np.linalg.solve(U, np.linalg.solve(L, d))

    return z

def range_space_solver(H: NDArray[np.float64], g: NDArray[np.float64], 
                       A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the system using the range space of the constraint matrix.

    Args:
    ----------
    H: NDArray[np.float64]
        The Hessian matrix of the quadratic problem.
    g: NDArray[np.float64]
        The gradient of the quadratic problem.
    A: NDArray[np.float64]
        The constraint matrix A of the system.
    b: NDArray[np.float64]
        The right hand side constraint vector b of the system.

    Returns:
    ----------
    NDArray[np.float64]
        The solution of the system.
    """
    # Step 1: Cholesky factorize H = L L^T
    L = np.linalg.cholesky(H)
    
    # Step 2: Solve H v = g for v (to be used in step 4)
    v = np.linalg.solve(H, g)
    
    # Steps for computing Y = H^{-1} A indirectly using Cholesky factors
    Z = np.linalg.solve(L.T, A.T)  # Solves LZ = A for Z
    Y = np.linalg.solve(L.T, Z)  # Solves L^T Y = Z for Y, giving Y = H^{-1} A

    # Step 3: Form H_A = A^T Y correctly
    H_A = A @ Y
    
    # Step 4: Solve H_A lambda = b + A^T v for lambda
    lambda_ = np.linalg.solve(H_A, b + A @ v)
    
    # Step 5: Solve H x = A lambda - g for x
    x = np.linalg.solve(H, A.T @ lambda_ - g)

    return np.concatenate([x, lambda_], axis=None)