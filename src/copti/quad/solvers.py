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
                       A: NDArray[np.float64], b: NDArray[np.float64]) \
                        -> NDArray[np.float64]:
    """Solve the system using the range space of the constraint matrix.

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

     # Step 1: Cholesky factorize H = L * L'
    L = scipy.linalg.cholesky(H, lower=True)
    
    # Step 2: Solve H*v = g for v
    v = scipy.linalg.cho_solve((L, True), g)
    
    # Step 3: Form H_A = A' * H^-1 * A = L_A * L_A' and its factorization
    # First solve L * Y = A.T for Y using forward substitution to avoid forming H^-1
    Y = scipy.linalg.solve_triangular(L, A.T, lower=True, trans='T')
    H_A = Y.T @ Y  # This is A' * H^-1 * A
    # Cholesky factorize H_A
    L_A = scipy.linalg.cholesky(H_A, lower=True)
    
    # Step 4: Solve H_A * lambda = b + A' * v for lambda
    rhs_lambda = b + A @ v
    lambda_ = scipy.linalg.cho_solve((L_A, True), rhs_lambda)
    
    # Step 5: Solve H * x = A * lambda - g for x
    x = scipy.linalg.cho_solve((L, True), A.T @ lambda_ - g)
    
    return np.concatenate([x, lambda_])