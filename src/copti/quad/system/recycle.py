"""Equality Constraint Quadratic Optimization module."""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from scipy.linalg import lu, ldl, solve
import scipy.linalg

def matrix_form(n:int, u_bar: float, d0: float) -> \
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """resycle system for equality constraint convex quadratic optimization.

    The problem is defined as the following minimization problem:
    min_u 1/2 * sum_{i=1}^{n+1} (u_i-u_bar)^2

    s.t.    -u_1 + u_n              = -d0
            u_i - u_{i+1}           = 0
            u_{n-1} - u_n - u_{n+1} = 0
    
    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.

    Returns:
    ----------
    H: NDArray[np.float64]
        The Hessian matrix of the system.
    g: NDArray[np.float64]
        The gradient vector of the system.
    A: NDArray[np.float64]
        The constraint matrix A of the system.
    b: NDArray[np.float64]
        The right hand side constraint vector b of the system.
    """

    H = np.eye(n+1) #n+1 x n+1 identity matrix
    
    g = np.full(n+1, -u_bar)
    
    #Create the A matrix
    A = np.zeros((n+1, n+1))
    
    # Setting the first constraint -u1 + u10 = -d0
    A[0, 0] = -1   # for -u1
    A[0, n-1] = 1  # for u10
    
    # Setting the u_i - u_{i+1} = 0 constraints for i=1 to n-1
    for i in range(1, n):
        A[i, i-1] = 1   # for u_i
        A[i, i] = -1    # for -u_{i+1}
    
    # Last constraint: u_{9} - u_{10} - u_{11} = 0
    A[n, n-2] = 1   # for u_{9}
    A[n, n-1] = -1  # for -u_{10}
    A[n, n] = -1    # for -u_{11}
    
    
    b0 = np.array([
        [-d0],
    ])
    b = np.concatenate((b0, np.zeros((n, 1))), axis=0)

    return H, g, A, b

def KKT_matrix(n:int, u_bar: float, d0: float) -> NDArray[np.float64]:
    """Resycle KKT matrix for equality constraint convex quadratic optimization.

    The problem is defined as the following minimization problem:
    min_u 1/2 * sum_{i=1}^{n+1} (u_i-u_bar)^2

    s.t.    -u_1 + u_n              = -d0
            u_i - u_{i+1}           = 0
            u_{n-1} - u_n - u_{n+1} = 0
    
    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.

    Returns:
    ----------
    KKT: NDArray[np.float64]
        The KKT matrix of the system.
    """

    H, _, A, _ = matrix_form(n, u_bar, d0)
    A_T = A.T
    m = A.shape[0]
    KKT = np.block([
        [H, -A],
        [-A_T, np.zeros((m, m))]
    ])

    return KKT

def LU_solver(n:int, u_bar: float, d0: float) -> NDArray[np.float64]:
    """Solve the system using LU factorization.

    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.
    
    Returns:
    ----------
    z: NDArray[np.float64]
        The solution of the system.
    """
    _, g, _, b = matrix_form(n, u_bar, d0)
    d = - np.concatenate((g, b), axis=None)
    K = KKT_matrix(n, u_bar, d0)

    # Perform LU decomposition
    P, L, U = lu(K, permute_l=False) #type: ignore

    # Rearrange d using the permutation matrix P
    d = np.dot(P, d)

    # Solve the system of linear equations
    z = np.linalg.solve(U, np.linalg.solve(L, d))

    return z

def LDL_solver(n:int, u_bar: float, d0: float) -> NDArray[np.float64]:
    """Solve the system using LDL factorization.

    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.

    Returns:
    ----------
    z: NDArray[np.float64]
        The solution of the system.
    """
    _, g, _, b = matrix_form(n, u_bar, d0)
    d = - np.concatenate((g, b), axis=None)
    K = KKT_matrix(n, u_bar, d0)

    # Perform LDL decomposition
    L, D, perm = ldl(K)
    P = np.eye(perm.shape[0])[perm]

    # Rearrange d using the permutation matrix P
    d = np.dot(P, d)

    # Solve the system of linear equations
    z = np.linalg.solve(D, np.linalg.solve(L, d))

    return z

def QR_null_space_solver(n:int, u_bar: float, d0: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve the system using QR factorization and the null space.

    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.

    Returns:
    ----------
    x : NDArray[np.float64]
        The solution of the system.
    lambda_ : NDArray[np.float64]
        The Lagrange multipliers of the system.
    """
    H, g, A, b = matrix_form(n, u_bar, d0)

    # Step 1: QR factorization of A^T to find Y and Z
    Q, R = np.linalg.qr(A.T, mode='complete')
    m = A.shape[0]  # Number of constraints
    Y = Q[:, :m]
    Z = Q[:, m:]
    
    # Step 2: Solve for p_Y using (AY)p_Y = b - AY(AY)^{-1}AY^Tg
    AY = A @ Y
    # Adjusted computation of p_Y to handle singular or nearly singular matrices
    x_Y = np.linalg.pinv(AY @ AY.T) @ (b - AY @ np.linalg.pinv(AY.T) @ (AY @ Y.T @ g))

    
    # Step 3: Solve for p_Z using the reduced system (Z^THZ)p_Z = Z^T(HYp_Y - g)
    ZTHZ = Z.T @ H @ Z
    rhs = Z.T @ (H @ Y @ x_Y - g)
    x_Z = solve(ZTHZ, rhs, assume_a='pos')
    
    # Step 4: Compute the total step p = Yp_Y + Zp_Z
    x = Y @ x_Y + Z @ x_Z
    
    # Step 5: Obtain the Lagrange multipliers lambda
    lambda_ = np.linalg.solve(AY @ AY.T, AY @ (g + H @ x))
    
    return x, lambda_

def range_space_solver(n:int, u_bar: float, d0: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve the system using the range space of the constraint matrix.

    Args:
    ----------
    n: int
        Number of variables in the optimization problem.
    u_bar: float
        The mean value of the variables.
    d0: float
        The value of the first constraint.

    Returns:
    ----------
    x : NDArray[np.float64]
        The solution of the system.
    lambda_ : NDArray[np.float64]
        The Lagrange multipliers of the system.
    """
    H, g, A, b = matrix_form(n, u_bar, d0)

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
    rhs_lambda = b + A.T @ v
    lambda_ = scipy.linalg.cho_solve((L_A, True), rhs_lambda)
    
    # Step 5: Solve H * x = A * lambda - g for x
    x = scipy.linalg.cho_solve((L, True), A @ lambda_ - g)
    
    return x, lambda_