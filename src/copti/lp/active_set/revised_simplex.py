from typing import List, Tuple
from numpy import inf
from numpy.typing import NDArray
import numpy as np
from scipy.optimize import linprog

def find_feasible_solution(A: NDArray[np.float64], b: NDArray[np.float64]) \
            -> Tuple[NDArray[np.float64], List[int], List[int]]:
    """
    Find a feasible point for the system Ax = b with x >= 0,
    or detect infeasibility, by solving a related LP problem.
    
    Args:
    ----------
    A : NDArray[np.float64]
        Coefficient matrix.
    b : NDArray[np.float64]
        Right-hand side vector.
    
    Returns:
    ----------
    x_feasible : NDArray[np.float64]
        Feasible point for the system Ax = b with x >= 0.
    N : List[int]
        Indices of the non-basic set.
    B : List[int]
        Indices of the basic set.

    Raises:
    ----------
    ValueError
        If the problem is infeasible.
    """
    m, n = A.shape
    
    # Objective function: minimize the sum of s and t (e's are ones)
    c = np.concatenate([np.zeros(n), np.ones(m), np.ones(m)])  # Coefficients for x are 0, s and t are 1
    
    # Constraint matrix: Ax + s - t = b
    # Note: np.hstack and np.vstack are used to construct the augmented matrices
    A_eq = np.hstack([A, np.eye(m), -np.eye(m)])
    
    # Bounds for x, s, t >= 0
    bounds = [(0, None)] * (n + 2*m)
    
    # Solve the LP
    result = linprog(c, A_eq=A_eq, b_eq=b, bounds=bounds, method='highs')
    
    if result.success:
        # Extract the solution for x (first n variables)
        x_feasible = result.x[:n]
        #Determine the non-basic set N from x_N = 0
        N = list(np.where(x_feasible == 0)[0])
        #Determine the basic set B from x_B > 0
        B = list(np.where(x_feasible > 0)[0])

        return x_feasible, N, B
    else:
        raise ValueError("The problem is infeasible.")

def solve(x0: NDArray[np.float64], A: NDArray[np.float64], 
                    g: NDArray[np.float64], B: List[int], N: List[int]) -> Tuple[NDArray[np.float64], bool]:
    """
    Implements the Revised Simplex Algorithm to solve LP optimization problems using an initial feasible point, the constraint matrix, and gradient.

    Args:
    ----------
    x0 : NDArray[np.float64]
        Initial feasible point.
    A : NDArray[np.float64]
        Constraint matrix.
    g : NDArray[np.float64]
        Gradient vector.
    B : List[int]
        Indices of the basic variables.
    N : List[int]
        Indices of the non-basic variables.

    Returns:
    ---------
    x : NDArray[np.float64]
        Optimized solution vector.
    optimal : bool
        True if an optimal solution is found, False if unbounded.
    """
    x = x0.copy()
    optimal = False
    while True:
        # Construct basis B and non-basis N matrices
        B_matrix = A[:, B]
        N_matrix = A[:, N]
        
        # Calculate the basic and non-basic gradients
        gB = g[B]
        gN = g[N]

        # Solve for mu: B_matrix*mu = gB
        mu = np.linalg.solve(B_matrix, gB)
        
        # Compute: lambda_ = gN - N_matrix*mu
        lambda_ = gN - N_matrix @ mu
        
        # Optimality condition
        if np.all(lambda_ >= 0):
            optimal = True
            break
        
        # Entering variable (index in N)
        s = np.argmin(lambda_)
        entering_index = N[s]
        
        # Solve for direction d: B_matrix*d = A[:, entering_index]
        d = np.linalg.solve(B_matrix, A[:, entering_index])
        
        # Determine leaving variable
        ratios = np.where(d > 0, x[B] / d, np.inf)
        j = np.argmin(ratios)
        if ratios[j] == inf:
            # Problem is unbounded
            break
        
        # Update x
        alpha = ratios[j]
        x[B] -= alpha * d
        x[entering_index] = alpha
        
        # Update B and N sets
        leaving_index = B[j]
        B[j] = entering_index
        N[s] = leaving_index
        
    return x, optimal
