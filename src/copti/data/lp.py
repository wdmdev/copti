from typing import Tuple
import numpy as np
from numpy import random, dot
from numpy.typing import NDArray

def random_lp_problem(m: int, n: int) -> \
        Tuple[NDArray[np.float64], NDArray[np.float64], 
              NDArray[np.float64], NDArray[np.float64],
              NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates a random LP problem that satisfies the first-order optimality conditions.
    
    Args:
    ----------
    m : int
        The number of rows in matrix A.
    n : int
        The number of columns in matrix A, should be greater than or equal to m.
    
    Returns:
    ----------
    A :         NDArray[np.float64]
                The matrix A in R^(m x n) used in the LP problem, generated randomly.
    b :         NDArray[np.float64]
                The vector b in R^m, derived as Ax, ensuring feasibility.
    c :         NDArray[np.float64]
                The cost vector c in R^n, derived as A'lambda + s, ensuring optimality.
    x :         NDArray[np.float64]
                The decision variable vector x in R^n, partially random and partially zeros, indicating a basic feasible solution.
    lambda_ :   NDArray[np.float64]
                The Lagrange multipliers for the solution
    s:          NDArray[np.float64]
                The slack variables for the solution
    
    """
    if n < m:
        raise ValueError("n must be greater or equal to m for a valid LP problem")

    # Randomly generate matrix A
    A = random.rand(m, n)
    
    # Generate x with first m entries as random positive numbers and the rest 0
    x = random.rand(n)
    x[m:] = 0
    
    # Generate s with first m entries as 0 and the rest as random positive numbers
    s = random.rand(n)
    s[:m] = 0
    
    # Generate a random vector lambda
    lambda_ = random.rand(m)
    
    # Calculate c as A'lambda + s
    c = dot(A.T, lambda_) + s
    
    # Calculate b as Ax
    b = dot(A, x)
    
    return A, b, c, x, lambda_, s