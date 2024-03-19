from typing import Tuple, Set
import numpy as np
from numpy.typing import NDArray

def dual_active_set(H: NDArray[np.float64], g: NDArray[np.float64], 
                              A: NDArray[np.float64], b: NDArray[np.float64]) \
                                -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Set[int]]:
    """
    Solves a convex quadratic programming problem using the Dual Active Set Algorithm.
    
    Args:
    ----------
    H (NDArray[np.float64]): Symmetric positive definite matrix in the quadratic term of the objective function.
    g (NDArray[np.float64]): Coefficient vector in the linear term of the objective function.
    A (NDArray[np.float64]): Constraint matrix.
    b (NDArray[np.float64]): Constraint vector.
    
    Returns:
    ----------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Set[int]]: 
        A tuple containing the solution vector x, the Lagrange multipliers mu, the path of solutions sol_path, 
        and the set of active constraints W at the solution.
    """
    # Initial setup
    sol_path = np.array([])
    x = -np.linalg.inv(H) @ g
    sol_path = np.append(sol_path, x)
    mu = np.zeros(A.shape[0])
    W = set()

    # Main loop
    while True:
        # Check optimality
        r = np.where(A @ x - b < 0)[0]
        if len(r) == 0:
            break  # Optimal solution found
        r = r[0]  # Select a violating constraint

        while A[r, :] @ x - b[r] < 0:
            # Construct the KKT matrix for the current working set W
            AW = A[list(W), :]
            KKT = np.block([[H, -AW.T], [-AW, np.zeros((len(W), len(W)))]])
            rhs = np.vstack([A[r, :].reshape(-1, 1), np.zeros((len(W), 1))])
            solution = np.linalg.solve(KKT, rhs)
            p = solution[:H.shape[0]]
            v = solution[H.shape[0]:]

            if np.allclose(p, 0):
                if np.all(v >= 0):
                    raise ValueError("The problem is infeasible.")
                else:
                    # Remove a constraint from the working set
                    j_star = np.argmin(-mu[list(W)] / v.flatten())
                    W.remove(j_star)
            else:
                # Compute step length and update x and mu
                t1 = np.inf if len(W) == 0 else min(-mu[list(W)] / v.flatten()[v.flatten() < 0], default=np.inf)
                t2 = min(t1, -(A[r, :] @ x - b[r]) / (A[r, :] @ p))
                x += (t2 * p).flatten()
                sol_path = np.vstack([sol_path, x])
                if W:
                    mu[list(W)] += t2 * v[np.array(W) - min(W)]
                mu[r] += t2
                if t2 == t1:
                    W.remove(np.argmin(-mu[list(W)] / v.flatten()[v.flatten() < 0]))
                else:
                    W.add(r)
    
    return x, mu, sol_path, W
