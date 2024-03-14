import numpy as np
from numpy.typing import NDArray
from typing import Set, List, Tuple
from copti.quad.solvers import range_space_solver, LU_solver
from scipy.linalg import lu_factor, lu_solve


def primal_active_set(H: NDArray[np.float64], g: NDArray[np.float64], A: NDArray[np.float64], 
                      b: NDArray[np.float64], x0: NDArray[np.float64],
                      k:int, tol: float = 1e-5) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Primal Active Set Method for Convex Inequality Constrained QPs.

    Args:
    ----------
        H : np.ndarray
            The Hessian matrix.
        g : np.ndarray
            The objective function coefficients.
        A_ineq : np.ndarray
            The inequality constraint matrix.
        b_ineq : np.ndarray
            The inequality constraint vector.
        x0 : np.ndarray
            The initial feasible point.
        k : int
            The maximum number of iterations.
        tol : float, optional
            Tolerance for 0 checks in the algorithm. Defaults to 1e-5.
    
    Returns:
    ----------
        A tuple of x and mu. 
        Where x is the optimal solution and mu are the Lagrange multipliers.
    """

    xk = x0
    Wk = _get_active_set(A, xk, b)

    for _ in range(k):
        # solve for step direction p_star and lagrange multipliers mu
        p_star, mu = _solve_qp(H, g, A, xk, Wk)

        # if p_star is approximately 0
        if abs(np.linalg.norm(p_star)) < tol:
            if np.all([mu_i >= 0 for mu_i in mu]):
                # The optimal solution has been found
                return xk, mu
            else:
                # remove most negative lagrange multiplier from the working set
                j = np.argmin(mu)
                Wk.remove(j)
        else:
            # Compute the distance to the nearest inactive constraint in the search direction
            A_i, b_i, indicies_map = _get_valid_inactive_constraints(A, b, Wk, p_star)
            alpha = min(1, min((b_i - A_i@xk) / (A_i@p_star)))

            # Check for blocking constraints
            if alpha < 1:
                # add blocking constraint to the working set
                j_masked = np.argmin((b_i- A_i@xk) / (A_i@p_star))
                j = indicies_map[j_masked]
                Wk.add(j)

            # Take a step in the direction of p_star
            xk += alpha*p_star
        
    return xk, mu




def _get_active_set(A_ineq: NDArray[np.float64], x: NDArray[np.float64], 
                    b_ineq: NDArray[np.float64], tol:float = 1e-5) -> Set[int]:
    """ Get the working set for the inequality constraints A*x <= b.
    
    Args:
    ----------
        A_ineq : np.ndarray
            The inequality constraint matrix.
        x : np.ndarray
            The current point.
        b_ineq : np.ndarray
            The inequality constraint vector.
        tol : float, optional
            Tolerance for the working set. Defaults to 1e-5.
    
    Returns:
    ----------
        np.ndarray
            The working set for the inequality constraints.
    """
    return {i for i, (a, b) in enumerate(zip(A_ineq, b_ineq)) if abs(a @ x - b) <= tol}

def _solve_qp(H: NDArray[np.float64], g:NDArray[np.float64],
                A: NDArray[np.float64], xk: NDArray[np.float64], Wk: Set[int]) \
                    -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    Wk = list(Wk)
    if Wk:
        solution = range_space_solver(H=H, g=np.dot(H,xk) + g, A=A[Wk], b=np.zeros(len(Wk)))
    else:
        solution = np.linalg.solve(H, np.dot(H, -np.dot(H, xk) - g))

    p_star = solution[:len(xk)] #direction to move
    mu = np.zeros(A.shape[0])

    if Wk:
        for i, mu_i in zip(Wk, solution[len(xk):]):
            mu[i] = mu_i

    return p_star, mu

def _get_valid_inactive_constraints(A: NDArray[np.float64], b: NDArray[np.float64], 
                              Wk: Set[int], p_star:NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.signedinteger]]:
    Wk = list(Wk)
    i_mask = np.ones(A.shape, dtype=bool)
    i_mask[Wk] = False
    original_indices = np.arange(A.shape[0])
    indices_map = original_indices[i_mask[:,0]]  # Mapping indices from Ai back to A
    # Use mask to choose inactive constraints
    A_i = A[i_mask[:,:]].reshape(i_mask[:,0].sum(), -1)
    b_i = b[i_mask[:,0]]
    ap_mask = A_i@p_star < 0

    return A_i[ap_mask], b_i[ap_mask], indices_map[ap_mask]