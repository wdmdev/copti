import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from typing import List, Tuple

def find_feasible_point(g: NDArray[np.float64], A: NDArray[np.float64], 
                        b: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Find a feasible point for the inequality constraints A*x <= b.
    This can be done using a simple linear program or any other method suitable for your problem.

    Args:
    ----------
        g : np.ndarray
            The objective function coefficients.
        A : np.ndarray
            The inequality constraint matrix.
        b : np.ndarray
            The inequality constraint vector.
    
    Returns:
    ----------
        np.ndarray
            A feasible point for the inequality constraints.
    """
    res = linprog(g, A_eq=A, b_eq=b, method='highs')
    if res.success:
        return res.x
    else:
        raise ValueError("Feasible point not found")


def primal_active_set(H: NDArray[np.float64], g: NDArray[np.float64], A_ineq: NDArray[np.float64], 
                      b_ineq: NDArray[np.float64], x0: NDArray[np.float64],
                      tol: float = 1e-5) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            The initial point.
        tol : float, optional
            Tolerance for 0 checks in the algorithm. Defaults to 1e-5.
    
    Returns:
    ----------
        A tuple of x and mu. 
        Where x is the optimal solution and mu are the Lagrange multipliers.
    """

    xk = x0
    Wk = _get_working_set(A_ineq, xk, b_ineq)

    while True:
        if Wk:
            Ak = A_ineq[Wk]
            matrix = np.block([[H, -Ak.T],
                               [-Ak, np.zeros((Ak.shape[0], Ak.shape[0]))]])
            vector = np.concatenate([-(H @ xk) - g, np.zeros(Ak.shape[0])])
        else:
            # If the working set is empty, solve the unconstrained problem
            matrix = H
            vector = -(H @ xk) - g

        solution = np.linalg.solve(matrix, vector)
        p_star = solution[:len(xk)] #direction to move
        mu = np.zeros(len(A_ineq))

        if Wk:
            for i, mu_i in zip(Wk, solution[len(xk):]):
                mu[i] = mu_i

        if abs(np.linalg.norm(p_star)) <= tol:
            if not Wk or all(mu_i >= 0 for mu_i in mu): #type: ignore
                # Optimal solution found
                return xk, mu #type: ignore
            else:
                # Remove constraint with most negative mu
                j = np.argmin(mu)
                Wk.pop(j)
        else:
            # Compute distance to nearest inactive constraint in search direction
            alpha = np.inf
            j = None
            for i, (a_i, b_i) in enumerate(zip(A_ineq, b_ineq)):
                if i not in Wk and a_i @ p_star < 0:
                    alpha_i = (b_i - a_i @ xk) / (a_i @ p_star)
                    if alpha_i < alpha:
                        alpha, j = alpha_i, i

            if alpha < 1:
                xk += alpha * p_star
                Wk.append(j)
            else:
                xk += p_star

def _get_working_set(A_ineq: NDArray[np.float64], x: NDArray[np.float64], 
                    b_ineq: NDArray[np.float64], tol:float = 1e-5) -> List[int]:
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
    return [i for i, (a, b) in enumerate(zip(A_ineq, b_ineq)) if abs(a @ x - b) <= tol]
