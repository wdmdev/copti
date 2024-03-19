from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def to_standard_form(
    g: NDArray[np.float64], 
    A: Optional[NDArray[np.float64]] = None, 
    b: Optional[NDArray[np.float64]] = None, 
    Al: Optional[NDArray[np.float64]] = None, 
    bl: Optional[NDArray[np.float64]] = None, 
    Au: Optional[NDArray[np.float64]] = None, 
    bu: Optional[NDArray[np.float64]] = None, 
    l: Optional[NDArray[np.float64]] = None, 
    u: Optional[NDArray[np.float64]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Converts a general LP optimization problem into its standard form, with optional constraints.

    Args:
    ----------
    g: NDArray[np.float64] - Objective function coefficients.
    A: Optional[NDArray[np.float64]] - Equality constraint coefficients matrix.
    b: Optional[NDArray[np.float64]] - Equality constraint constants.
    Al: Optional[NDArray[np.float64]] - Lower inequality constraint coefficients matrix.
    bl: Optional[NDArray[np.float64]] - Lower inequality constraint constants.
    Au: Optional[NDArray[np.float64]] - Upper inequality constraint coefficients matrix.
    bu: Optional[NDArray[np.float64]] - Upper inequality constraint constants.
    l: Optional[NDArray[np.float64]] - Lower bounds for variables.
    u: Optional[NDArray[np.float64]] - Upper bounds for variables.

    Returns:
    ---------
    g_bar: NDArray[np.float64] - New objective function coefficients.
    A_bar: NDArray[np.float64] - New constraints coefficients matrix.
    b_bar: NDArray[np.float64] - New constraints constants.
    """
    n_vars = g.shape[0]
    zero_mat = np.zeros((n_vars, n_vars))
    I = np.eye(n_vars)
    neg_I = -np.eye(n_vars)

    # Constructing g_bar
    g_bar = np.concatenate((-g, g, np.zeros(n_vars * 4)))

    # Handling optional parameters
    A = A if A is not None else np.zeros((0, n_vars))
    b = b if b is not None else np.zeros(0)
    Al = Al if Al is not None else np.zeros((0, n_vars))
    bl = bl if bl is not None else np.zeros(0)
    Au = Au if Au is not None else np.zeros((0, n_vars))
    bu = bu if bu is not None else np.zeros(0)
    l = l if l is not None else np.zeros(n_vars)
    u = u if u is not None else np.zeros(n_vars)

    # Constructing A_bar
    A_top = np.concatenate((-A, A, zero_mat, zero_mat, zero_mat, zero_mat), axis=1)
    A_middle_top = np.concatenate((-Al, Al, neg_I, zero_mat, zero_mat, zero_mat), axis=1)
    A_middle_bottom = np.concatenate((-Au, Au, zero_mat, I, zero_mat, zero_mat), axis=1)
    A_bottom_top = np.concatenate((neg_I, I, zero_mat, zero_mat, neg_I, zero_mat), axis=1)
    A_bottom_bottom = np.concatenate((neg_I, I, zero_mat, zero_mat, zero_mat, I), axis=1)
    A_bar = np.vstack((A_top, A_middle_top, A_middle_bottom, A_bottom_top, A_bottom_bottom))

    # Constructing b_bar
    b_bar = np.concatenate((b, bl, bu, l, u))

    return g_bar, A_bar, b_bar
