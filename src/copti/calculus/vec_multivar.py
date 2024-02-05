from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
import sympy as sp
from sympy import hessian
from scipy.optimize import approx_fprime

def f_jac(f: sp.Matrix, symbols: Tuple[sp.Symbol], x0: NDArray[np.float64]) -> Tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], NDArray[np.float64]]:
    """ Compute the Jacobian of a multivariate vector function f(x) at x.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Vector function to compute Jacobian of.
        x (NDArray[np.float64]): Point at which to compute Jacobian.

    Return:
        Tuple[Callable[[NDArray[np.float64]], np.float64], NDArray[np.float64]]: the function f, Jacobian of f at x.
    """
    sym_jac = f.jacobian(symbols)

    # Convert the Sympy matrix f to a NumPy function
    f_numpy_func = sp.lambdify(symbols, f, 'numpy')
    def f_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f_numpy_func(*x)

    # Convert the Sympy matrix jac to a NumPy array
    jac_numpy = sp.lambdify(symbols, sym_jac, 'numpy')

    return f_func, jac_numpy(*x0)

def f_hess(f: sp.Matrix, symbols: Tuple[sp.Symbol], x0: NDArray[np.float64]) -> Tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], NDArray[np.float64]]:
    """ Compute the Hessian of a multivariate vector function f(x) at x.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Vector function to compute Hessian of.
        x (NDArray[np.float64]): Point at which to compute Hessian.

    Return:
        Tuple[Callable[[NDArray[np.float64]], np.float64], NDArray[np.float64]]: the function f, Hessian of f at x.
    """

    #Convert the Sympy matrix f to a NumPy function
    f_numpy_func = sp.lambdify(symbols, f, 'numpy')
    def f_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f_numpy_func(*x)

    # Convert the Sympy matrix jac to a NumPy array
    sym_hess = [hessian(fi, symbols) for fi in f]
    hess_numpy = [sp.lambdify(symbols, hess, 'numpy') for hess in sym_hess]

    def hess_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([hess(*x) for hess in hess_numpy])

    return f_func, hess_func(x0)

def forward_finite_jac(f: Callable[[NDArray[np.float64]], NDArray[np.float64]], 
                       x: NDArray[np.float64], eps: float) -> Tuple[Callable[[NDArray[np.float64]], 
                                                                             NDArray[np.float64]], NDArray[np.float64]]:
    """ Compute the Jacobian of a vector function using forward finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], NDArray[np.float64]]): Vector function to compute Jacobian of.
        x (NDArray[np.float64]): Point at which to compute Jacobian.
        h (float): Step size.

    Return:
        NDArray[np.float64]: Jacobian of f at x.
    """
    #apply approx_fprime to each row of f
    jac = np.array([approx_fprime(x, lambda x: f(x)[i], eps) for i in range(len(f(x)))])
    return f, jac

def forward_finite_hess(f: Callable[[NDArray[np.float64]], NDArray[np.float64]], x: NDArray[np.float64], 
                            eps: float) -> Tuple[Callable[[NDArray[np.float64]], 
                                                          NDArray[np.float64]], NDArray[np.float64]]:
    """ Compute the Hessian of a vector function using forward finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], NDArray[np.float64]]): Vector function to compute Hessian of.
        x (NDArray[np.float64]): Point at which to compute Hessian.
        eps (float): Step size.

    Return:
        NDArray[np.float64]: Hessian of f at x.
    """
    # Compute the Hessian using forward finite differences for loop implementation

    # Get the dimension of the output of f
    output_dim = len(f(x))

    # Initialize an empty array for the Hessian
    hess = np.empty((output_dim, len(x), len(x)))

    # Compute the Hessian for each component of f
    for k in range(output_dim):
        A = np.empty((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                # Compute the second derivative using the central difference formula
                h_i = np.zeros(len(x))
                h_i[i] = np.sqrt(eps)*(1 + np.abs(x[i]))
                h_j = np.zeros(len(x))
                h_j[j] = np.sqrt(eps)*(1 + np.abs(x[j]))
                A[i,j] = (f(x + h_i + h_j)[k] - f(x + h_i)[k] - f(x + h_j)[k] + f(x)[k]) / (h_i[i] * h_j[j])
        hess[k,:,:] = (A + A.T) / 2

    return f, hess