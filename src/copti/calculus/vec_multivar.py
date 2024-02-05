from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
import sympy as sp
from sympy import hessian

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
