from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from autograd import grad, jacobian

def f_grad_hess(f: Callable[[NDArray[np.float64]], np.float64], 
                x: NDArray[np.float64]) -> Tuple[Callable[[NDArray[np.float64]], np.float64], 
                                                 NDArray[np.float64], NDArray[np.float64]]:
    """ Compute the gradient and Hessian of a multivariate function f(x) at x using automatic differentiation.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Function to compute gradient and Hessian of.
        x (NDArray[np.float64]): Point at which to compute gradient and Hessian.

    Return:
        Tuple[Callable[[NDArray[np.float64]], np.float64], NDArray[np.float64], NDArray[np.float64]]: the function f, Gradient and Hessian of f at x.
    """
    # Gradient function
    grad_func = grad(f) # type: ignore

    # Hessian function (Jacobian of the gradient)
    hess_func = jacobian(grad_func) # type: ignore

    # Compute gradient and Hessian at x
    gradient = grad_func(x)
    hessian = hess_func(x)

    return f, gradient, hessian


def central_finite_diff_grad(f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], eps: float = 1e-8) -> float:
    """Compute the gradient of a multivariate function f(x) at x using central finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Function to compute gradient of.
        x (NDArray[np.float64]): Point at which to compute gradient.
        eps (float): Used for step size. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Gradient of f at x.
    """
    ones = np.ones(x.shape)
    tmp_x = np.concatenate((np.abs(x), ones), axis=0).reshape(-1, x.shape[0])
    h = np.sqrt(eps*np.max(tmp_x, axis=0))
    pxh = x + h[:, None] * np.eye(len(x))
    mxh = x - h[:, None] * np.eye(len(x))
    return (np.apply_along_axis(f, 1, pxh) - np.apply_along_axis(f, 1, mxh)) / (2*h)

def forward_finite_diff_grad(f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Compute the gradient of a multivariate function f(x) at x using forward finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Function to compute gradient of.
        x (NDArray[np.float64]): Point at which to compute gradient.
        eps (float): Used for step size. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Gradient of f at x.
    """
    ones = np.ones(x.shape)
    tmp_x = np.concatenate((np.abs(x), ones), axis=0).reshape(-1, x.shape[0])
    h = np.sqrt(eps*np.max(tmp_x, axis=0))
    pxh = x + h[:, None] * np.eye(len(x))
    return (np.apply_along_axis(f, 1, pxh) - f(x)) / h

def backward_finite_diff_grad(f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], eps: float = 1e-8) -> float:
    """Compute the gradient of a multivariate function f(x) at x using backward finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Function to compute gradient of.
        x (NDArray[np.float64]): Point at which to compute gradient.
        eps (float): Used for step size. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Gradient of f at x.
    """
    ones = np.ones(x.shape)
    tmp_x = np.concatenate((np.abs(x), ones), axis=0).reshape(-1, x.shape[0])
    h = np.sqrt(eps*np.max(tmp_x, axis=0))
    mxh = x - h[:, None] * np.eye(len(x))
    return (f(x) - np.apply_along_axis(f, 1, mxh)) / h

def finite_diff_hess(f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], eps: float = 1e-8) -> NDArray[np.float64]:
    """Compute the hessian of a multivariate function f(x) at x using finite differences.

    Args:
        f (Callable[[NDArray[np.float64]], np.float64]): Function to compute hessian of.
        x (NDArray[np.float64]): Point at which to compute hessian.
        eps (float): Used for step size. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Hessian of f at x.
    """
    n = len(x)
    hess = np.zeros((n, n))
    tmp_x = np.append(np.abs(x), 1)
    h = np.sqrt(eps*np.max(tmp_x, axis=0))

    for i in range(n):
        for j in range(n):
            x_ijp = np.array(x)
            x_ijp[i] += h
            x_ijp[j] += h

            x_ijm = np.array(x)
            x_ijm[i] += h
            x_ijm[j] -= h

            x_jim = np.array(x)
            x_jim[i] -= h
            x_jim[j] += h

            x_jimj = np.array(x)
            x_jimj[i] -= h
            x_jimj[j] -= h

            hess[i, j] = (f(x_ijp) - f(x_ijm) - f(x_jim) + f(x_jimj)) / (4 * h**2)

    return hess
