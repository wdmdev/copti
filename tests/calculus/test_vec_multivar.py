import numpy as np
from numpy.typing import NDArray
import sympy as sp
from copti.calculus.vec_multivar import f_jac, f_hess, forward_finite_jac, forward_finite_hess


def test_f_jac():
    """Test that f_jac returns the correct result."""
    # Define a simple vector function
    x1, x2 = sp.symbols('x1 x2')
    f = sp.Matrix([x1**2, x2**2])

    def f_numpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[x[0]**2], 
                        [x[1]**2]])

    def jac_f_numpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[2*x[0], 0], [0, 2*x[1]]])
    
    # Define the point at which to evaluate f and jac_f
    x0 = np.array([1., 2.])
    
    # Compute jac_f using sym_f_jac
    f_, jac = f_jac(f, (x1, x2), x0) #type: ignore
    
    # Check that the result is correct
    np.testing.assert_almost_equal(f_(x0), f_numpy(x0), decimal=2)
    np.testing.assert_almost_equal(jac, jac_f_numpy(x0), decimal=2)

def test_f_hess():
    """Test that f_hess returns the correct result."""
    # Define a simple vector function
    x1, x2 = sp.symbols('x1 x2')
    f = sp.Matrix([x1**2, x2**2])

    def f_numpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[x[0]**2], 
                        [x[1]**2]])

    def hess_f_numpy() -> NDArray[np.float64]:
        return np.array([[[2., 0.], [0., 0.]], 
                         [[0., 0.], [0., 2.]]])
    
    # Define the point at which to evaluate f and hess_f
    x0 = np.array([1., 2.])
    
    # Compute hess_f using sym_f_hess
    f_, hess = f_hess(f, (x1, x2), x0) #type: ignore
    
    # Check that the result is correct
    np.testing.assert_almost_equal(f_(x0), f_numpy(x0), decimal=2)
    np.testing.assert_almost_equal(hess, hess_f_numpy(), decimal=2)

def test_forward_finite_jac():
    """Test that forward_finite_jac returns the correct result."""
    # Define a simple vector function
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[x[0]**2], 
                        [x[1]**2]])

    def jac_f_numpy(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[2*x[0], 0], [0, 2*x[1]]])
    
    # Define the point at which to evaluate f and jac_f
    x0 = np.array([1., 2.])
    
    # Compute jac_f using sym_f_jac
    f_, jac = forward_finite_jac(f, x0, eps=1e-8) #type: ignore
    
    # Check that the result is correct
    np.testing.assert_almost_equal(f_(x0), f(x0), decimal=2)
    np.testing.assert_almost_equal(jac, jac_f_numpy(x0), decimal=2)

def test_forward_finite_hess():
    """Test that forward_finite_hess returns the correct result."""
    # Define a simple vector function
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[x[0]**2], 
                        [x[1]**2]])

    def hess_f_numpy() -> NDArray[np.float64]:
        return np.array([[[2., 0.], [0., 0.]], 
                         [[0., 0.], [0., 2.]]])
    
    # Define the point at which to evaluate f and hess_f
    x0 = np.array([1., 2.])
    
    # Compute hess_f using sym_f_hess
    f_, hess = forward_finite_hess(f, x0, eps=1e-8) #type: ignore
    
    # Check that the result is correct
    np.testing.assert_almost_equal(f_(x0), f(x0), decimal=2)
    np.testing.assert_almost_equal(hess, hess_f_numpy(), decimal=2)


if __name__ == "__main__":
    test_f_jac()
    test_f_hess()
    test_forward_finite_jac()
    test_forward_finite_hess()