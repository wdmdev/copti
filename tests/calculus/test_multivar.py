from copti.calculus import multivar

import numpy as np
from numpy.typing import NDArray

def test_f_grad_hess_returns_correct_result():
    """Test that f_grad_hess returns the correct result."""
    # Define a simple function
    def f(x: NDArray[np.float64]) -> np.float64:
        return x[0]**2 + x[1]**2

    # Define the gradient of f
    def grad_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2*x[0], 2*x[1]])

    # Define the hessian of f
    def hess_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[2., 0.], [0., 2.]])

    # Define the point at which to evaluate f, grad_f, and hess_f
    x = np.array([1., 2.])

    # Compute f, grad_f, and hess_f using f_grad_hess
    f_, grad_f_, hess_f_ = multivar.f_grad_hess(f, x)

    # Check that the results are correct
    assert f_(x) == f(x)
    assert np.all(grad_f_ == grad_f(x))
    assert np.all(hess_f_ == hess_f(x))

def test_central_finite_diff_grad_returns_correct_result():
    """Test that central_finite_diff_grad returns the correct result."""
    # Define a simple function
    def f(x: NDArray[np.float64]) -> float:
        return x[0]**2 + x[1]**2

    # Define the gradient of f
    def grad_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2*x[0], 2*x[1]])

    # Define the point at which to evaluate f and grad_f
    x = np.array([1., 2.])

    # Compute grad_f using central_finite_diff_grad
    grad_f_ = multivar.central_finite_diff_grad(f, x, eps=1e-8)

    # Check that the result is correct
    np.testing.assert_almost_equal(grad_f_, grad_f(x), decimal=2)

def test_forward_finite_diff_grad_returns_correct_result():
    """Test that forward_finite_diff_grad returns the correct result."""
    # Define a simple function
    def f(x: NDArray[np.float64]) -> float:
        return x[0]**2 + x[1]**2

    # Define the gradient of f
    def grad_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2*x[0], 2*x[1]])

    # Define the point at which to evaluate f and grad_f
    x = np.array([1., 2.])

    # Compute grad_f using forward_finite_diff_grad
    grad_f_ = multivar.forward_finite_diff_grad(f, x, eps=1e-8)

    # Check that the result is correct
    np.testing.assert_almost_equal(grad_f_, grad_f(x), decimal=2)

def test_backward_finite_diff_grad_returns_correct_result():
    """Test that backward_finite_diff_grad returns the correct result."""
    # Define a simple function
    def f(x: NDArray[np.float64]) -> float:
        return x[0]**2 + x[1]**2

    # Define the gradient of f
    def grad_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2*x[0], 2*x[1]])

    # Define the point at which to evaluate f and grad_f
    x = np.array([1., 2.])

    # Compute grad_f using backward_finite_diff_grad
    grad_f_ = multivar.backward_finite_diff_grad(f, x, eps=1e-8)

    # Check that the result is correct
    np.testing.assert_almost_equal(grad_f_, grad_f(x), decimal=2)


def test_finite_diff_hess_returns_correct_result():
    """Test that finite_diff_hess returns the correct result."""
    # Define a simple function
    def f(x: NDArray[np.float64]) -> float:
        return 3*x[0]**2 + 2*x[1]**2

    # Define the hessian of f
    def hess_f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[6., 0.], [0., 4.]])

    # Define the point at which to evaluate f and hess_f
    x = np.array([1., 2.])

    # Compute hess_f using finite_diff_hess
    hess_f_ = multivar.finite_diff_hess(f, x, eps=1e-8)

    # Check that the result is correct
    np.testing.assert_almost_equal(hess_f_, hess_f(x), decimal=2)

if __name__ == "__main__":
    test_f_grad_hess_returns_correct_result()
    test_forward_finite_diff_grad_returns_correct_result()
    test_backward_finite_diff_grad_returns_correct_result()
    test_central_finite_diff_grad_returns_correct_result()
    test_finite_diff_hess_returns_correct_result()