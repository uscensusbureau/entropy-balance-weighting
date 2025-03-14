import numpy as np
import scipy.sparse as sp

import entropy_balance_weighting as wgt


def test_simple() -> None:
    """Test tiny form solvable by hand."""
    x_sample = sp.csr_array(np.array([[1.0], [2.0]]))
    weights0 = np.ones(2)
    mean_population_moments = np.array([1.5])
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=x_sample,
        weights0=weights0,
    )
    new_weights = res.new_weights / np.sum(res.new_weights)
    assert np.allclose(new_weights, np.ones(2) / 2)


def test_simple2() -> None:
    """Test another tiny form solvable by hand."""
    x = sp.csr_array(np.array([[-1.0], [2.0]]))
    q0 = np.ones(2) / 2.0
    mean_population_moments = np.array([0.0])
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments, x_sample=x, weights0=q0
    )
    new_weights = res.new_weights / np.sum(res.new_weights)
    assert np.allclose(new_weights, np.array([2 / 3, 1 / 3]))


def test_complex_dense_example() -> None:
    """Test a fuller dense matrix."""
    np.random.seed(12522)
    x = sp.csr_array(np.random.uniform(size=(10000, 40)))
    mean_population_moments = np.mean(np.random.uniform(size=(300, 40)), 0)
    weights0 = np.ones(10000)
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments, x_sample=x, weights0=weights0
    )
    assert np.allclose(
        x.T.dot(res.new_weights), mean_population_moments * np.sum(weights0)
    )
