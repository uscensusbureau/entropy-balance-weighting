import numpy as np
import scipy.sparse as sp
from scipy.optimize import approx_fprime

from entropy_balance_weighting.ebw_penalty import woodbury_times_vector
from entropy_balance_weighting.shared import chain_dot, criterion
from entropy_balance_weighting.typing import FArr


def test_criterion() -> None:
    """Test the criterion function."""
    weights0 = np.ones(100)
    f_val, grad, inverse_hess_diag = criterion(np.ones(100), weights0=weights0)
    assert np.isclose(f_val, 0.0)
    assert np.allclose(grad, 0.0)
    assert np.allclose(inverse_hess_diag, 1.0)

    def f_val_only(x: FArr) -> float:
        return criterion(x, weights0=weights0)[0]

    ngrad = approx_fprime(np.ones(100), f_val_only)
    assert np.allclose(grad, ngrad)


def test_criterion_hess() -> None:
    """Test the Hessian calculation."""
    weights0 = np.ones(100)
    for j in range(100):
        np.random.seed(155)
        x = np.random.random(size=100) + 0.05
        f_val, grad0, inverse_hess_diag = criterion(x, weights0=weights0)
        x[j] += 1e-5
        _, grad1, _ = criterion(x, weights0=weights0)
        assert np.allclose(
            (grad1 - grad0) / 1e-5,
            np.diag(1 / inverse_hess_diag)[j, :],
            atol=1e-3,
            rtol=1e-3,
        )


def test_woodbury_times_vector() -> None:
    """Test the Woodbury formula usage."""
    np.random.seed(444)
    Atmp = np.random.random((20, 5))
    A = Atmp.T @ Atmp
    Am1: sp.csr_array = sp.csr_array(np.linalg.inv(A))

    U = sp.csr_array(np.random.random((5, 2)))
    V = sp.csr_array(np.random.random((2, 5)))

    x = np.random.random(5)
    inverse_term = np.linalg.inv(A + U.toarray() @ V.toarray())
    full = inverse_term.dot(x)
    wb = woodbury_times_vector(Am1, U, V, x)
    assert np.allclose(full, wb)


def test_sparse_chain_dot() -> None:
    """Test the chained dot product."""
    np.random.seed(1252)
    x, y, z = (
        sp.random_array((50, 50), density=0.5, format="csr"),
        sp.random_array((50, 50), density=0.5, format="csr"),
        sp.random_array((50, 50), density=0.5, format="csr"),
    )
    res = chain_dot(x, y, z).toarray()
    true = (x @ y @ z).toarray()
    assert np.allclose(res, true)


def test_dense_chain_dot() -> None:
    """Test the chained dot product."""
    np.random.seed(1252)
    x, y, z = (
        np.random.random(size=(50, 50)),
        np.random.random(size=(50, 50)),
        np.random.random(size=(50, 50)),
    )
    res = chain_dot(x, y, z)
    true = x @ y @ z
    assert np.allclose(res, true)
