import numpy as np
import scipy.sparse as sp
from scipy.optimize import approx_fprime

import entropy_balance_weighting as wgt
from entropy_balance_weighting import shared
from entropy_balance_weighting.typing import FArr, SparseArr


def test_penalty_growth() -> None:
    """Test that moments are better fit by higher penalty params."""
    np.random.seed(12525)
    x_sample = np.random.random((1000, 3))
    weights0 = np.ones(1000)
    mean_population_moments = x_sample[-100:, :].mean(0)
    last_ce_norm = np.inf
    for penalty_parameter in [0.01, 0.05, 0.2, 0.6, 1.0, 5.0]:
        res = wgt.entropy_balance_penalty(
            mean_population_moments=mean_population_moments,
            x_sample=sp.csr_array(x_sample),
            weights0=weights0,
            penalty_parameter=penalty_parameter,
        )

        A_mat = sp.diags_array(weights0) @ x_sample
        agg_population_moments = mean_population_moments * np.sum(weights0)
        ce_norm = float(
            np.linalg.norm(A_mat.T @ res.new_weights - agg_population_moments)
        )
        assert ce_norm < last_ce_norm
        last_ce_norm = ce_norm


def test_penalty_components() -> None:
    """
    Test matching one dimension of moments.

    Increasing just one dimension of penalty
    decrases that dimension's miss, and decreases
    it more than it decreases the other dimensions'
    misses.
    """
    np.random.seed(12525)
    x_sample = np.random.random((1000, 3))
    weights0 = np.ones(1000)
    mean_population_moments = x_sample[-100:, :].mean(0)
    penalty_parameter = np.ones(3)

    penalty_parameter[0] = 0.01
    res = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=sp.csr_array(x_sample),
        weights0=weights0,
        penalty_parameter=penalty_parameter,
    )
    last_cv = res.constraint_violations
    for pp in [0.05, 0.2, 0.6, 1.0, 5.0, 10.0]:
        penalty_parameter[0] = pp
        res = wgt.entropy_balance_penalty(
            mean_population_moments=mean_population_moments,
            x_sample=sp.csr_array(x_sample),
            weights0=weights0,
            penalty_parameter=penalty_parameter,
        )
        cv = res.constraint_violations
        assert cv[0] < last_cv[0]
        assert (cv[0] / last_cv[0]) <= (cv[1] / last_cv[1])
        last_cv = cv


def test_penalty_vectorparam() -> None:
    """Test that moments are better fit by higher penalty params."""
    np.random.seed(12525)
    x_sample = np.random.random((1000, 3))
    weights0 = np.ones(1000)
    mean_population_moments = x_sample[-100:, :].mean(0)

    penalty_parameter = np.ones(3)
    res_vec = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=sp.csr_array(x_sample),
        weights0=weights0,
        penalty_parameter=penalty_parameter,
    )

    penalty_parameter_scalar = 1.0
    res_scalar = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=sp.csr_array(x_sample),
        weights0=weights0,
        penalty_parameter=penalty_parameter_scalar,
    )
    assert np.allclose(res_vec.new_weights, res_scalar.new_weights)


def test_quadratic_penalty_functional_forms() -> None:
    """Test tiny form solvable by hand."""
    x_sample = sp.csr_array(np.array([[1.0], [2.0]]))
    weights0 = np.ones(2)
    mean_population_moments = np.array([1.6])
    penalty_parameter = sp.diags_array(np.ones(1), format="csr")

    A_mat = sp.diags_array(weights0) @ x_sample
    agg_population_moments = mean_population_moments * np.sum(weights0)

    def new_criterion(ratio: FArr) -> FArr:
        out: FArr = np.sum(
            weights0 * (ratio * np.log(ratio) - ratio + 1)
        ) + 1 / 2 * np.sum(
            penalty_parameter.diagonal()
            * (A_mat.T.dot(ratio) - agg_population_moments) ** 2
        )
        return out

    def gradient(ratio: FArr) -> FArr:
        out: FArr = weights0 * np.log(ratio) + shared.chain_dot(
            A_mat, penalty_parameter, A_mat.T.dot(ratio) - agg_population_moments
        )
        return out

    def hessian(ratio: FArr) -> SparseArr:
        out: SparseArr = sp.diags_array(weights0 * 1 / ratio) + shared.chain_dot(
            A_mat, penalty_parameter, A_mat.T
        )
        return out

    def woodbury_inverse_hess(ratio: FArr) -> SparseArr:
        w0m1_r = sp.diags_array(ratio * 1 / weights0, format="csr")
        U = shared.chain_dot(A_mat, penalty_parameter)
        V = shared.chain_dot(A_mat.T, w0m1_r)
        i2 = sp.eye_array(2)
        inner_inv = np.array([sp.linalg.inv(sp.eye_array(1) + V @ U)])
        wb: SparseArr = i2 - U @ inner_inv @ V
        wb = w0m1_r @ wb
        return wb

    x0 = np.ones(2)
    wb = woodbury_inverse_hess(x0)
    true_hess = hessian(x0)
    true_inv_hess = np.linalg.inv(true_hess.toarray())

    xk = np.array([0.71, 2.2])
    fprime = approx_fprime(xk, new_criterion)
    true_grad = gradient(xk)

    approx_hess = approx_fprime(xk, gradient)
    true_hess = hessian(xk)

    wb = woodbury_inverse_hess(xk)
    true_inv_hess = np.linalg.inv(true_hess.toarray())

    assert np.allclose(wb, true_inv_hess)
    assert np.all(np.abs(approx_hess - true_hess.toarray()) < 1e-4)
    assert np.all(np.abs(fprime - true_grad) < 1e-4)
