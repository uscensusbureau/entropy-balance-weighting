import logging

import numpy as np
import scipy.sparse as sp

import entropy_balance_weighting as wgt


def complex_dense_example() -> None:
    """Test a fuller dense matrix."""
    np.random.seed(1252)
    n = 100000
    k = 2000
    x = np.random.uniform(size=(n, k))
    x[:, 0:1000] = x[:, 1000:2000]
    x[np.abs(x) < 0.95] = 0.0
    mean_population_moments = np.mean(x[-n // 3 :, :], 0)

    weights0 = np.ones(n)
    res_h = wgt.entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        options={"dual_only": True, "force_dense": False},
    )
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        options={"force_dense": False},
    )
    assert np.allclose(
        x.T.dot(res.new_weights), mean_population_moments * np.sum(weights0)
    )
    assert np.allclose(
        x.T.dot(res_h.new_weights * np.sum(weights0)),
        mean_population_moments * np.sum(weights0),
    )
    assert np.isclose(np.corrcoef(res.new_weights, res_h.new_weights)[0, 1], 1.0)
    logging.info(
        "Both routines satisfy moment conditions and have final weight correlations of 1."
    )


def complex_sparse_example() -> None:
    """Test a fuller sparse matrix."""
    np.random.seed(12522)
    n = 70000
    k = 20
    full = np.random.uniform(size=(n, k))
    full[full < 0.5] = 0.0
    x = sp.csr_array(full)

    moments = np.random.uniform(size=(n, k))
    moments[moments < 0.5] = 0.0
    mean_population_moments = np.mean(moments, 0)

    full[:, -2] = full[:, -1]
    x = sp.csr_array(full)

    weights0 = np.ones(n)
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments, x_sample=x, weights0=weights0
    )
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=x.toarray(),
        weights0=weights0,
    )
    res = wgt.entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        options={"bounds": (0.0, 1.18)},
    )
    print("Differences in final moment gaps:")
    print(x.T.dot(res.new_weights) - mean_population_moments * np.sum(weights0))


def sparse_penalty_example() -> None:
    """Show how the penalty function version works."""
    np.random.seed(12522)
    n = 70000
    k = 20
    full = np.random.uniform(size=(n, k))
    full[full < 0.95] = 0.0
    x = sp.csr_array(full)

    moments = np.random.uniform(size=(n, k))
    moments[moments < 0.95] = 0.0
    mean_population_moments = np.mean(moments, 0)

    weights0 = np.ones(n)

    weights0 = np.ones(n)
    _ = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        penalty_parameter=1e-2,
    )

    weights0 = np.ones(n)
    _ = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        penalty_parameter=1e-1,
    )

    weights0 = np.ones(n)
    _ = wgt.entropy_balance_penalty(
        mean_population_moments=mean_population_moments,
        x_sample=x,
        weights0=weights0,
        penalty_parameter=1e-0,
    )


if __name__ == "__main__":
    complex_dense_example()
    complex_sparse_example()
    sparse_penalty_example()
