import numpy as np
import polars as pl
import pypardiso  # noqa: F401
from formulaic import model_matrix

import entropy_balance_weighting as ebw


def test_colinear_dense() -> None:
    """Test colinearity removal."""
    for i in range(5):
        np.random.seed(100 + i)
        weights0 = np.ones(100)
        x = np.random.random((100, 2))
        x[:, 1] = -0.3 * x[:, 0]
        moments = np.random.random((1000, 2))
        moments[:, 1] = -0.3 * moments[:, 0]
        moments = moments.mean(0)
        res = ebw.entropy_balance(
            x_sample=x, weights0=weights0, mean_population_moments=moments
        )
        assert res.converged
        assert np.allclose(np.dot(x.T, res.new_weights), moments * np.sum(weights0))


def test_colinear_sparse_easy() -> None:
    """Test colinearity removal with sparse matrices."""
    for i in range(5):
        np.random.seed(100 + i)
        n = 100000
        weights0 = 1 + (np.random.random(n) - 0.5) * 0.1
        x = np.random.randint(2, size=(n, 4))
        df = pl.from_numpy(x)
        X = model_matrix(
            "C(column_0) + C(column_1)",
            df.to_arrow(),
            output="sparse",
            ensure_full_rank=False,
        )

        x = np.random.randint(2, size=(n, 2))
        df = pl.from_numpy(x)
        X_moments = model_matrix(
            X.model_spec, df.to_arrow(), output="sparse", ensure_full_rank=False
        )
        moments = np.array(X_moments.mean(0)).ravel()

        res = ebw.entropy_balance(
            x_sample=X, weights0=weights0, mean_population_moments=moments
        )
        assert res.converged
        assert np.allclose(
            (X.T.dot(res.new_weights)).ravel(), moments * np.sum(weights0)
        )


def test_colinear_sparse() -> None:
    """Test colinearity removal with sparse matrices."""
    for i in range(5):
        np.random.seed(111 + i)
        n = 10000
        weights0 = 1 + (np.random.random(n) - 0.5) * 0.1
        x = np.random.randint(3, size=(n, 4))
        x = np.c_[x, np.repeat(np.arange(n // 100), 100)]
        df = pl.from_numpy(x)
        X = model_matrix(
            "(C(column_0) + C(column_1) + C(column_2) + C(column_3)) : C(column_4)",
            df.to_arrow(),
            output="sparse",
            ensure_full_rank=False,
        )

        x = np.random.randint(3, size=(n, 4))
        x = np.c_[x, np.repeat(np.arange(n // 100), 100)]
        df_moments = pl.from_numpy(x)
        X_moments = model_matrix(
            X.model_spec, df_moments.to_arrow(), output="sparse", ensure_full_rank=False
        )
        moments = np.array(X_moments.mean(0)).ravel()

        res = ebw.entropy_balance(
            x_sample=X, weights0=weights0, mean_population_moments=moments
        )
        assert res.converged
        assert np.allclose(
            (X.T.dot(res.new_weights)).ravel(), moments * np.sum(weights0)
        )
