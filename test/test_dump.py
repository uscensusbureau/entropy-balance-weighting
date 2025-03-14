import uuid
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import entropy_balance_weighting as ebw
import entropy_balance_weighting.shared as shared


def test_problem_dump() -> None:
    """Test that the problem data is saved when intended."""
    for i in range(100):
        np.random.seed(125122 + i)
        x = np.random.uniform(size=(300, 10 + i))
        if np.random.random() < 0.5:
            x = sp.csc_array(x)
        mean_population_moments = np.mean(np.random.uniform(size=(300, 10 + i)), 0)
        weights0 = np.ones(300)
        probname = str(uuid.uuid4()) + "_problem.zip"
        failname = str(uuid.uuid4()) + "_fail.zip"
        res = ebw.entropy_balance(
            mean_population_moments=mean_population_moments,
            x_sample=x,
            weights0=weights0,
            options={"save_problem_data": probname, "save_failure_data": failname},
        )
        prob_exists = Path(probname).is_file()
        fail_exists = Path(failname).is_file()
        Path(probname).unlink(missing_ok=True)
        Path(failname).unlink(missing_ok=True)
        assert prob_exists
        if res.converged:
            assert not fail_exists
        else:
            assert fail_exists


def test_problem_dump_and_load() -> None:
    """Test that the problem data can be reused."""
    for i in range(100):
        np.random.seed(1122 + i)
        x = np.random.uniform(size=(300, 10 + i))
        if np.random.random() < 0.5:
            x = sp.csc_array(x)
        mean_population_moments = np.mean(np.random.uniform(size=(300, 10 + i)), 0)
        weights0 = np.ones(300)
        probname = str(uuid.uuid4()) + "_problem.zip"
        res = ebw.entropy_balance(
            mean_population_moments=mean_population_moments,
            x_sample=x,
            weights0=weights0,
            options={"save_problem_data": probname},
        )

        new_moments, new_x, new_w0 = shared.load_problem_from_zip(probname)
        Path(probname).unlink(missing_ok=True)

        res2 = ebw.entropy_balance(
            mean_population_moments=new_moments, x_sample=new_x, weights0=new_w0
        )
        assert res.converged == res2.converged
        assert np.allclose(res.new_weights, res2.new_weights)
        if (res.failure_weights is not None) and (res2.failure_weights is not None):
            assert np.allclose(res.failure_weights, res2.failure_weights)
