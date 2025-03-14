import numpy as np
import scipy.optimize
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from entropy_balance_weighting import entropy_balance
from entropy_balance_weighting.typing import FArr, NDArray


@settings(max_examples=300, deadline=None)
@given(
    x=arrays(dtype=np.float64, shape=(5, 2)),
    weights0=arrays(dtype=np.float64, shape=5),
    mean_population_moments=arrays(dtype=np.float64, shape=2),
)
def test_interface_allpossible_float_inputs(
    x: FArr,
    weights0: FArr,
    mean_population_moments: FArr,
) -> None:
    """Test that entropy balance only finishes or ValueErrors."""
    try:
        res = entropy_balance(
            mean_population_moments=mean_population_moments,
            weights0=weights0,
            x_sample=x,
            options={"bounds": (0, None)},
        )
        assert res
    except ValueError:
        assert True


num_obs = st.shared(st.integers(10, 20))
num_moments = st.shared(st.integers(2, 5))


@given(
    base_x=arrays(
        dtype=np.float64,
        elements=st.floats(0.20, 1.9),
        shape=st.tuples(num_obs, num_moments),
    ),
    valid_elements=arrays(dtype=np.bool_, shape=st.tuples(num_obs, num_moments)),
    weights0=arrays(dtype=np.float64, elements=st.floats(0.01, 5.2), shape=num_obs),
    num_obs=num_obs,
)
@settings(max_examples=300, deadline=None)
def test_interface_positive_float_inputs(
    base_x: FArr, valid_elements: NDArray[np.bool_], weights0: FArr, num_obs: int
) -> None:
    """
    Check that the elastic mode always converges in a small example.

    If a linear programming check shows there is no feasible
    solution, we double check that elastic mode doesn't think
    it's found one.
    """
    np.random.seed(182)
    x = base_x * valid_elements
    assume(np.sum(valid_elements) > (0.2 * x.size))
    mean_population_moments = np.sum(
        x[-num_obs // 2 :, :]
        * weights0[-num_obs // 2 :, None]
        / np.sum(weights0[-num_obs // 2 :]),
        0,
    ) * (1.0 + 0.1 * (np.random.random(base_x.shape[1]) - 0.5))
    res_linprog = scipy.optimize.linprog(
        np.zeros(x.shape[0]),
        A_eq=x.T,
        b_eq=np.sum(weights0) * mean_population_moments,
        bounds=(0.00, None),
    )
    res = entropy_balance(
        mean_population_moments=mean_population_moments,
        weights0=weights0,
        x_sample=x,
        options={"bounds": (0, None), "max_steps": 100},
    )
    if res_linprog.status == 2:
        assert np.linalg.norm(res.constraint_violations) > 1e-5
    assert res.converged
