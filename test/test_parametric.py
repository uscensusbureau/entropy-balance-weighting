import numpy as np
import scipy.optimize
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from entropy_balance_weighting import entropy_balance
from entropy_balance_weighting.typing import FArr, NDArray


@settings(max_examples=1000, deadline=None)
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
        )
        assert res
    except ValueError:
        assert True


num_obs = st.shared(st.integers(10, 20))
num_moments = st.shared(st.integers(2, 5))


@given(
    base_x=arrays(
        dtype=np.float64,
        elements=st.floats(0.90, 1.1),
        shape=st.tuples(num_obs, num_moments),
    ),
    valid_elements=arrays(dtype=np.bool_, shape=st.tuples(num_obs, num_moments)),
    weights0=arrays(dtype=np.float64, elements=st.floats(0.8, 1.2), shape=num_obs),
    num_obs=num_obs,
)
@settings(max_examples=1000, deadline=None)
def test_interface_positive_float_inputs(
    base_x: FArr, valid_elements: NDArray[np.bool_], weights0: FArr, num_obs: int
) -> None:
    """
    Check that feasibility = convergence in a small example.

    When the LM are unbounded near the solution (e.g. r is 0 at the only
    optimum) the dual criterion is easier to set up convergence criteria
    for, so I test both approaches to cover those edge cases.
    """
    x = base_x * valid_elements
    assume(np.sum(valid_elements) > (0.2 * x.size))
    mean_population_moments = np.sum(
        x[-num_obs // 2 :, :]
        * weights0[-num_obs // 2 :, None]
        / np.sum(weights0[-num_obs // 2 :]),
        0,
    )
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
    )
    if res.equality_multipliers_estimate is None:
        res.equality_multipliers_estimate = np.zeros(1)
    assert (
        (res_linprog.success and res.converged)
        or (not res_linprog.success and not res.converged)
    ) or (np.linalg.norm(res.equality_multipliers_estimate) > 10.0)
