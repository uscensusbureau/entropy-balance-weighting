import tempfile
import zipfile

import numexpr as ne
import numpy as np
import scipy.sparse as sp
import sparse_dot_mkl as sdmkl

from entropy_balance_weighting.typing import AnyArray, FArr, TypeVar, Union

sparse_array = sp.csr_array
sp_fmt_flag = "csr"


class EntropyBalanceResults:
    """A container for the results of the EBW routines."""

    def __init__(
        self,
        new_weights: FArr,
        converged: bool,
        n_iterations: int,
        constraint_violations: FArr,
        error_message: str = "",
        failure_weights: Union[None, FArr] = None,
        equality_multipliers_estimate: Union[None, FArr] = None,
        moment_slack_multipliers_estimate: Union[None, FArr] = None,
        eta: Union[None, float] = None,
    ) -> None:
        self.new_weights = new_weights
        self.converged = converged
        self.n_iterations = n_iterations
        self.constraint_violations = constraint_violations
        self.error_message = error_message
        self.failure_weights = failure_weights
        self.equality_multipliers_estimate = equality_multipliers_estimate
        self.moment_slack_multipliers_estimate = moment_slack_multipliers_estimate
        self.eta = eta


def criterion(
    g: FArr,
    weights0: FArr,
) -> tuple[float, FArr, FArr]:
    """Calculate the criterion, jacobian, and (diagonal of the) Hessian."""
    if np.any(g <= 0.0):
        return (np.inf, np.array([np.inf]), np.array([np.inf]))
    lg = np.log(g)  # noqa: F841
    out = (
        float(ne.evaluate("sum(weights0 * (g * lg - g + 1))")),
        ne.evaluate("weights0 * lg"),
        ne.evaluate("g/weights0"),
    )
    return out


def step_sizes_converged(
    step_tol: float, primal_step: FArr, dual_step: FArr, delta_ck: FArr
) -> bool:
    """Determine whether to continue iterating by step size."""
    primal_step_converged = bool(np.linalg.norm(primal_step) < step_tol)
    ck_change_small = bool(np.linalg.norm(delta_ck) < step_tol)
    return primal_step_converged and ck_change_small


def full_problem_violation(Cd: FArr, Ce: FArr) -> float:
    """
    Calculate the full problem optimality violation.

    ||Jac{Lagrangian} - 0||**2 + ||ConstraintViolation||**2 < eps
    """
    out = float(np.sqrt(np.linalg.norm(Cd) ** 2 + np.linalg.norm(Ce) ** 2))
    return out


def calculate_feasibile_step_lengths(
    point: FArr, candidate_step: FArr, *, tau: float = 1.0
) -> float:
    """
    Determine whether we have to stop a step short for feasibility.

    tau is the fraction-to-the-boundry parameter that keeps you off 0 if desired.
    """
    largest_step_ratio = float(
        np.min(
            -tau * point[candidate_step < 0] / candidate_step[candidate_step < 0],
            initial=np.inf,
        )
    )
    out = min(1.0, largest_step_ratio)
    return out


T = TypeVar("T")


def chain_dot(*args: T) -> T:
    """Evaluate (A @ (B @ (C @ ... (Y @ Z))))."""
    out = args[-1]
    for n in range(2, len(args) + 1):
        out = sdmkl.dot_product_mkl(args[-n], out)
    return out


def inputs_are_invalid(
    x_sample: AnyArray, weights0: FArr, mean_population_moments: FArr
) -> bool:
    """Sanity check inputs for error out."""
    if not sp.issparse(x_sample):
        nans = bool(
            np.any(np.isnan(x_sample))
            or np.any(np.isnan(weights0))
            or np.any(np.isnan(mean_population_moments))
        )
        infs = bool(
            np.any(np.isinf(x_sample))
            or np.any(np.isinf(weights0))
            or np.any(np.isinf(mean_population_moments))
        )
    else:
        nans = bool(
            np.any(np.isnan(x_sample.data))
            or np.any(np.isnan(weights0))
            or np.any(np.isnan(mean_population_moments))
        )
        infs = bool(
            np.any(np.isinf(x_sample.data))
            or np.any(np.isinf(weights0))
            or np.any(np.isinf(mean_population_moments))
        )
    nonpos_weights = bool(np.any(weights0 <= 0))

    return nans or infs or nonpos_weights


def dump_problem_to_zip(
    zip_filename: str, mean_population_moments: FArr, x_sample: AnyArray, weights0: FArr
) -> None:
    """Create a zip file that contains the data necessary to rerun the EBW problem."""
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
            np.save(temp_file.name, mean_population_moments)
            zipf.write(temp_file.name, "moments.npy")
        if sp.issparse(x_sample):
            with tempfile.NamedTemporaryFile(suffix=".npz") as temp_file:
                sp.save_npz(temp_file.name, x_sample)
                zipf.write(temp_file.name, "x.npz")
        else:
            with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
                np.save(temp_file.name, x_sample)
                zipf.write(temp_file.name, "x.npy")
        with tempfile.NamedTemporaryFile(suffix=".npy") as temp_file:
            np.save(temp_file.name, weights0)
            zipf.write(temp_file.name, "w0.npy")


def load_problem_from_zip(zip_filename: str) -> tuple[FArr, AnyArray, FArr]:
    """Load a zip file that contains the data necessary to rerun the EBW problem."""
    with zipfile.ZipFile(zip_filename, "r") as zipf:
        with zipf.open("moments.npy") as moment_file:
            moments = np.load(moment_file)
        if "x.npz" in zipf.namelist():
            with zipf.open("x.npz") as x_file:
                x = sp.load_npz(x_file)
        else:
            with zipf.open("x.npy") as x_file:
                x = np.load(x_file)
        with zipf.open("w0.npy") as w0_file:
            weights0 = np.load(w0_file)
    return moments, x, weights0
