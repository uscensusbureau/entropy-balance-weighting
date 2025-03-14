import logging
from timeit import default_timer as timer

import numpy as np
import pypardiso
import scipy
import scipy.sparse as sp
import sparse_dot_mkl as sdmkl

import entropy_balance_weighting.shared as shared
from entropy_balance_weighting.shared import sp_fmt_flag, sparse_array
from entropy_balance_weighting.typing import Any, AnyArray, FArr, Optional, Union

logger = logging.getLogger(__name__)


def entropy_balance_penalty(
    mean_population_moments: FArr,
    x_sample: AnyArray,
    weights0: FArr,
    penalty_parameter: Union[float, FArr],
    options: Optional[dict[str, Any]] = None,
) -> shared.EntropyBalanceResults:
    """
    Run Entropy Balancing with a penalty function instead of constraints.

    Solves the problem of reweighting existing data to
    match moments taken from outside of the data set. The
    changes to the weights are the lowest-entropy adjustment
    that can match the new moments, relative to a quadratic
    penalty function on the moments. A higher "penalty_parameter"
    is guaranteed to match the moments at least as well as a lower one.
    As the penalty_parameter goes to infinity, this approaches the
    constrained formulation. (In practice, use the constrained
    formulation of ebw.entropy_balance instead of very large
    penalty_parameter values.)

    With this criterion function, neither colinearity nor
    inconsistent moment constraints causes a problem.

    Implementation uses a simple Woodbury transformation to avoid
    any construction of NxN matrices. Details are in
    in Sanders (2024) "Accelerating Entropy Balance Reweighting".
    Implementation details may change, so future results may not be
    numerically identical to current, nor is future convergence guarenteed.

    Parameters
    ----------
    mean_population_moments : FArr
        The original population moments.

    x_sample : SparseArr
        The #obs x #moments sparse observation-level data.

    weights0 : FArr
        The initial weights.

    penalty_parameter: float | FArr
        The "cost" for each unit of the quadratic miss (A^T r - m)' P (A^T r - m).
        P is a diagonal matrix with penalty_parameter on the main diagonal.

    options : Optional[dict[Any, Any]], optional.
        An optional dictionary of algorithm options to be set, by default None

        max_steps : int, default 30.
            Maximum number of iterations to perform. Default is 100
            for bounded.

        bounds : tuple[float, float], default (0, None).
            Lower and upper ratio bounds.
            Must contain initial guess (or 1.0 if initial guess not set).

        intial_ratio_guess : FArr, default np.ones(n).
            Initial values of weight ratios to guess.
            Included to allow for "warm starts" from previous runs,
            otherwise typically slows convergence.

        optimality_criterion : float, default 1e-5.
            Acceptable deviation of optimality error from 0.

        step_tol : float, default 1e-8.
            Step sizes that are considered zero.


    Returns
    -------
    shared.EntropyBalanceResults
        An object holding the results. The most useful attributes it has are
        "new_weights", a FArr of the new weights, and "converged", a boolean
        that is True if successfully converged and False otherwise.

    Examples
    --------
    Generate some Xs and a set of external moments:
    >>> import entropy_balance_weighting as ebw
    >>> import numpy as np

    >>> np.random.seed(1252)
    >>> n = 10000
    >>> k = 3
    >>> x = np.random.uniform(size=(n, k))

    >>> weights0 = np.ones(n)
    >>> mean_population_moments = np.mean(np.random.uniform(size=(300, k)), 0)

    Observe the aggregates of the sampled X's miss the moments:
    >>> np.dot(x.T, weights0) - mean_population_moments * np.sum(weights0)
    array([ 93.81757113, -24.70450999, -95.58057256])

    Run the EBW-penalty procedure, returning an object containing the results:
    >>> res = ebw.entropy_balance_penalty(
    ...     mean_population_moments=mean_population_moments,
    ...     x_sample=x, weights0=weights0,
    ...     penalty_parameter=1.0
    ... )

    Check the new weights "look" close to the original weights:
    >>> res.new_weights
    array([1.0200403 , 0.99277514, 1.0042485 , ..., 1.0826045 , 0.92997977,
        1.05624043], shape=(10000,))

    Verify the weighted totals are closer to the moments, but not equal:
    >>> np.dot(x.T, res.new_weights) - mean_population_moments * np.sum(weights0)
    array([ 0.12334873, -0.01981342, -0.10367712])

    Re-run with more aggresive penalty:
    >>> res = ebw.entropy_balance_penalty(
    ...     mean_population_moments=mean_population_moments,
    ...     x_sample=x, weights0=weights0,
    ...     penalty_parameter=2.0
    ... )

    And finally, check that high-penalty moments are closer than low-penalty moments:
    >>> np.dot(x.T, res.new_weights) - mean_population_moments * np.sum(weights0)
    array([ 0.061712  , -0.00991234, -0.05186907])

    """
    time_elapsed = timer()

    if shared.inputs_are_invalid(x_sample, weights0, mean_population_moments):
        raise ValueError(
            "Inputs include invalid values (NaNs, non-positive weights, etc)"
        )

    if not options:
        options = {}

    if sp.issparse(x_sample):
        x_sample = sparse_array(x_sample)
    A_mat = sdmkl.dot_product_mkl(
        sp.diags_array(weights0, format=sp_fmt_flag), x_sample
    )
    agg_population_moments = mean_population_moments * np.sum(weights0)

    if options.get("bounds"):
        return entropy_balance_penalty_bounded(
            mean_population_moments=mean_population_moments,
            x_sample=x_sample,
            weights0=weights0,
            penalty_parameter=penalty_parameter,
            options=options,
        )

    n, k = x_sample.shape
    logging.info("Penalty-based Sparse Entropy Balancing: Sanders (2024)")
    logging.info(f"Problem Size: {n} rows, {k} moments")
    logging_header()

    penalty_parameter = check_penalty_parameter(k, penalty_parameter)

    ratio = np.copy(options.get("initial_ratio_guess", np.ones(n)))

    failure_state = True
    status = {
        "n_steps": -1,
        "f_val": 0.0,
        "norm_Ce": np.array(np.inf),
        "norm_Cd": np.array(np.inf),
        "primal_step_size": np.inf,
        "norm_Cs": 0.0,
        "optimality_violation": np.inf,
        "backtrack": 1.0,
    }

    primal_step = np.full(n, np.inf)
    dual_step = np.full(k, np.inf)
    backtrack = 1.0
    while True:
        f_val, _, _ = shared.criterion(ratio, weights0)
        Ce = constraint_gap(A_mat, ratio, agg_population_moments)
        Cd = weights0 * np.log(ratio) + shared.chain_dot(
            A_mat, sp.diags_array(penalty_parameter, format="csc"), Ce
        )
        Am1 = sp.diags_array(1 / weights0 * ratio, format="csc")
        U = sdmkl.dot_product_mkl(
            A_mat, sp.diags_array(penalty_parameter, format="csc")
        )
        V = A_mat.T
        rhs = Cd
        newton_step = woodbury_times_vector(Am1, U, V, -rhs)
        candidate_primal_step = newton_step
        optimality_violation = shared.full_problem_violation(Cd, Ce)
        status = {
            "n_steps": status["n_steps"] + 1,  # type: ignore
            "f_val": f_val,
            "norm_Ce": np.linalg.norm(Ce),
            "norm_Cd": np.linalg.norm(Cd),
            "norm_Cs": 0.0,
            "primal_step_size": np.linalg.norm(primal_step),
            "dual_step_size": np.linalg.norm(dual_step),
            "optimality_violation": optimality_violation,
            "backtrack": backtrack,
        }
        log_status(status)

        backtrack = shared.calculate_feasibile_step_lengths(
            ratio, candidate_primal_step
        )

        ratio += candidate_primal_step

        if np.linalg.norm(Cd) < options.get("optimality_violation", 1e-5) or bool(
            np.linalg.norm(primal_step) < options.get("step_tol", 1e-8)
        ):
            failure_state = False
            break
        elif status["n_steps"] > (max_steps := options.get("max_steps", 30)):
            logging.info(f"Max steps {max_steps} exceeded.")
            failure_state = True
            break

    Ce = constraint_gap(A_mat, ratio, agg_population_moments)
    biggest_miss = np.argmax(np.abs(Ce))
    logging.info(
        f"Largest miss in aggregate moments is index {biggest_miss} with value {Ce[biggest_miss]}."
    )

    logging.info(f"Time elapsed: {timer() - time_elapsed}")
    logging.info(f"Optimization completed, success?: {not failure_state}")

    new_weights = ratio * weights0 if not failure_state else weights0
    results = shared.EntropyBalanceResults(
        new_weights=new_weights,
        failure_weights=ratio * weights0,
        converged=not failure_state,
        n_iterations=status["n_steps"],  # type: ignore
        constraint_violations=A_mat.T.dot(ratio) - agg_population_moments,
    )
    return results


def entropy_balance_penalty_bounded(
    mean_population_moments: FArr,
    x_sample: AnyArray,
    weights0: FArr,
    penalty_parameter: Union[float, FArr],
    options: Optional[dict[str, Any]] = None,
) -> shared.EntropyBalanceResults:
    """Run Penalty-based EBW with bounds."""
    if not options:
        options = {}

    bounds: tuple[float, Optional[float]] = options.get("bounds", (0.0, None))

    time_elapsed = timer()

    n, k = x_sample.shape
    logging.info("Penalty-based Sparse Entropy Balancing: Sanders (2024)")
    logging.info(f"Problem Size: {n} rows, {k} moments")
    logging_header()

    penalty_parameter = check_penalty_parameter(k, penalty_parameter)

    A_mat = sdmkl.dot_product_mkl(sp.diags_array(weights0, format="csr"), x_sample)
    A_ineq = sp.block_array([[sp.eye_array(n), -sp.eye_array(n)]], format="csr")

    if bounds[0] < 0.0:
        bounds = (0.0, bounds[1])
    if bounds[1] is not None:
        bounds = np.concatenate((np.full(n, bounds[0]), -np.full(n, bounds[1])))  # type: ignore
    else:
        bounds = np.full(n, bounds[0])  # type: ignore

    agg_population_moments = mean_population_moments * np.sum(weights0)
    ratio = options.get("initial_ratio_guess", np.ones(n))

    failure_state = True
    status = {
        "n_steps": -1,
        "f_val": 0.0,
        "norm_Ce": np.array(np.inf),
        "norm_Cd": np.array(np.inf),
        "norm_Cs": np.array(np.inf),
        "primal_step_size": np.inf,
        "backtrack": 1.0,
        "optimality_violation": 1.0,
    }

    primal_step = np.full(n, np.inf)
    dual_step = np.full(k, np.inf)
    backtrack_primal = 1.0
    dot = sdmkl.dot_product_mkl
    slacks = dot(A_ineq.T, ratio) - bounds
    mu = 1.0
    lambda_ineq = mu / slacks

    while True:
        f_val, _, _ = shared.criterion(ratio, weights0)
        Ce = constraint_gap(A_mat, ratio, agg_population_moments)
        Cd = (
            weights0 * np.log(ratio)
            + shared.chain_dot(
                A_mat, sp.diags_array(penalty_parameter, format="csc"), Ce
            )
            - dot(A_ineq, lambda_ineq)
        )
        Cs = slacks * lambda_ineq - mu

        if perturbed_KKT_error_function(Cd, Cs) <= mu:
            zeta = np.min(slacks * lambda_ineq) / np.mean(slacks * lambda_ineq)
            sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
            mu = sigma * np.mean(slacks * lambda_ineq)
            Cs = slacks * lambda_ineq - mu

        H = sp.diags_array(1 / ratio * weights0, format="csc")
        p_r = _penalty_bounded_primal_step(
            Cd, lambda_ineq, mu, slacks, H, penalty_parameter, A_mat, A_ineq
        )

        candidate_primal_step = p_r
        implied_dual_step = _recover_penalty_bounded_dual_step(
            p_r, slacks, mu, lambda_ineq, A_ineq
        )
        implied_slacks_step = _recover_penalty_bounded_slacks_step(p_r, A_ineq)

        status = {
            "n_steps": status["n_steps"] + 1,  # type: ignore
            "f_val": f_val,
            "norm_Ce": np.linalg.norm(Ce),
            "norm_Cd": np.linalg.norm(Cd),
            "norm_Cs": np.linalg.norm(Cs),
            "primal_step_size": np.linalg.norm(primal_step),
            "dual_step_size": np.linalg.norm(dual_step),
            "backtrack": backtrack_primal,
            "optimality_violation": mu,
        }
        log_status(status)

        if np.linalg.norm(candidate_primal_step / n) > 1:
            logging.info(
                f"The penalty {penalty_parameter} is numerically unstable, decreasing it 20% to {penalty_parameter/1.2}."
            )
            penalty_parameter /= 1.2
            continue

        backtrack_primal = shared.calculate_feasibile_step_lengths(
            slacks, implied_slacks_step, tau=0.995
        )
        backtrack_dual = shared.calculate_feasibile_step_lengths(
            lambda_ineq, implied_dual_step, tau=0.995
        )

        primal_step = candidate_primal_step * backtrack_primal
        slacks_step = implied_slacks_step * backtrack_primal
        dual_step = implied_dual_step * backtrack_dual
        ratio += primal_step
        slacks += slacks_step
        lambda_ineq += dual_step

        if (np.linalg.norm(Cd) < options.get("optimality_violation", 1e-5)) or bool(
            np.linalg.norm(primal_step) < options.get("step_tol", 1e-8)
        ):
            failure_state = False
            break
        elif status["n_steps"] > (max_steps := options.get("max_steps", 100)):
            logging.info(f"Max steps {max_steps} exceeded.")
            failure_state = True
            break

    Ce = constraint_gap(A_mat, ratio, agg_population_moments)
    biggest_miss = np.argmax(np.abs(Ce))
    logging.info(
        f"Largest miss in aggregate moments is index {biggest_miss} with value {Ce[biggest_miss]}."
    )
    logging.info(f"Largest ratio: {np.max(ratio)}")
    logging.info(f"Smallest ratio: {np.min(ratio)}")

    logging.info(f"Time elapsed: {timer() - time_elapsed}")
    logging.info(f"Optimization completed, success?: {not failure_state}")

    new_weights = ratio * weights0 if not failure_state else weights0
    results = shared.EntropyBalanceResults(
        new_weights=new_weights,
        failure_weights=ratio * weights0,
        converged=not failure_state,
        n_iterations=status["n_steps"],  # type: ignore
        constraint_violations=A_mat.T.dot(ratio) - agg_population_moments,
    )
    return results


def _penalty_bounded_primal_step(
    Cd: FArr,
    lambda_ineq: FArr,
    mu: float,
    slacks: FArr,
    H: AnyArray,
    penalty_parameter: FArr,
    A_mat: AnyArray,
    A_ineq: AnyArray,
) -> FArr:
    dot = sdmkl.dot_product_mkl
    Hb = H + dot(
        dot(A_ineq, sp.diags_array(1 / slacks * lambda_ineq, format="csr")),
        A_ineq.T,
    )
    Am1 = sp.diags_array(1 / Hb.diagonal(), format="csr")
    U = dot(A_mat, sp.diags_array(penalty_parameter, format="csr"))
    V = A_mat.T

    rhs = Cd + dot(A_ineq, lambda_ineq - mu / slacks)
    p_r = woodbury_times_vector(Am1, U, V, -rhs)
    return p_r


def _recover_penalty_bounded_dual_step(
    p_r: FArr, slacks: FArr, mu: float, lambda_ineq: FArr, A_ineq: AnyArray
) -> FArr:
    dot = sdmkl.dot_product_mkl
    out: FArr = dot(
        sp.diags_array(lambda_ineq / slacks, format="csr"),
        (-dot(A_ineq.T, p_r) - (slacks - mu / lambda_ineq)),
    )
    return out


def _recover_penalty_bounded_slacks_step(p_r: FArr, A_ineq: AnyArray) -> FArr:
    out: FArr = sdmkl.dot_product_mkl(A_ineq.T, p_r)
    return out


def woodbury_times_vector(invA: AnyArray, U: AnyArray, V: AnyArray, x: FArr) -> FArr:
    """
    Calculate (A + UV)^{-1} @ x in a memory-efficient way.

    If U is NxK and V is KxN with N>>K, explicitly forming
    the UV product may be impossible, so use the Woodbury
    formula to make sure we never have more than an NxK
    multiplication.
    """
    d = sdmkl.dot_product_mkl
    t1 = shared.chain_dot(V, invA, x)
    if sp.issparse(V) and sp.issparse(invA) and sp.issparse(U):
        t2 = pypardiso.spsolve(
            sp.eye_array(V.shape[0], format="csc") + shared.chain_dot(V, invA, U), t1
        )
    else:
        t2 = scipy.linalg.solve(
            sp.eye_array(V.shape[0]) + shared.chain_dot(V, invA, U), t1
        )
    out: FArr = d(invA, x) - shared.chain_dot(invA, U, np.atleast_1d(t2))
    return out


def logging_header() -> None:
    """Write the column headers to log."""
    logging.info(
        f"{'#':^5}{'Criterion':^14}{'||C_e||':^18}{'||C_d||':^14}{'||C_s||':^14}{'Bcktrck':^9}{'PrimalStepSize':^15}{'DualStepSize':^15}{'||C_e|| + ||C_d||':^14}"
    )


def log_status(s: dict[str, Any]) -> None:
    """Log relevant problem statistics."""
    logging.info(
        f"{s['n_steps']:^5}{s['f_val']:^14.6f}{s['norm_Ce']:^18.6f}{s['norm_Cd']:^14.4f}{s['norm_Cs']:^14.4f}{s['backtrack']:^9.3f}{s['primal_step_size']:^15.8f}{s['dual_step_size']:^15.8f}{s['optimality_violation']:^5.4e}"
    )


def perturbed_KKT_error_function(Cd: FArr, Cs: FArr) -> float:
    """Calculate error in perturbed KKT potential solution."""
    lagrange_foc_err = float(np.linalg.norm(Cd))
    complementary_slackness_err = float(np.linalg.norm(Cs))
    out: float = max(lagrange_foc_err, complementary_slackness_err)
    return out


def constraint_gap(A_mat: AnyArray, ratio: FArr, agg_population_moments: FArr) -> FArr:
    """Calculate how much the constraints differ from 0."""
    out: FArr = sdmkl.dot_product_mkl(A_mat.T, ratio) - agg_population_moments
    return out


def check_penalty_parameter(k: int, penalty_parameter: Union[float, FArr]) -> FArr:
    """Check penalty parameter for invalid inputs."""
    if np.ndim(penalty_parameter) == 0:
        penalty_parameter = np.full(k, penalty_parameter)
    elif np.ndim(penalty_parameter) > 1:
        raise ValueError("Penalty parameter must be float or vector.")
    elif np.shape(penalty_parameter) != (k,):
        raise ValueError("Penalties must be one per moment.")
    if not np.all(penalty_parameter > 0.0):
        raise ValueError("Penalty parameters must be strictly positive.")
    return np.array(penalty_parameter)
