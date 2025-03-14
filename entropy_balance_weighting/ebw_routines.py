import logging
from timeit import default_timer as timer

import numexpr as ne
import numpy as np
import pypardiso
import scipy
import scipy.sparse as sp
import sparse_dot_mkl as sdmkl

import entropy_balance_weighting.shared as shared
from entropy_balance_weighting.shared import sp_fmt_flag, sparse_array
from entropy_balance_weighting.typing import Any, AnyArray, FArr, Optional

logger = logging.getLogger(__name__)


def entropy_balance(
    *,
    mean_population_moments: FArr,
    x_sample: AnyArray,
    weights0: FArr,
    options: Optional[dict[Any, Any]] = None,
) -> shared.EntropyBalanceResults:
    """
    Reweight a sample using entropy balancing.

    Solves the problem of reweighting existing data to
    match moments taken from outside of the data set. The
    changes to the weights are the lowest-entropy adjustment
    that can match the new moments.

    The arguments should be scaled so that
    np.mean(weights0[:, None] * x_sample, axis=0)
    would ideally equal mean_population_moments.

    Implementation of minimizing the entropy divergence
    currently follows a mixed dual/primal-dual approach described
    in Sanders (2024) "Accelerating Entropy Balance Reweighting".
    Implementation details may change, so future results may not be
    numerically identical to current, nor is future convergence guarenteed.

    Some features that are of use can be accessed by passing in the
    "options" dictionary: for example, explicit bounds on the ratios
    of new to old weights, setting the initial guess,
    and convergence criterion.

    Parameters
    ----------
    mean_population_moments : FArr
        The target population moments.

    x_sample : scipy.SparseArray | numpy.NDArray
        The #obs x #moments observation-level data.
        Can be either a scipy sparse array (csc_array/csr_array)
        or dense numpy array. Speed/memory tradeoffs will apply
        depending on the structure of the matrix.

    weights0 : FArr
        The initial weights.

    options : Optional[dict[str, Any]], optional.
        An optional dictionary of algorithm options to be set, by default None

        max_steps : int, default 30.
            Maximum number of iterations to perform.

        bounds : tuple[float, float], default (0, None).
            Lower and upper ratio bounds.
            Must contain initial guess (or lb<1.0<ub if initial guess not set).
            With this option not null, the returned weights may only give
            an approximate moments match, and failure to match exactly
            means the initial problem is infeasible. See the paper for details.

        intial_ratio_guess : FArr, default np.ones(n).
            Initial values of weight ratios to guess.
            Included to allow for "warm starts" from previous runs,
            otherwise typically slows convergence.

        optimality_criterion : float, default 1e-5.
            Acceptable deviation of optimality error from 0.

        save_problem_data: str | None, default None.
            If non-null, write the moments, X, and weights0 to a zip file at this
            location for later use.

        save_failure_data: str | None, default None.
            If non-null and convergence fails,
            write the moments, X, and weights0 to a zip file at this location for
            later use.

        step_tol : float, default 1e-8.
            Step sizes that are considered zero.

        eta : float, default 1.0.
            Penalty parameter used in elastic mode. Not intended for
            significant use. If the problem is infeasible this governs
            the tradeoff between satisfying constraints and minimizing the
            criterion. (If the problem is feasible, has no effect.)

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

    Run the EBW procedure, returning an object containing the results:
    >>> res = ebw.entropy_balance(
    ...         mean_population_moments=mean_population_moments,
    ...         x_sample=x, weights0=weights0
    ... )

    Check the new weights "look" close to the original weights:
    >>> res.new_weights
    array([1.02006127, 0.99276564, 1.00425154, ..., 1.08270518, 0.92989575,
        1.05630744], shape=(10000,))

    Verify the weighted totals now match the moments:
    >>> np.dot(x.T, res.new_weights) - mean_population_moments * np.sum(weights0)
    array([ 1.09139364e-11, -1.36424205e-11, -9.09494702e-13])

    Run the EBW procedure explicitly enforcing bounds on the new ratios:
    >>> res = ebw.entropy_balance(
    ...         mean_population_moments=mean_population_moments,
    ...         x_sample=x, weights0=weights0,
    ...         options={"bounds": (0.95, 1.05)}
    ... )
    >>> res.new_weights
    array([1.05, 0.95, 1.05, ..., 1.05, 0.95, 1.05], shape=(10000,))

    Under these bounds, can't satisfy all three moment conditions simultaneously.
    Satisfy as many constraints as possible, and still fit better than initial weights:
    >>> res.constraint_violations
    array([ 2.39854377e+01,  6.66659616e-10, -5.26597432e-10])

    """
    start_time = timer()

    if shared.inputs_are_invalid(x_sample, weights0, mean_population_moments):
        raise ValueError(
            "Inputs include invalid values (NaNs, non-positive weights, etc)"
        )

    if not options:
        options = {}

    if options.get("bounds"):
        return entropy_balance_elastic(
            mean_population_moments=mean_population_moments,
            x_sample=x_sample,
            weights0=weights0,
            options=options,
        )

    logger.info("\nEntropy Balance Rewighting, Sanders (2024)")
    logger.info(f"Input matrix is sparse? {sp.issparse(x_sample)}")
    n, k = x_sample.shape
    logger.info(f"Problem Size: {n} rows, {k} moments")
    logging_header()

    q: FArr = weights0 / np.sum(weights0)

    if sp.issparse(x_sample):
        x_sample = sparse_array(x_sample)

    multipliers = np.zeros(k)
    m = mean_population_moments
    continue_weighting = True
    n_steps = 0

    failure_state = True
    dual_step = np.full_like(multipliers, np.inf)

    status = {
        "n_steps": -1,
        "f_val": 0.0,
        "norm_Ce": np.array(np.inf),
        "norm_Cd": np.array(np.inf),
        "primal_step_size": np.inf,
        "dual_step_size": np.inf,
        "optimality_violation": np.inf,
        "backtrack": 1.0,
    }

    primal_step = np.full(n, np.inf)
    backtrack = 1.0
    wstar = np.copy(options.get("initial_ratio_guess", np.ones(n))) * q
    dot = sdmkl.dot_product_mkl

    while continue_weighting:
        f_val = ne.evaluate("sum(wstar / q * log(wstar / q) + wstar / q - 1)")
        Ce = np.sum(weights0) * (dot(x_sample.T, wstar) - m)
        Cd = np.log(wstar / q) - dot(x_sample, multipliers)
        optimality_violation = shared.full_problem_violation(Cd, Ce)

        status = {
            "n_steps": status["n_steps"] + 1,  # type: ignore
            "f_val": f_val,
            "norm_Ce": np.linalg.norm(Ce),
            "norm_Cd": np.linalg.norm(Cd),
            "primal_step_size": np.linalg.norm(primal_step * np.sum(weights0)),
            "dual_step_size": np.linalg.norm(dual_step),
            "optimality_violation": optimality_violation,
            "backtrack": backtrack,
        }
        log_status(status)
        wstar_c = dot(sp.diags_array(np.sqrt(wstar), format=sp_fmt_flag), x_sample)

        hess_lowerdiag = sdmkl.gram_matrix_mkl(wstar_c)
        hess = (
            hess_lowerdiag
            + hess_lowerdiag.T
            - sp.diags_array(hess_lowerdiag.diagonal())
        )

        dual_step_penalty = max(
            1e-8, float(1e-5 * np.linalg.norm(np.concatenate((Cd, Ce))) ** 0.55)
        )

        lhs_dual_step = hess + dual_step_penalty * sp.eye_array(k)
        rhs_dual_step = -(Ce / np.sum(weights0) - dot(x_sample.T, wstar * Cd))

        if sp.issparse(lhs_dual_step):
            dual_step = pypardiso.spsolve(
                lhs_dual_step, rhs_dual_step, set_matrix_type=2
            )
        else:
            while True:
                try:
                    dual_step = scipy.linalg.solve(
                        lhs_dual_step + dual_step_penalty * sp.eye_array(k),
                        rhs_dual_step,
                        assume_a="pos",
                    )
                    break
                except np.linalg.LinAlgError:
                    dual_step_penalty *= 10
                    continue
        primal_step = (-Cd + dot(x_sample, np.atleast_1d(dual_step))) * wstar
        backtrack = shared.calculate_feasibile_step_lengths(
            wstar, primal_step, tau=0.995
        )
        primal_new_w = wstar + backtrack * primal_step
        x_times_lm = dot(x_sample, multipliers + dual_step)  # noqa: F841
        dual_new_w = q * ne.evaluate("exp(x_times_lm)")

        Ce_primal = np.sum(weights0) * (dot(x_sample.T, primal_new_w) - m)
        Ce_dual = np.sum(weights0) * (dot(x_sample.T, dual_new_w) - m)

        if np.linalg.norm(Ce_dual) < np.linalg.norm(Ce_primal):
            new_w = dual_new_w
        else:
            new_w = primal_new_w

        if np.any(new_w == 0.0) or (backtrack < 0.01):
            logger.info("Bad step. usually means feasibility in doubt.")
            logger.info(
                "Run entropy_balancing_bounded with bounds=(0.0, None)"
                "\nwhich will return a certificate of infeasibility, along"
                "\nwith a best attempt at a solution."
            )
            failure_state = True
            break

        multipliers += dual_step
        wstar = new_w
        n_steps += 1

        if optimality_violation < options.get("optimality_violation", 1e-5):
            logger.info("Optimality converged.")
            failure_state = False
            break
        elif shared.step_sizes_converged(
            options.get("step_tol", 1e-16),
            primal_step,
            dual_step,
            np.sum(weights0) * dot(x_sample.T, primal_step),
        ):
            logger.info("Step sizes converged.")
            failure_state = False
            break
        elif n_steps > (max_steps := options.get("max_steps", 30)):
            failure_state = True
            logger.info(f"Max steps {max_steps} exceeded.")
            break
        elif np.any([np.isnan(np.linalg.norm(x)) for x in (f_val, Cd, Ce, wstar)]):
            logger.info("NaN in optimality conditions")
            failure_state = True
            break
    logger.info(f"Time elapsed: {timer() - start_time}")
    logger.info(f"Optimization completed, success?: {not failure_state}")

    if dumpfile := options.get("save_problem_data", None):
        shared.dump_problem_to_zip(
            dumpfile, mean_population_moments, x_sample, weights0
        )
    if (dumpfile := options.get("save_failure_data", None)) and failure_state:
        shared.dump_problem_to_zip(
            dumpfile, mean_population_moments, x_sample, weights0
        )

    new_weights = wstar if not failure_state else weights0
    results = shared.EntropyBalanceResults(
        new_weights=new_weights * np.sum(weights0),
        failure_weights=wstar * np.sum(weights0),
        equality_multipliers_estimate=multipliers,
        converged=not failure_state,
        n_iterations=status["n_steps"],  # type: ignore
        constraint_violations=np.sum(weights0) * (dot(x_sample.T, wstar) - m),
    )

    return results


def entropy_balance_elastic(
    *,
    mean_population_moments: FArr,
    x_sample: AnyArray,
    weights0: FArr,
    options: Optional[dict[str, Any]] = None,
) -> shared.EntropyBalanceResults:
    """EBW Reweighting with bound constraints."""
    time_elapsed = timer()

    if not options:
        options = {}

    bounds: tuple[float, Optional[float]] = options.get("bounds", (0.0, None))

    n, k = x_sample.shape
    logger.info("Elastic-Mode Bounded Entropy Balance Reweighting: Sanders (2024)")
    logger.info(f"Input matrix is sparse? {sp.issparse(x_sample)}")
    logger.info(f"Problem Size: {n} rows, {k} moments")
    logging_header()

    if sp.issparse(x_sample):
        x_sample = sparse_array(x_sample)
    A_mat = sdmkl.dot_product_mkl(
        sp.diags_array(weights0, format=sp_fmt_flag), x_sample
    )
    agg_population_moments = mean_population_moments * np.sum(weights0)

    if bounds[0] < 0.0:
        bounds = (0.0, bounds[1])
    if bounds[1] is not None:
        A_ineq = sp.block_array(
            [[sp.eye_array(n), -sp.eye_array(n)]], format=sp_fmt_flag
        )
        bounds = np.concatenate((np.full(n, bounds[0]), -np.full(n, bounds[1])))  # type: ignore
    else:
        A_ineq = sp.eye_array(n, format=sp_fmt_flag)
        bounds = np.full(n, bounds[0], dtype=np.float64)  # type: ignore
    ratio = np.copy(options.get("initial_ratio_guess", np.ones(n)))
    multipliers_eq = np.zeros(k)
    multipliers_ineq = np.zeros(len(bounds)) + 0.05

    mu_s = 0.05
    mu_u = 0.05
    mu_v = 0.05

    cv = A_mat.T.dot(ratio) - agg_population_moments
    eq_violations_plus = np.zeros(k) + 0.01
    eq_violations_plus[cv < 0] = -cv[cv < 0] + 0.01

    eq_violations_minus = np.zeros(k) + 0.01
    eq_violations_minus[cv > 0] = cv[cv > 0] + 0.01

    lm_eqslack_plus = mu_u / eq_violations_plus

    lm_eqslack_minus = mu_u / eq_violations_minus

    slacks = A_ineq.T.dot(ratio) - bounds

    eta: float = options.get(
        "eta", 1.5 * np.max(np.concatenate((lm_eqslack_plus, lm_eqslack_minus)))
    )

    failure_state = True
    status = {
        "n_steps": -1,
        "f_val": 0.0,
        "norm_Ce": np.array(np.inf),
        "norm_Cd": np.array(np.inf),
        "primal_step_size": np.inf,
        "optimality_violation": np.inf,
        "backtrack": 1.0,
    }

    def calc_optimality_violation(*args: FArr) -> FArr:
        out: FArr = np.sqrt(np.sum(np.concatenate(args) ** 2))
        return out

    r_step = np.full(n, np.inf)
    dual_step_eq: FArr = np.full(k, np.inf)
    backtrack = 1.0
    dot = sdmkl.dot_product_mkl
    while True:
        f_val, grad, inverse_hess_diag = shared.criterion(ratio, weights0)
        Cd = 1 / eta * grad - dot(A_mat, multipliers_eq) - dot(A_ineq, multipliers_ineq)
        Ce = (
            dot(A_mat.T, ratio)
            - agg_population_moments
            + eq_violations_plus
            - eq_violations_minus
        )
        Ci = dot(A_ineq.T, ratio) - slacks - bounds
        Cu = 1.0 - multipliers_eq - lm_eqslack_plus
        Cv = 1.0 + multipliers_eq - lm_eqslack_minus
        Clu = eq_violations_plus * lm_eqslack_plus - mu_u
        Clv = eq_violations_minus * lm_eqslack_minus - mu_v
        Cs = slacks * multipliers_ineq - mu_s

        optimality_violation = calc_optimality_violation(
            Cd, Ce, Ci, Cu, Cv, Clu, Clv, Cs
        )

        status = {
            "n_steps": status["n_steps"] + 1,  # type: ignore
            "f_val": f_val,
            "norm_Ce": np.linalg.norm(Ce),
            "norm_Cd": np.linalg.norm(Cd),
            "primal_step_size": np.linalg.norm(r_step),
            "dual_step_size": np.linalg.norm(dual_step_eq),
            "optimality_violation": optimality_violation,
            "backtrack": backtrack,
        }
        log_status(status)

        dual_step_penalty = max(
            1e-8,
            float(
                1e-5
                * np.linalg.norm(np.concatenate((Cd, Ce, Ci, Cu, Cv, Clu, Clv, Cs)))
                ** 0.55
            ),
        )

        l_i_invs = sp.diags_array(multipliers_ineq / slacks, format=sp_fmt_flag)
        h_tilde_diag: FArr = (
            1 / eta * 1 / inverse_hess_diag
            + shared.chain_dot(A_ineq, l_i_invs, A_ineq.T).diagonal()
        )
        htilde = sp.diags_array(h_tilde_diag, format=sp_fmt_flag)

        inv_h_sqrt = sp.diags_array(np.sqrt(1 / h_tilde_diag), format=sp_fmt_flag)
        AtHA_sqrt = sdmkl.dot_product_mkl(inv_h_sqrt, A_mat)
        AtHA_lowerdiag = sdmkl.gram_matrix_mkl(AtHA_sqrt, cast=True).T
        AtHA = (
            AtHA_lowerdiag
            + AtHA_lowerdiag.T
            - sp.diags_array(AtHA_lowerdiag.diagonal())
        )

        lhs = AtHA + sp.diags_array(
            eq_violations_plus / lm_eqslack_plus
            + eq_violations_minus / lm_eqslack_minus,
            format=sp_fmt_flag,
        )
        rhs = (
            Ce
            + (eq_violations_minus / lm_eqslack_minus)
            * (Cv + 1 / eq_violations_minus * Clv)
            - (eq_violations_plus / lm_eqslack_plus)
            * (Cu + 1 / eq_violations_plus * Clu)
            - shared.chain_dot(
                A_mat.T,
                sp.diags_array(1 / htilde.diagonal(), format=sp_fmt_flag),
                Cd
                + A_ineq.dot(
                    multipliers_ineq / slacks * (Ci + 1 / multipliers_ineq * Cs)
                ),
            )
        )
        if sp.issparse(lhs):
            dual_step_eq = -pypardiso.spsolve(
                lhs + dual_step_penalty * sp.eye_array(k), rhs
            )
        else:
            while True:
                try:
                    dual_step_eq = -scipy.linalg.solve(
                        lhs + dual_step_penalty * sp.eye_array(k), rhs
                    )
                    break
                except np.linalg.LinAlgError:
                    dual_step_penalty *= 10
                    continue
        r_step = (
            1
            / h_tilde_diag
            * (
                dot(A_mat, dual_step_eq)
                - Cd
                - dot(A_ineq, (Ci * multipliers_ineq + Cs) / slacks)
            ).ravel()
        )
        l_i_step = (
            multipliers_ineq
            / slacks
            * (-dot(A_ineq.T, r_step) - Ci - 1 / multipliers_ineq * Cs)
        )

        u_step: FArr = (eq_violations_plus / lm_eqslack_plus) * (
            dual_step_eq - (Cu + 1 / eq_violations_plus * Clu)
        )
        v_step: FArr = (eq_violations_minus / lm_eqslack_minus) * (
            -dual_step_eq - (Cv + 1 / eq_violations_minus * Clv)
        )

        s_step = (
            -slacks
            - (1 / multipliers_ineq) * slacks * l_i_step
            + mu_s / multipliers_ineq
        )
        l_u_step = 1 / eq_violations_plus * (-Clu - lm_eqslack_plus * u_step)
        l_v_step = 1 / eq_violations_minus * (-Clv - lm_eqslack_minus * v_step)

        def fsl(x: FArr, y: FArr) -> float:
            return shared.calculate_feasibile_step_lengths(x, y, tau=0.995)

        largest_slack_step = fsl(slacks, s_step)
        largest_dual_ineq_step = fsl(multipliers_ineq, l_i_step)
        largest_dual_u_step = fsl(lm_eqslack_plus, l_u_step)
        largest_dual_v_step = fsl(lm_eqslack_minus, l_v_step)
        largest_dual_step = min(
            largest_dual_ineq_step, largest_dual_u_step, largest_dual_v_step
        )
        largest_u_step = fsl(eq_violations_plus, u_step)
        largest_v_step = fsl(eq_violations_minus, v_step)
        largest_primal_step = min(largest_u_step, largest_v_step, largest_slack_step)

        ratio += r_step * largest_primal_step
        slacks += s_step * largest_primal_step
        multipliers_eq += dual_step_eq * largest_dual_step
        multipliers_ineq += l_i_step * largest_dual_step
        eq_violations_plus += u_step * largest_primal_step
        eq_violations_minus += v_step * largest_primal_step
        lm_eqslack_plus += l_u_step * largest_dual_step
        lm_eqslack_minus += l_v_step * largest_dual_step

        zeta = np.min(slacks * multipliers_ineq) / np.mean(slacks * multipliers_ineq)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_s = sigma * np.mean(slacks * multipliers_ineq)

        zeta = np.min(eq_violations_plus * lm_eqslack_plus) / np.mean(
            eq_violations_plus * lm_eqslack_plus
        )
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_u = sigma * np.mean(eq_violations_plus * lm_eqslack_plus)

        zeta = np.min(eq_violations_minus * lm_eqslack_minus) / np.mean(
            eq_violations_minus * lm_eqslack_minus
        )
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_v = sigma * np.mean(eq_violations_minus * lm_eqslack_minus)

        if eta < (
            max_lm := max(
                np.max(np.abs(multipliers_eq)),
                np.max(multipliers_ineq),
                np.max(lm_eqslack_plus),
                np.max(lm_eqslack_minus),
            )
        ):
            eta = 2.0 * max_lm

        alternate_optimality_violation = calc_optimality_violation(
            np.exp(
                eta
                * (dot(A_mat, multipliers_eq) + dot(A_ineq, multipliers_ineq))
                / weights0
            )
            - ratio,
            Ce,
            Ci,
            Cu,
            Cv,
            Clu,
            Clv,
            Cs,
        )

        if min(
            float(optimality_violation), float(alternate_optimality_violation)
        ) < options.get("optimality_violation", 1e-5):
            failure_state = False
            logger.info("Optimality converged.")
            break
        elif shared.step_sizes_converged(
            options.get("step_tol", 1e-8),
            r_step * largest_primal_step,
            dual_step_eq * largest_dual_step,
            np.sum(weights0) * dot(x_sample.T, r_step * largest_primal_step),
        ):
            logger.info("Step sizes converged.")
            failure_state = False
            break
        elif status["n_steps"] > (max_steps := options.get("max_steps", 100)):
            logger.info(f"Max steps {max_steps} exceeded.")
            failure_state = True
            break

    logger.info(f"Time elapsed: {timer() - time_elapsed}")
    logger.info(f"Optimization completed, success?: {not failure_state}")

    if dumpfile := options.get("save_problem_data", None):
        shared.dump_problem_to_zip(
            dumpfile, mean_population_moments, x_sample, weights0
        )
    if (dumpfile := options.get("save_failure_data", None)) and failure_state:
        shared.dump_problem_to_zip(
            dumpfile, mean_population_moments, x_sample, weights0
        )

    new_weights = ratio * weights0 if not failure_state else weights0
    results = shared.EntropyBalanceResults(
        new_weights=new_weights,
        failure_weights=ratio * weights0,
        equality_multipliers_estimate=multipliers_eq,
        moment_slack_multipliers_estimate=np.concatenate(
            (lm_eqslack_plus, lm_eqslack_minus)
        ),
        converged=not failure_state,
        n_iterations=status["n_steps"],  # type: ignore
        constraint_violations=A_mat.T.dot(ratio) - agg_population_moments,
        eta=eta,
    )
    return results


def logging_header() -> None:
    """Write the column headers to log."""
    logger.info(
        f"{'#':^5}{'Criterion':^14}{'||Eq. Const.||':^18}{'||FOC Lagr.||':^14}{'PrimalStepSize':^15}{'DualStepSize':^15}{'Opt. Violation.':^14}"
    )


def log_status(s: dict[str, Any]) -> None:
    """Log relevant problem statistics."""
    logger.info(
        f"{s['n_steps']:^5}{s['f_val']:^14.6f}{s['norm_Ce']:^18.6f}{s['norm_Cd']:^14.4f}{s['primal_step_size']:^15.8f}{s['dual_step_size']:^15.8f}{s['optimality_violation']:^14.8f}"
    )
