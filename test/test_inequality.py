import numpy as np
import scipy.sparse as sp

from entropy_balance_weighting import shared
from entropy_balance_weighting.ebw_penalty import (
    _penalty_bounded_primal_step,
    _recover_penalty_bounded_dual_step,
    _recover_penalty_bounded_slacks_step,
)
from entropy_balance_weighting.typing import Callable, FArr


def test_inequality_steps() -> None:
    """Test the derivation of inequality bounded step matrices."""
    n = 25
    k = 2
    np.random.seed(102)
    weights0 = np.ones(n)
    X = np.random.random((n, k))
    mean_population_moments = np.mean(X[-n // 2, :], 0)
    A_mat = weights0[:, None] * X
    A_ineq = np.block([[np.eye(n), -np.eye(n)]])
    agg_population_moments = mean_population_moments * np.sum(weights0)

    bounds = np.concatenate((np.full(n, 0.0), -np.full(n, 2.0)))

    mu = 0.02
    delta = 1e-5

    def _criterion(r: FArr) -> FArr:
        out: FArr = np.sum(weights0 * (r * np.log(r) - r + 1))
        return out

    def _grad_criterion(r: FArr) -> FArr:
        out: FArr = weights0 * np.log(r)
        return out

    def _hess_criterion(r: FArr) -> FArr:
        out: FArr = np.diag(weights0 / r)
        return out

    def _Ce(r: FArr) -> FArr:
        out: FArr = A_mat.T.dot(r) - agg_population_moments
        return out

    def _Cd(r: FArr, s: FArr, l_e: FArr, l_i: FArr) -> FArr:
        out: FArr = _grad_criterion(r)
        out -= A_mat.dot(l_e)
        out -= A_ineq.dot(l_i)
        return out

    def _CI(r: FArr, s: FArr) -> FArr:
        out: FArr = A_ineq.T.dot(r) - s - bounds
        return out

    def _Cs(s: FArr, l_i: FArr) -> FArr:
        out: FArr = s * l_i - mu
        return out

    def _calculate_step_sizes_fullmat(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> tuple[FArr, FArr, FArr, FArr]:
        hess = _hess_criterion(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)
        bigS = np.diag(s)
        bigLI = np.diag(l_i)
        z = np.zeros
        rhs = np.concatenate([cd, ce, ci, cs])
        lhs = np.block(
            [
                [hess, A_mat, A_ineq, z((n, 2 * n))],
                [A_mat.T, -delta * np.eye(k), z((k, 2 * n)), z((k, 2 * n))],
                [A_ineq.T, z((2 * n, k)), z((2 * n, 2 * n)), -np.eye(2 * n)],
                [z((2 * n, n)), z((2 * n, k)), -np.eye(2 * n), bigS * bigLI],
            ]
        )
        step_size = np.linalg.solve(lhs, -rhs)
        pr, pe, pi, ps = (
            step_size[0:n],
            -step_size[n : n + k],
            -step_size[n + k : n + k + 2 * n],
            step_size[n + k + 2 * n :],
        )
        return pr, pe, pi, ps

    r0 = np.ones(n)
    s0 = np.zeros(2 * n) + 1.0
    l_e = np.zeros(k)
    l_i = np.zeros(2 * n) + 0.5

    assert not np.allclose(_Ce(r0), 0.0)

    pr, pe, pi, ps = _calculate_step_sizes_fullmat(r0, s0, l_e, l_i)
    r1 = r0 + pr
    assert (
        _criterion(r1) - _criterion(r0)
    ) > 0.0  # Have to give up some criterion to get feasibile
    assert np.linalg.norm(_Ce(r1)) <= 1 / 10 * np.linalg.norm(
        _Ce(r0)
    )  # with regularized step we don't have exact feasibility

    def _calculate_step_sizes_noslacks(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> tuple[FArr, FArr, FArr]:
        hess = _hess_criterion(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)
        z = np.zeros
        rhs = np.concatenate([cd, ce, ci + (1 / l_i) * cs])
        lhs = np.block(
            [
                [hess, A_mat, A_ineq],
                [A_mat.T, -delta * np.eye(k), z((k, 2 * n))],
                [A_ineq.T, z((2 * n, k)), -np.diag(s * 1 / l_i)],
            ]
        )
        step_size = np.linalg.solve(lhs, -rhs)
        pr, pe, pi = (
            step_size[0:n],
            -step_size[n : n + k],
            -step_size[n + k : n + k + 2 * n],
        )
        return pr, pe, pi

    pr_nos, pe_nos, pi_nos = _calculate_step_sizes_noslacks(r0, s0, l_e, l_i)
    assert np.allclose(pr, pr_nos)
    assert np.allclose(pe, pe_nos)
    assert np.allclose(pi, pi_nos)

    def _calculate_step_sizes_noslacks_or_ineq(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> tuple[FArr, FArr]:
        hess = _hess_criterion(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)
        rhs = np.concatenate([cd + (A_ineq * l_i / s).dot(ci + 1 / l_i * cs), ce])
        lhs = np.block(
            [
                [hess + (A_ineq * l_i / s) @ A_ineq.T, A_mat],
                [A_mat.T, -delta * np.eye(k)],
            ]
        )
        step_size = np.linalg.solve(lhs, -rhs)
        pr, pe = step_size[0:n], -step_size[n : n + k]
        return pr, pe

    pr_nos_or_i, pe_nos_or_i = _calculate_step_sizes_noslacks_or_ineq(r0, s0, l_e, l_i)
    assert np.allclose(pr_nos_or_i, pr)
    assert np.allclose(pe_nos_or_i, pe)

    def _calculate_step_sizes_eqdual_only(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> FArr:
        hess = _hess_criterion(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)

        H_sigma = hess + (A_ineq * l_i / s) @ A_ineq.T
        lhs = A_mat.T @ np.linalg.inv(H_sigma) @ A_mat + delta * np.eye(k)

        rhs = ce - (A_mat.T @ np.linalg.inv(H_sigma)).dot(
            cd + (A_ineq * l_i / s).dot(ci + 1 / l_i * cs)
        )

        step_size: FArr = np.linalg.solve(lhs, -rhs)
        return step_size

    def _recover_pi_step(pr: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr) -> FArr:
        out: FArr = -(ci + 1 / l_i * cs)
        out = out - A_ineq.T.dot(pr)
        out = out * l_i * 1 / s
        return out

    def _recover_slack_step(pi: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr) -> FArr:
        out: FArr = -cs
        out = out - s * pi
        out = out / l_i
        return out

    def _recover_primal_step(
        pe: FArr, r: FArr, cd: FArr, ce: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr
    ) -> FArr:
        hess = _hess_criterion(r)
        lagrange_inv_hess = 1 / (np.diag(hess) + (l_i / s)[0:n] + (l_i / s)[n:])
        out: FArr = -(cd + (A_ineq * l_i / s).dot(ci + 1 / l_i * cs))
        out = out + A_mat.dot(pe)
        out = out * lagrange_inv_hess
        return out

    pe_only = _calculate_step_sizes_eqdual_only(r0, s0, l_e, l_i)
    assert np.allclose(pr, pr_nos)
    assert np.allclose(pr, pr_nos_or_i)
    assert np.allclose(
        pr,
        _recover_primal_step(
            pe, r0, _Cd(r0, s0, l_e, l_i), _Ce(r0), _CI(r0, s0), _Cs(s0, l_i), l_i, s0
        ),
    )

    assert np.allclose(pi, pi_nos)
    assert np.allclose(pi_nos, _recover_pi_step(pr, _CI(r0, s0), _Cs(s0, l_i), l_i, s0))

    assert np.allclose(ps, _recover_slack_step(pi, _CI(r0, s0), _Cs(s0, l_i), l_i, s0))

    assert np.allclose(pe, pe_only)


def test_ineq_bounds() -> None:
    """Test the derivation of inequality bounded step matrices."""
    n = 100
    k = 2
    np.random.seed(99)
    weights0 = np.ones(n)
    X = np.random.random((n, k))
    mean_population_moments = np.mean(X[10:, :], 0)
    A_mat = weights0[:, None] * X
    agg_population_moments = mean_population_moments * np.sum(weights0)

    def _criterion(r: FArr) -> FArr:
        out: FArr = np.sum(weights0 * (r * np.log(r) - r + 1))
        return out

    def _grad_criterion(r: FArr) -> FArr:
        out: FArr = weights0 * np.log(r)
        return out

    def _hess_diag_criterion(r: FArr) -> FArr:
        out: FArr = weights0 / r
        return out

    minimize_penalty_bounded_full(
        _criterion,
        _grad_criterion,
        _hess_diag_criterion,
        A_mat,
        agg_population_moments,
        lbound=0.5,
        ubound=2.0,
    )

    minimize_seperable_full(
        _criterion,
        _grad_criterion,
        _hess_diag_criterion,
        A_mat,
        agg_population_moments,
        lbound=0.5,
        ubound=2.0,
    )


def minimize_seperable(
    crit: Callable[[FArr], FArr],
    grad: Callable[[FArr], FArr],
    hess_diag: Callable[[FArr], FArr],
    A: FArr,
    b: FArr,
    lbound: float,
    ubound: float,
) -> None:
    """Minimize a linear + bound-constrained additively separable criterion function."""
    n, k = A.shape
    A_ineq = np.block([[np.eye(n), -np.eye(n)]])

    if lbound > 1.0:
        raise ValueError("lower bound above 1 not acceptable")
    elif ubound < 1.0:
        raise ValueError("upper bound less than 1 unacceptable")
    bounds = np.concatenate((np.full(n, lbound), -np.full(n, ubound)))

    def _calculate_step_sizes_eqdual_only(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> FArr:
        hess_diagvals = hess_diag(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)

        H_sigma = np.diag(hess_diagvals) + (A_ineq * l_i / s) @ A_ineq.T
        lhs = A.T @ np.linalg.inv(H_sigma) @ A + delta * np.eye(k)

        rhs = ce - (A.T @ np.linalg.inv(H_sigma)).dot(
            cd + (A_ineq * l_i / s).dot(ci + 1 / l_i * cs)
        )

        step_size: FArr = np.linalg.solve(lhs, -rhs)
        return step_size

    def _Ce(r: FArr) -> FArr:
        out: FArr = A.T.dot(r) - b
        return out

    def _Cd(r: FArr, s: FArr, l_e: FArr, l_i: FArr) -> FArr:
        out: FArr = grad(r)
        out -= A.dot(l_e)
        out -= A_ineq.dot(l_i)
        return out

    def _CI(r: FArr, s: FArr) -> FArr:
        out: FArr = A_ineq.T.dot(r) - s - bounds
        return out

    def _Cs(s: FArr, l_i: FArr) -> FArr:
        out: FArr = s * l_i - mu
        return out

    def _recover_pi_step(pr: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr) -> FArr:
        out: FArr = -(ci + 1 / l_i * cs)
        out = out - A_ineq.T.dot(pr)
        out = out * l_i * 1 / s
        return out

    def _recover_slack_step(pi: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr) -> FArr:
        out: FArr = -cs
        out = out - s * pi
        out = out / l_i
        return out

    def _recover_primal_step(
        pe: FArr, r: FArr, cd: FArr, ce: FArr, ci: FArr, cs: FArr, l_i: FArr, s: FArr
    ) -> FArr:
        hessdiag = hess_diag(r)
        lagrange_inv_hess = 1 / (hessdiag + (l_i / s)[0:n] + (l_i / s)[n:])
        out: FArr = -(cd + (A_ineq * l_i / s).dot(ci + 1 / l_i * cs))
        out = out + A.dot(pe)
        out = out * lagrange_inv_hess
        return out

    def take_step(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> tuple[FArr, FArr, FArr, FArr]:
        ce = _Ce(r)
        ci = _CI(r, s)
        cd = _Cd(r, s, l_e, l_i)
        cs = _Cs(s, l_i)
        pe = _calculate_step_sizes_eqdual_only(r, s, l_e, l_i)
        pr = _recover_primal_step(pe, r, cd, ce, ci, cs, l_i, s)
        pi = _recover_pi_step(pr, ci, cs, l_i, s)
        ps = _recover_slack_step(pi, ci, cs, l_i, s)
        return pr, pe, pi, ps

    mu = 0.02
    delta = 1e-5

    r0 = np.ones(n)
    s0 = np.concatenate((np.full(n, 1 - lbound), np.full(n, ubound - 1)))
    l_e = np.zeros(k)
    l_i = np.ones(2 * n)

    def error_fn(r: FArr, s: FArr, l_e: FArr, l_i: FArr) -> tuple[np.floating, ...]:
        error_d = np.linalg.norm(_Cd(r0, s0, l_e, l_i))
        error_s = np.linalg.norm(_Cs(s0, l_i))
        error_e = np.linalg.norm(_Ce(r0))
        error_i = np.linalg.norm(_CI(r0, s0))
        return error_d, error_s, error_e, error_i

    i = 1
    while np.linalg.norm((error_fn(r0, s0, l_e, l_i))) > 1e-5:
        pr, pe, pi, ps = take_step(r0, s0, l_e, l_i)

        largest_slack_step = float(
            np.min(-0.995 * s0[ps < 0] / ps[ps < 0], initial=np.inf)
        )
        backtrack_slack = min(1.0, largest_slack_step)

        largest_dual_step = float(
            np.min(-0.995 * l_i[pi < 0] / pi[pi < 0], initial=np.inf)
        )
        backtrack_dual = min(1.0, largest_dual_step)

        r0 += pr * backtrack_slack
        s0 += ps * backtrack_slack
        l_e += pe * backtrack_dual
        l_i += pi * backtrack_dual

        zeta = np.min(s0 * l_i) / np.mean(s0 * l_i)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu = sigma * np.mean(s0 * l_i)
        i += 1


def minimize_seperable_full(
    crit: Callable[[FArr], FArr],
    grad: Callable[[FArr], FArr],
    hess_diag: Callable[[FArr], FArr],
    A: FArr,
    b: FArr,
    lbound: float,
    ubound: float,
) -> None:
    """Minimize a linear + bound-constrained additively separable criterion function."""
    n, k = A.shape
    A_ineq = np.block([[np.eye(n), -np.eye(n)]])

    if lbound > 1.0:
        raise ValueError("lower bound above 1 not acceptable")
    elif ubound < 1.0:
        raise ValueError("upper bound less than 1 unacceptable")
    bounds = np.concatenate((np.full(n, lbound), -np.full(n, ubound)))

    def _Ce(r: FArr) -> FArr:
        out: FArr = A.T.dot(r) - b
        return out

    def _Cd(r: FArr, s: FArr, l_e: FArr, l_i: FArr) -> FArr:
        out: FArr = grad(r)
        out -= A.dot(l_e)
        out -= A_ineq.dot(l_i)
        return out

    def _CI(r: FArr, s: FArr) -> FArr:
        out: FArr = A_ineq.T.dot(r) - s - bounds
        return out

    def _Cs(s: FArr, l_i: FArr) -> FArr:
        out: FArr = s * l_i - mu
        return out

    def take_step(
        r: FArr, s: FArr, l_e: FArr, l_i: FArr
    ) -> tuple[FArr, FArr, FArr, FArr]:
        hess = hess_diag(r)
        cd = _Cd(r, s, l_e, l_i)
        ce = _Ce(r)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)
        bigS = np.diag(s)
        bigLI = np.diag(l_i)
        z = np.zeros
        rhs = np.concatenate([cd, ce, ci, cs])
        lhs = np.block(
            [
                [np.diag(hess), -A, -A_ineq, z((n, 2 * n))],
                [A.T, delta * np.eye(k), z((k, 2 * n)), z((k, 2 * n))],
                [A_ineq.T, z((2 * n, k)), z((2 * n, 2 * n)), -np.eye(2 * n)],
                [z((2 * n, n)), z((2 * n, k)), bigS, bigLI],
            ]
        )
        step_size = np.linalg.solve(lhs, -rhs)
        pr, pe, pi, ps = (
            step_size[0:n],
            step_size[n : n + k],
            step_size[n + k : n + k + 2 * n],
            step_size[n + k + 2 * n :],
        )
        return pr, pe, pi, ps

    mu = 0.02
    delta = 1e-5

    r0 = np.ones(n)
    s0 = np.concatenate((np.full(n, 1 - lbound), np.full(n, ubound - 1)))
    l_e = np.zeros(k)
    l_i = np.ones(2 * n)

    def error_fn(r: FArr, s: FArr, l_e: FArr, l_i: FArr) -> tuple[np.floating, ...]:
        error_d = np.linalg.norm(_Cd(r0, s0, l_e, l_i))
        error_s = np.linalg.norm(_Cs(s0, l_i))
        error_e = np.linalg.norm(_Ce(r0))
        error_i = np.linalg.norm(_CI(r0, s0))
        return error_d, error_s, error_e, error_i

    i = 1
    while np.linalg.norm((error_fn(r0, s0, l_e, l_i))) > 1e-5:
        pr, pe, pi, ps = take_step(r0, s0, l_e, l_i)

        largest_slack_step = float(np.min(-0.995 * s0[ps < 0] / ps[ps < 0]))
        backtrack_slack = min(1.0, largest_slack_step)

        largest_dual_step = float(np.min(-0.995 * l_i[pi < 0] / pi[pi < 0]))
        backtrack_dual = min(1.0, largest_dual_step)

        r0 += pr * backtrack_slack
        s0 += ps * backtrack_slack
        l_e += pe * backtrack_dual
        l_i += pi * backtrack_dual

        zeta = np.min(s0 * l_i) / np.mean(s0 * l_i)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu = sigma * np.mean(s0 * l_i)
        i += 1


def minimize_penalty_bounded_full(
    crit: Callable[[FArr], FArr],
    grad: Callable[[FArr], FArr],
    hess_diag: Callable[[FArr], FArr],
    A: FArr,
    b: FArr,
    lbound: float,
    ubound: float,
) -> None:
    """Minimize a linear + bound-constrained additively separable criterion function."""
    n, k = A.shape
    A_ineq = np.block([[np.eye(n), -np.eye(n)]])

    if lbound > 1.0:
        raise ValueError("lower bound above 1 not acceptable")
    elif ubound < 1.0:
        raise ValueError("upper bound less than 1 unacceptable")
    bounds = np.concatenate((np.full(n, lbound), -np.full(n, ubound)))
    penalty_parameter = np.ones(k)

    def _Ce(r: FArr) -> FArr:
        out: FArr = A.T.dot(r) - b
        return out

    def _Cd(r: FArr, s: FArr, l_i: FArr) -> FArr:
        out: FArr = grad(r)
        out += shared.chain_dot(
            A, sp.diags_array(penalty_parameter, format="csr"), A.T.dot(r) - b
        )
        out -= A_ineq.dot(l_i)
        return out

    def _CI(r: FArr, s: FArr) -> FArr:
        out: FArr = A_ineq.T.dot(r) - s - bounds
        return out

    def _Cs(s: FArr, l_i: FArr) -> FArr:
        out: FArr = s * l_i - mu
        return out

    def take_step(r: FArr, s: FArr, l_i: FArr) -> tuple[FArr, FArr, FArr]:
        hess = hess_diag(r)
        cd = _Cd(r, s, l_i)
        ci = _CI(r, s)
        cs = _Cs(s, l_i)
        bigS = np.diag(s)
        bigLI = np.diag(l_i)
        z = np.zeros
        rhs = np.concatenate([cd, ci, cs])
        lhs = np.block(
            [
                [
                    np.diag(hess)
                    + shared.chain_dot(
                        A, sp.diags_array(penalty_parameter, format="csr"), A.T
                    ),
                    -A_ineq,
                    z((n, 2 * n)),
                ],
                [A_ineq.T, z((2 * n, 2 * n)), -np.eye(2 * n)],
                [z((2 * n, n)), bigS, bigLI],
            ]
        )
        step_size = np.linalg.solve(lhs, -rhs)
        pr, pi, ps = (
            step_size[0:n],
            step_size[n : n + 2 * n],
            step_size[n + 2 * n :],
        )
        return pr, pi, ps

    def take_step_schur_approach(
        r: FArr, s: FArr, l_i: FArr
    ) -> tuple[FArr, FArr, FArr]:
        hess = hess_diag(r)
        cd = _Cd(r, s, l_i)
        pr = _penalty_bounded_primal_step(
            cd,
            l_i,
            mu,
            s,
            sp.diags_array(hess, format="csr"),
            penalty_parameter,
            sp.csr_array(A),
            sp.csr_array(A_ineq),
        )
        pi = _recover_penalty_bounded_dual_step(pr, s, mu, l_i, sp.csr_array(A_ineq))
        ps = _recover_penalty_bounded_slacks_step(pr, sp.csr_array(A_ineq))
        return pr, pi, ps

    mu = 0.02

    r0 = np.ones(n)
    s0 = np.concatenate((np.full(n, 1 - lbound), np.full(n, ubound - 1)))
    l_i = np.ones(2 * n)

    def error_fn(r: FArr, s: FArr, l_i: FArr) -> tuple[np.floating, ...]:
        error_d = np.linalg.norm(_Cd(r0, s0, l_i))
        error_s = np.linalg.norm(_Cs(s0, l_i))
        error_i = np.linalg.norm(_CI(r0, s0))
        return error_d, error_s, error_i

    i = 1
    while np.linalg.norm((error_fn(r0, s0, l_i))) > 1e-5:
        pr, pi, ps = take_step(r0, s0, l_i)
        alt_pr, alt_pi, alt_ps = take_step_schur_approach(r0, s0, l_i)
        assert np.allclose(pr, alt_pr)
        assert np.allclose(pi, alt_pi)
        assert np.allclose(ps, alt_ps)

        largest_slack_step = float(np.min(-0.995 * s0[ps < 0] / ps[ps < 0]))
        backtrack_slack = min(1.0, largest_slack_step)

        largest_dual_step = float(np.min(-0.995 * l_i[pi < 0] / pi[pi < 0]))
        backtrack_dual = min(1.0, largest_dual_step)

        r0 += pr * backtrack_slack
        s0 += ps * backtrack_slack
        l_i += pi * backtrack_dual

        zeta = np.min(s0 * l_i) / np.mean(s0 * l_i)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu = sigma * np.mean(s0 * l_i)
        i += 1
