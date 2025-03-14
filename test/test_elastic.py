import numpy as np
import scipy.sparse as sp

from entropy_balance_weighting import entropy_balance
from entropy_balance_weighting.typing import FArr


def test_elastic_feasible_randomw0() -> None:
    """Test that if there is some feasible solution, bounded finds it."""
    np.random.seed(10052)
    sample_size = 10000
    n_moments = 3
    x = np.random.randint(0, 2, size=(sample_size, n_moments)).astype(np.float64)
    mean_population_moments = np.array([0.5, 0.5, 0.5])
    weights0 = np.random.random(sample_size) + 0.5
    x_sparse = sp.csr_array(x)
    res_normal = entropy_balance(
        x_sample=x_sparse,
        weights0=weights0,
        mean_population_moments=mean_population_moments,
    )
    res_bounded = entropy_balance(
        x_sample=x_sparse,
        weights0=weights0,
        mean_population_moments=mean_population_moments,
        options={"bounds": (0.0, None)},
    )
    assert np.allclose(res_normal.new_weights, res_bounded.new_weights)
    assert np.allclose(
        x.T.dot(res_bounded.new_weights) - np.sum(weights0) * mean_population_moments,
        0.0,
    )


def test_elastic_infeasible() -> None:
    """Test that even if there is no feasible solution, it works."""
    np.random.seed(10052)
    sample_size = 500
    n_moments = 20
    x = np.random.randint(0, 2, size=(sample_size, n_moments)).astype(np.float64)
    mean_population_moments = np.full(n_moments, 0.5)
    weights0 = np.random.random(sample_size) + 0.5
    weights0 = weights0 / weights0.mean()
    x_sparse = sp.csr_array(x)
    res_bounded = entropy_balance(
        x_sample=x_sparse,
        weights0=weights0,
        mean_population_moments=mean_population_moments,
        options={"bounds": (0.95, 1.05)},
    )
    assert res_bounded.converged
    assert np.linalg.norm(res_bounded.constraint_violations) > 10.0


def test_elastic_infeasible_penalty() -> None:
    """Test that a higher penalty gives a closer fit in case of infeasibility."""
    np.random.seed(10052)
    sample_size = 500
    n_moments = 20
    x = np.random.randint(0, 2, size=(sample_size, n_moments)).astype(np.float64)
    mean_population_moments = np.full(n_moments, 0.5)
    weights0 = np.random.random(sample_size) + 0.5
    weights0 = weights0 / weights0.mean()
    x_sparse = sp.csr_array(x)
    res_base = entropy_balance(
        x_sample=x_sparse,
        weights0=weights0,
        mean_population_moments=mean_population_moments,
        options={"bounds": (0.85, 1.25), "eta": 1.0},
    )
    if res_base.eta is None:
        res_base.eta = 1.0
    res_bounded = entropy_balance(
        x_sample=x_sparse,
        weights0=weights0,
        mean_population_moments=mean_population_moments,
        options={"bounds": (0.85, 1.25), "eta": res_base.eta * 10.0},
    )
    assert res_base.converged
    assert res_bounded.converged
    assert np.linalg.norm(res_base.constraint_violations, ord=1) > np.linalg.norm(
        res_bounded.constraint_violations, ord=1
    )


def ebw_elastic(n: int, eta: float, seed: int, bounds: tuple[float, float]) -> FArr:
    """Test the sparse implementation."""
    k = 3
    np.random.seed(seed)
    weights0 = np.ones(n)
    X = np.random.random((n, k))
    mean_population_moments = np.mean(X[-n // 2 :, :], 0)
    X[:, -2] = X[:, -1]
    res = entropy_balance(
        mean_population_moments=mean_population_moments,
        x_sample=sp.csr_array(X),
        weights0=weights0,
        options={"bounds": bounds, "eta": eta},
    )
    return res.new_weights


def elastic_mode_full(
    n: int, eta: float, seed: int, bounds: tuple[float, float]
) -> tuple[FArr, FArr]:
    """Test the derivation of inequality bounded step matrices."""
    k = 3
    np.random.seed(seed)
    weights0 = np.ones(n)
    X = np.random.random((n, k))
    mean_population_moments = np.mean(X[-n // 2 :, :], 0)
    X[:, -2] = X[:, -1]
    A_mat = weights0[:, None] * X
    A_ineq = np.block([[np.eye(n), -np.eye(n)]])
    agg_population_moments = mean_population_moments * np.sum(weights0)

    bounds = np.concatenate(
        (np.full(n, bounds[0]), -np.full(n, bounds[1]))
    )  # negative is correct
    delta = 1e-4

    def _criterion(r: FArr) -> FArr:
        out: FArr = np.sum(weights0 * (r * np.log(r) - r + 1))
        return out

    def _grad_criterion(r: FArr) -> FArr:
        out: FArr = weights0 * np.log(r)
        return out

    def _hess_criterion(r: FArr) -> FArr:
        out: FArr = np.diag(weights0 / r)
        return out

    def _Cd(r: FArr, s: FArr, l_e: FArr, l_i: FArr, eta: float) -> FArr:
        out: FArr = 1 / eta * _grad_criterion(r)
        out -= A_mat.dot(l_e)
        out -= A_ineq.dot(l_i)
        return out

    def _Ce(r: FArr, u: FArr, v: FArr) -> FArr:
        out: FArr = A_mat.T.dot(r) - agg_population_moments + u - v
        return out

    def _CI(r: FArr, s: FArr) -> FArr:
        out: FArr = A_ineq.T.dot(r) - s - bounds
        return out

    def _Cu(l_e: FArr, l_u: FArr, eta: float) -> FArr:
        out: FArr = 1.0 - l_e - l_u
        return out

    def _Cv(l_e: FArr, l_v: FArr, eta: float) -> FArr:
        out: FArr = 1.0 + l_e - l_v
        return out

    def _Clu(u: FArr, l_u: FArr, mu: float) -> FArr:
        out: FArr = u * l_u - mu
        return out

    def _Clv(v: FArr, l_v: FArr, mu: float) -> FArr:
        out: FArr = v * l_v - mu
        return out

    def _Cs(s: FArr, l_i: FArr, mu: float) -> FArr:
        out: FArr = s * l_i - mu
        return out

    def _calculate_step_sizes_fullmat(
        r: FArr,
        u: FArr,
        v: FArr,
        s: FArr,
        l_e: FArr,
        l_i: FArr,
        l_u: FArr,
        l_v: FArr,
        mu_s: float,
        mu_u: float,
        mu_v: float,
        eta: float,
    ) -> dict[str, FArr]:
        hess = _hess_criterion(r)
        cd = _Cd(r, s, l_e, l_i, eta)
        ce = _Ce(r, u, v)
        ci = _CI(r, s)
        cu = _Cu(l_e, l_u, 1.0)
        cv = _Cv(l_e, l_v, 1.0)
        clu = _Clu(u, l_u, mu_u)
        clv = _Clv(v, l_v, mu_v)
        cs = _Cs(s, l_i, mu_s)
        bigU = np.diag(u)
        bigV = np.diag(v)
        bigS = np.diag(s)
        bigLU = np.diag(l_u)
        bigLV = np.diag(l_v)
        bigLI = np.diag(l_i)

        rhs = np.concatenate([cd, ce, ci, cu, cv, clu, clv, cs])
        # fmt: off
        lhs: FArr = sp.block_array(
            [
                [1 / eta * hess, -A_mat, -A_ineq, None, None, None, None, None],
                [A_mat.T, delta * np.eye(k), None, np.eye(k), -np.eye(k), None, None, None],
                [A_ineq.T, None, None, None, None, None, None, -np.eye(2 * n)],
                [None, -np.eye(k), None, None, None, -np.eye(k), None, None],
                [None, np.eye(k), None, None, None, None, -np.eye(k), None],
                [None, None, None, bigLU, None, bigU, None, None],
                [None, None, None, None, bigLV, None, bigV, None],
                [None, None, bigS, None, None, None, None, bigLI],
            ]
        ).toarray()
        # fmt: on

        step_size = np.linalg.solve(lhs, -rhs)
        lengths = (n, k, 2 * n, k, k, k, k, 2 * n, None)
        names = ("pr", "p_l_e", "p_l_i", "pu", "pv", "p_l_u", "p_l_v", "ps")
        steps: dict[str, FArr] = {}
        for i in range(len(names)):
            res = step_size[0 : lengths[i]]
            step_size = step_size[lengths[i] :]
            steps.update({names[i]: res})

        h_tilde = 1 / eta * hess + A_ineq @ np.diag(l_i / s) @ A_ineq.T

        p_l_e = -np.linalg.solve(
            A_mat.T @ np.linalg.inv(h_tilde) @ A_mat
            + delta * np.eye(k)
            + np.diag(u / l_u + v / l_v),
            ce
            + v / l_v * (cv + 1 / v * clv)
            - u / l_u * (cu + 1 / u * clu)
            - (A_mat.T @ np.linalg.inv(h_tilde)).dot(
                cd + A_ineq.dot(l_i / s * (ci + 1 / l_i * cs))
            ),
        )
        steps.update({"p_l_e": p_l_e})

        r_step: FArr = np.linalg.solve(
            h_tilde,
            A_mat.dot(steps["p_l_e"]) - cd - A_ineq.dot((ci * l_i + cs) / s),
        )
        steps.update({"pr": r_step})

        l_i_step = l_i / s * (-A_ineq.T.dot(r_step) - ci - 1 / l_i * cs)

        u_step = (u / l_u) * (p_l_e - (cu + 1 / u * clu))
        v_step = v / l_v * (-p_l_e - (cv + 1 / v * clv))

        s_step = -s - (1 / l_i) * s * l_i_step + mu_s * (1 / l_i)
        l_u_step = 1 / u * (-clu - l_u * u_step)
        l_v_step = 1 / v * (-clv - l_v * v_step)

        steps.update({"p_l_u": l_u_step, "p_l_v": l_v_step, "p_l_i": l_i_step})

        steps.update({"pu": u_step, "pv": v_step, "ps": s_step})
        return steps

    mu_s = 0.02
    mu_u = 0.02
    mu_v = 0.02

    r0 = np.ones(n)
    u0 = np.zeros(k) + 0.01
    v0 = np.zeros(k) + 0.01
    s0 = A_ineq.T.dot(r0) - bounds
    l_e = np.zeros(k)
    l_u = np.zeros(k) + 0.05
    l_v = np.zeros(k) + 0.05
    l_i = np.zeros(2 * n) + 0.05

    def error_fn(
        r: FArr,
        u: FArr,
        v: FArr,
        s: FArr,
        l_e: FArr,
        l_i: FArr,
        l_u: FArr,
        l_v: FArr,
        mu_s: float,
        mu_u: float,
        mu_v: float,
        eta: float,
    ) -> tuple[np.floating, ...]:
        error_d = np.linalg.norm(_Cd(r, s, l_e, l_i, eta))
        error_s = np.linalg.norm(_Cs(s, l_i, mu_s))
        error_e = np.linalg.norm(_Ce(r, u, v))
        error_i = np.linalg.norm(_CI(r, s))
        error_u = np.linalg.norm(_Cu(l_e, l_u, 1.0))
        error_v = np.linalg.norm(_Cv(l_e, l_v, 1.0))
        error_lu = np.linalg.norm(_Clu(u, l_u, mu_u))
        error_lv = np.linalg.norm(_Clv(v, l_v, mu_v))
        return error_d, error_s, error_e, error_i, error_u, error_v, error_lu, error_lv

    i = 1
    while (
        np.linalg.norm(
            error_fn(r0, u0, v0, s0, l_e, l_i, l_u, l_v, mu_s, mu_u, mu_v, eta)
        )
        > 1e-5
    ):
        steps = _calculate_step_sizes_fullmat(
            r0, u0, v0, s0, l_e, l_i, l_u, l_v, mu_s, mu_u, mu_v, eta
        )

        steps = steps
        ps = steps["ps"]
        pi = steps["p_l_i"]
        pu = steps["pu"]
        p_l_u = steps["p_l_u"]
        p_l_v = steps["p_l_v"]
        pv = steps["pv"]
        largest_slack_step = float(
            np.min(-0.995 * s0[ps < 0] / ps[ps < 0], initial=1.0)
        )
        backtrack_slack = min(1.0, largest_slack_step)

        largest_dual_step = float(
            np.min(-0.995 * l_i[pi < 0] / pi[pi < 0], initial=1.0)
        )
        backtrack_dual = min(1.0, largest_dual_step)

        largest_u_step = float(np.min(-0.995 * u0[pu < 0] / pu[pu < 0], initial=np.inf))
        largest_v_step = float(np.min(-0.995 * v0[pv < 0] / pv[pv < 0], initial=np.inf))
        backtrack_uv = min(1.0, largest_u_step, largest_v_step)

        largest_dual_u_step = float(
            np.min(-0.995 * l_u[p_l_u < 0] / p_l_u[p_l_u < 0], initial=1.0)
        )
        largest_dual_v_step = float(
            np.min(-0.995 * l_v[p_l_v < 0] / p_l_v[p_l_v < 0], initial=1.0)
        )
        backtrack_l_uv = min(1.0, largest_dual_u_step, largest_dual_v_step)

        merit0 = merit(
            r0, u0, v0, s0, eta, mu_s, mu_u, mu_v, _Ce(r0, u0, v0), _CI(r0, s0)
        )
        gbp = 1.0
        for _ in range(20):
            r1 = r0 + steps["pr"] * backtrack_slack * gbp
            s1 = s0 + steps["ps"] * backtrack_slack * gbp
            u1 = u0 + steps["pu"] * backtrack_uv * gbp
            v1 = v0 + steps["pv"] * backtrack_uv * gbp
            merit1 = merit(
                r1, u1, v1, s1, eta, mu_s, mu_u, mu_v, _Ce(r1, u1, v1), _CI(r1, s1)
            )
            if merit1 < merit0:
                break
            gbp = gbp * 0.8

        r0 += steps["pr"] * backtrack_slack * gbp
        s0 += steps["ps"] * backtrack_slack * gbp
        u0 += steps["pu"] * backtrack_uv * gbp
        v0 += steps["pv"] * backtrack_uv * gbp
        l_u += steps["p_l_u"] * backtrack_l_uv * gbp
        l_v += steps["p_l_v"] * backtrack_l_uv * gbp
        l_e += steps["p_l_e"] * backtrack_dual * gbp
        l_i += steps["p_l_i"] * backtrack_dual * gbp

        zeta = np.min(s0 * l_i) / np.mean(s0 * l_i)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_s = sigma * np.mean(s0 * l_i)

        zeta = np.min(u0 * l_u) / np.mean(u0 * l_u)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_u = sigma * np.mean(u0 * l_u)

        zeta = np.min(v0 * l_v) / np.mean(v0 * l_v)
        sigma = 0.1 * min(0.05 * (1 - zeta) / zeta, 2) ** 3
        mu_v = sigma * np.mean(v0 * l_v)

        if eta < max(np.max(np.abs(l_e)), np.max(l_i), np.max(l_u), np.max(l_v)):
            eta = 10 * eta

        i += 1

    return r0, u0 - v0


def merit(
    r: FArr,
    u: FArr,
    v: FArr,
    s: FArr,
    eta: float,
    mu_s: float,
    mu_u: float,
    mu_v: float,
    Ce: FArr,
    Ci: FArr,
) -> FArr:
    """Evaluate whether a step decrases the objective."""
    out: FArr = 1 / eta * np.sum(r * np.log(r) - r + 1) + np.sum(u + v)
    out += (
        -mu_s * np.sum(np.log(s)) - mu_u * np.sum(np.log(u)) - mu_v * np.sum(np.log(v))
    )
    out += np.sum(np.abs(Ce))
    out += np.sum(np.abs(Ci))
    return out


def test_comparing_full_and_partial() -> None:
    """Test the same output is generated by full matrix factorization + elimination."""
    r = ebw_elastic(n=15, seed=11313, bounds=(0.85, 1.15), eta=2.0)
    rfull, u_minus_v = elastic_mode_full(n=15, seed=11313, bounds=(0.85, 1.15), eta=2.0)
    assert np.corrcoef(r, rfull)[0, 1] > 0.999
    assert np.linalg.norm(r - rfull, 2) <= 1e-3
