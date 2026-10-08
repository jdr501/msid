"""Tests for the overlapping-window constant-B test (Tether Appendix F.2)."""

from __future__ import annotations

import numpy as np
import pytest

from msid.inference.stability import (
    _pick,
    _wald,
    _weighted_sigmas,
    _wild_draw,
    b_stability_test,
    b_stability_test_fixed_probs,
)


def _synthetic(mu, n=400, sd=0.01, seed=0):
    """Null draws whose window difference has mean mu, and an observed B at mu."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    D2 = np.repeat(base[None], n, axis=0)
    D1 = D2 + mu + rng.normal(scale=sd, size=(n, 3, 3))
    return base + mu, base, D1, D2


def test_result_at_the_null_mean_does_not_reject():
    """The bug this guards: observed centred at 0, bootstrap centred at its mean.

    When the window estimator is biased under the null, the observed difference
    sits at that bias.  Measured from the null mean it is unremarkable.
    """
    mu = np.full((3, 3), 0.5)  # a large null bias
    B1, B2, D1, D2 = _synthetic(mu)
    W, p, df = _wald(B1, B2, D1, D2, elements=None, relative=False)
    assert df == 9
    assert p > 0.2, f"a result at the null mean rejected: W={W:.2f}, p={p:.4f}"

    # the old statistic, observed uncentred against centred draws, explodes
    d = (B1 - B2).ravel(order="F")
    D = np.array([(a - b).ravel(order="F") for a, b in zip(D1, D2)])
    iV = np.linalg.pinv(np.cov(D.T, ddof=1))
    assert d @ iV @ d > 100 * W


def test_p_value_is_never_exactly_zero():
    B1, B2, D1, D2 = _synthetic(np.zeros((3, 3)))
    B1 = B1 + 10.0  # an absurd observed gap
    _, p, _ = _wald(B1, B2, D1, D2, elements=None, relative=False)
    assert p == pytest.approx(1.0 / (len(D1) + 1))


def test_relative_removes_the_scale_of_each_shock():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    rescaled = B @ np.diag([0.4, 2.5, 7.0])  # same directions, new scales
    assert np.allclose(_pick(B, None, True), _pick(rescaled, None, True))
    assert not np.allclose(_pick(B, None, False), _pick(rescaled, None, False))


def test_relative_refuses_diagonal_elements():
    with pytest.raises(ValueError, match="off-diagonal"):
        _pick(np.eye(3), [(0, 0)], relative=True)


def test_subset_picks_the_named_elements():
    B = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(_pick(B, [(2, 0), (2, 1)], False), [6.0, 7.0])


def test_unknown_null_is_refused(fitted_example):
    with pytest.raises(ValueError, match="null must be"):
        b_stability_test(fitted_example, "2020-05-01", null="bogus")


@pytest.mark.parametrize("null", ["wild", "gaussian"])
def test_end_to_end_and_subtest_reproduces_headline(fitted_example, null):
    res = b_stability_test(
        fitted_example,
        "2020-05-01",
        n_boot=30,
        min_regime_obs=5,
        null=null,
        n_jobs=1,
        random_state=0,
    )
    assert res.null == null
    assert res.draws1.shape == (res.n_boot, 3, 3)
    assert 1.0 / (res.n_boot + 1) <= res.p_value <= 1.0
    W, p, df = res.subtest()
    assert (W, p, df) == pytest.approx((res.statistic, res.p_value, 9))
    W31, _, df31 = res.subtest(elements=[(2, 0), (2, 1)])
    assert df31 == 2 and np.isfinite(W31)


def test_wild_draw_flips_one_sign_per_date():
    """Every date is multiplied by a single +1/-1, so the K shocks move together."""
    rng = np.random.default_rng(0)
    T, K = 50, 3
    E = rng.normal(size=(T, K))
    B0 = np.eye(K) + 0.1 * rng.normal(size=(K, K))
    fitted = rng.normal(size=(T, K))
    DYb = _wild_draw(fitted, E, B0, np.random.default_rng(1))
    Eb = (DYb - fitted) @ np.linalg.inv(B0).T
    ratio = Eb / E
    assert np.allclose(np.abs(ratio), 1.0)  # magnitudes preserved
    assert np.allclose(ratio, ratio[:, [0]])  # one sign per date
    assert not np.allclose(ratio, ratio[0, 0])  # signs do vary across dates


def test_weighted_sigmas_recovers_the_regime_covariances():
    """With near-degenerate probabilities the weighted covariance is the group one."""
    rng = np.random.default_rng(0)
    n, K = 600, 3
    A = np.eye(K) + 0.2 * rng.normal(size=(K, K))
    U = np.vstack([rng.normal(size=(n, K)) @ A.T, 3.0 * rng.normal(size=(n, K)) @ A.T])
    w = np.zeros((2 * n, 2))
    w[:n, 0] = 1.0
    w[n:, 1] = 1.0
    S1, S2 = _weighted_sigmas(U, w)
    assert np.allclose(S1, (U[:n].T @ U[:n]) / n)
    assert np.allclose(S2, (U[n:].T @ U[n:]) / n)
    # the second group is about 9 times the first
    assert 6.0 < np.trace(S2) / np.trace(S1) < 13.0


def test_uncentred_wald_differs_from_centred_when_the_null_mean_is_nonzero():
    B1, B2, D1, D2 = _synthetic(0.0, n=200)
    # shift every draw AND the observed value by the same amount: the null mean
    # is then far from zero, which is exactly when the two conventions diverge
    D1 = D1 + 0.5
    B1 = B1 + 0.5
    Wc, pc, _ = _wald(B1, B2, D1, D2, None, False, center=True)
    Wu, pu, _ = _wald(B1, B2, D1, D2, None, False, center=False)
    assert Wu > Wc
    assert pc >= 1.0 / (len(D1) + 1) and pu >= 1.0 / (len(D1) + 1)


def test_fixed_probs_end_to_end(fitted_example):
    res = b_stability_test_fixed_probs(
        fitted_example,
        "2020-05-01",
        n_boot=50,
        min_regime_obs=5,
        n_jobs=1,
        random_state=0,
    )
    assert res.centered is False
    assert res.draws1.shape == (res.n_boot, 3, 3)
    assert 1.0 / (res.n_boot + 1) <= res.p_value <= 1.0
    W, p, df = res.subtest()
    assert (W, p, df) == pytest.approx((res.statistic, res.p_value, 9))
    W31, _, df31 = res.subtest(elements=[(2, 0), (2, 1)])
    assert df31 == 2 and np.isfinite(W31)


def test_fixed_probs_refuses_more_than_two_regimes(fitted_example):
    class _ThreeRegimes:
        def __init__(self, inner):
            self._inner = inner
            self.M = 3

        def __getattr__(self, name):
            return getattr(self._inner, name)

    with pytest.raises(ValueError, match="exactly 2 regimes"):
        b_stability_test_fixed_probs(_ThreeRegimes(fitted_example), "2020-05-01")


def test_draft_exact_uses_the_plain_p_value(fitted_example):
    """draft_exact=True reproduces the paper's arithmetic: no +1 in the p-value."""
    res = b_stability_test_fixed_probs(
        fitted_example,
        "2020-05-01",
        n_boot=50,
        min_regime_obs=5,
        draft_exact=True,
        n_jobs=1,
        random_state=0,
    )
    assert res.draft is True
    assert res.centered is False
    assert "draft arithmetic" in res.null
    # the plain form can reach exactly 0 and never exceeds 1
    assert 0.0 <= res.p_value <= 1.0
    # the p-value is a multiple of 1/n_boot, which the +1 form never is
    assert np.isclose(res.p_value * res.n_boot, round(res.p_value * res.n_boot))
    W, p, df = res.subtest()
    assert (W, p, df) == pytest.approx((res.statistic, res.p_value, 9))


def test_eigenvalue_column_order_matches_the_drafts_convention():
    """Columns ordered by descending lambda, signs fixed by the largest element."""
    from msid.inference.stability import _b_eig_sorted

    rng = np.random.default_rng(0)
    K = 3
    B = np.eye(K) + 0.3 * rng.normal(size=(K, K))
    lam = np.array([0.05, 40.0, 3.0])  # deliberately unsorted
    S1 = B @ B.T
    S2 = B @ np.diag(lam) @ B.T
    Bh = _b_eig_sorted(S1, S2)
    # it reproduces the covariances it was built from
    assert np.allclose(Bh @ Bh.T, S1)
    # and the implied lambdas come out in descending order
    iB = np.linalg.inv(Bh)
    lam_hat = np.diag(iB @ S2 @ iB.T)
    assert np.all(np.diff(lam_hat) < 0)
    assert np.allclose(np.sort(lam_hat)[::-1], np.sort(lam)[::-1])
    # every column's largest-magnitude element is positive
    for j in range(K):
        assert Bh[np.argmax(np.abs(Bh[:, j])), j] > 0


def test_fixed_probs_accepts_eigenvalue_order_and_refuses_junk(fitted_example):
    res = b_stability_test_fixed_probs(
        fitted_example,
        "2020-05-01",
        n_boot=50,
        min_regime_obs=5,
        draft_exact=True,
        column_order="eigenvalue",
        n_jobs=1,
        random_state=0,
    )
    assert "eigenvalue column order" in res.null
    assert 0.0 <= res.p_value <= 1.0
    with pytest.raises(ValueError, match="column_order"):
        b_stability_test_fixed_probs(fitted_example, "2020-05-01", column_order="nope")
