"""Bootstrap null distribution for LR tests of restrictions on B."""

from __future__ import annotations

import numpy as np
import pytest

import msid
from msid.inference.boot_lr import polish_from


@pytest.fixture(scope="module")
def restricted_pair(fitted_example):
    """The shared K=3 fit plus a b32 = 0 restricted counterpart."""
    R = msid.Restrictions(3).b_zeros([(2, 1)])
    return fitted_example, fitted_example.model.fit(
        restrictions=R, n_starts=3, n_jobs=1, random_state=0
    )


@pytest.fixture(scope="module")
def boot_out(restricted_pair):
    """One bootstrap run shared across the assertions below -- it is the
    expensive step, and re-running it per test dominated the suite."""
    un, res = restricted_pair
    return msid.bootstrap_lr_test(un, res, n_boot=12, max_iter=40, n_jobs=1, random_state=0)


def test_result_fields_and_ranges(boot_out):
    out = boot_out
    assert out.df == 1
    assert out.n_ok <= 12
    assert 0.0 < out.p_value <= 1.0
    assert 0.0 <= out.p_chi2 <= 1.0
    assert out.p_value >= 1.0 / (out.n_ok + 1.0)  # the attainable floor
    assert set(out.crit) == {90, 95, 99}
    assert out.crit[90] <= out.crit[95] <= out.crit[99]
    assert out.draws is not None and out.draws.size == out.n_ok
    assert isinstance(out.rejected, bool)
    assert "b32=0" in out.h0
    assert "Bootstrap LR test" in str(out)


def test_polish_makes_lr_nonnegative(boot_out):
    """A restricted fit can out-run its parent; polishing must repair that."""
    out = boot_out
    assert out.statistic >= -1e-8
    assert out.loglik_unrestricted >= out.loglik_restricted - 1e-8
    assert out.polish_gain >= 0.0


def test_draws_are_nonnegative(boot_out):
    """Warm-starting both fits at the null makes every draw well signed."""
    assert np.all(boot_out.draws >= -1e-8)


def test_null_dgp_satisfies_the_restriction(restricted_pair):
    """The simulated data must actually impose H0.

    Flipping the reduced-form residuals would leave the restricted element in
    the sample second moments, and the bootstrap LR would then reproduce the
    observed one instead of the null.  Rotating through B0 is what prevents it.
    """
    _, res = restricted_pair
    DY, Z = res._DY, res._Z
    B0 = res.B_
    fitted0 = Z @ res.theta_.T
    eps0 = (DY - fitted0) @ np.linalg.inv(B0).T

    rng = np.random.default_rng(0)
    psi = rng.integers(0, 2, size=eps0.shape) * 2.0 - 1.0
    u_star = (psi * eps0) @ B0.T

    # Cov(u*) = B0 diag(.) B0' up to sampling error, so the implied structural
    # shocks stay orthogonal and the b32 = 0 column structure is preserved.
    eps_star = u_star @ np.linalg.inv(B0).T
    C = np.corrcoef(eps_star.T)
    off = C[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 0.15
    assert abs(B0[2, 1]) < 1e-10


def test_reproducible_under_a_fixed_seed(restricted_pair):
    un, res = restricted_pair
    kw = dict(n_boot=10, max_iter=30, n_jobs=1, random_state=7)
    a = msid.bootstrap_lr_test(un, res, **kw)
    b = msid.bootstrap_lr_test(un, res, **kw)
    assert a.p_value == b.p_value
    np.testing.assert_allclose(a.draws, b.draws)


def test_nesting_is_validated(fitted_example):
    """H0 must impose a superset of H1's zeros, and must add at least one."""
    R_a = msid.Restrictions(3).b_zeros([(2, 1)])
    R_b = msid.Restrictions(3).b_zeros([(2, 0)])
    fit_a = fitted_example.model.fit(restrictions=R_a, n_starts=2, n_jobs=1, random_state=0)
    fit_b = fitted_example.model.fit(restrictions=R_b, n_starts=2, n_jobs=1, random_state=0)

    with pytest.raises(ValueError, match="superset|every zero"):
        msid.bootstrap_lr_test(fit_a, fit_b, n_boot=12, max_iter=40, n_jobs=1)
    with pytest.raises(ValueError, match="nothing to test"):
        msid.bootstrap_lr_test(fit_a, fit_a, n_boot=12, max_iter=40, n_jobs=1)


def test_df_override_moves_only_the_chi2_column(restricted_pair):
    un, res = restricted_pair
    kw = dict(n_boot=10, max_iter=30, n_jobs=1, random_state=0)
    a = msid.bootstrap_lr_test(un, res, **kw)
    b = msid.bootstrap_lr_test(un, res, df_override=2, **kw)
    assert b.df == 2
    assert b.p_value == a.p_value
    assert b.p_chi2 != a.p_chi2


def test_polish_from_never_worsens(restricted_pair):
    un, res = restricted_pair
    ll, state = polish_from(un, res, max_iter=40)
    assert ll >= un.loglik_ - 1e-10
    assert state is None or np.isfinite(state.loglik)


def test_results_method_returns_pair(fitted_example):
    R = msid.Restrictions(3).b_zeros([(2, 1)])
    out, restricted = fitted_example.test_restrictions_bootstrap(
        R, n_boot=10, n_starts=2, max_iter=30, n_jobs=1, random_state=0
    )
    assert isinstance(out, msid.BootLRResult)
    assert restricted.restrictions.b_zero_indices == [(2, 1)]
    assert abs(restricted.B_[2, 1]) < 1e-10


@pytest.mark.slow
def test_size_is_reasonable_under_the_null(restricted_pair):
    """With enough draws the bootstrap 95% critical value should be finite and
    in a plausible neighbourhood of the chi-squared one."""
    un, res = restricted_pair
    out = msid.bootstrap_lr_test(un, res, n_boot=199, max_iter=200, n_jobs=-1, random_state=1)
    assert np.isfinite(out.crit[95])
    assert 0.1 < out.size_ratio < 20.0
