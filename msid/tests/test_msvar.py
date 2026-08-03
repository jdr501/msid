"""MSVAR path tests (spec Section 12, items 2 and 4)."""

from __future__ import annotations

import numpy as np
import pytest

import msid
from msid.datasets.simulate import default_msvecm_params, simulate_msvar, simulate_msvecm

from .conftest import closest_signed_permutation

A1 = [np.array([[0.5, 0.1], [0.0, 0.4]])]
B_TRUE = np.array([[1.0, 0.0], [0.4, 0.8]])
P2 = np.array([[0.95, 0.10], [0.05, 0.90]])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_msvar_m2_recovery(seed):
    lams = [np.ones(2), np.array([8.0, 0.2])]
    y, _ = simulate_msvar(700, A1, B_TRUE, lams, P2, random_state=100 + seed)
    model = msid.MSVAR(y, lags=1, n_regimes=2, deterministic="c")
    res = model.fit(n_starts=3, n_jobs=1, random_state=seed)
    B_aligned, perm, _ = closest_signed_permutation(res.B_, B_TRUE)
    assert np.allclose(B_aligned, B_TRUE, atol=0.3)
    lam = np.asarray(res.Lambda_[1])[perm]
    assert np.allclose(np.log(lam), np.log(lams[1]), atol=0.8)


def test_msvar_m3_runs():
    lams = [np.ones(2), np.array([8.0, 0.2]), np.array([0.3, 12.0])]
    P3 = np.array([[0.92, 0.05, 0.05], [0.04, 0.90, 0.05], [0.04, 0.05, 0.90]])
    y, _ = simulate_msvar(800, A1, B_TRUE, lams, P3, random_state=42)
    model = msid.MSVAR(y, lags=1, n_regimes=3, deterministic="c")
    res = model.fit(n_starts=3, n_jobs=1, random_state=0)
    assert np.isfinite(res.loglik_)
    assert res.P_.shape == (3, 3)
    assert np.allclose(res.P_.sum(axis=0), 1.0)


def test_msvar_longrun_and_irf():
    lams = [np.ones(2), np.array([6.0, 0.3])]
    y, _ = simulate_msvar(500, A1, B_TRUE, lams, P2, random_state=5)
    model = msid.MSVAR(y, lags=1, n_regimes=2, deterministic="c")
    res = model.fit(n_starts=2, n_jobs=1, random_state=0)
    Xi = res.Xi_
    assert Xi.shape == (2, 2)
    irf = res.irf(horizon=12, n_boot=0)
    # IRFs converge to zero for a stationary VAR
    assert np.abs(irf.irfs[-1]).max() < np.abs(irf.irfs[0]).max()


def test_msvar_nonstationary_xi_errors():
    """Xi must raise an informative error for a unit-root levels VAR."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal((300, 2)), axis=0)  # random walks
    model = msid.MSVAR(y, lags=1, n_regimes=2, deterministic="c")
    # a slope matrix with an exact unit root: A_1 = I
    Theta_unit = np.hstack([np.zeros((2, 1)), np.eye(2)])
    with pytest.raises(ValueError, match="not stationary"):
        model.longrun_map(Theta_unit)


def test_m4_smoke():
    """M=4 smoke test: estimation runs, identification block prints,
    invariance tests (6.1, 6.2) execute (spec 12.4)."""
    pars = default_msvecm_params(M=4)
    y, _ = simulate_msvecm(
        T=900,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=pars["Lambdas"],
        P=pars["P"],
        random_state=3,
    )
    model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=4)
    res = model.fit(n_starts=2, n_jobs=1, random_state=0, max_iter=150)
    assert np.isfinite(res.loglik_)
    assert len(res.Lambda_) == 4
    txt = res.summary(print_output=False)
    assert "Identification status" in txt
    inv = res.test_b_invariance()
    assert inv.df == int(4 * 3 * 4 / 2 - 9 - 3 * 3)
    j = res.test_overidentification(n_boot=50, n_jobs=1, random_state=1)
    assert j.q == (4 - 2) * 3 * (3 - 1) // 2
    assert 0.0 <= j.p_value <= 1.0


def test_n_regimes_validation(example_data):
    with pytest.raises(ValueError):
        msid.MSVECM(example_data, lags=1, coint_rank=1, n_regimes=5)
    with pytest.raises(ValueError):
        msid.MSVECM(example_data, lags=1, coint_rank=1, n_regimes=1)
