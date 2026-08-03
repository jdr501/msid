"""Bootstrap determinism and numerical-stability tests (spec 12.6, 12.8)."""

from __future__ import annotations

import numpy as np
import pytest

import msid
from msid.datasets.simulate import default_msvecm_params, simulate_msvecm
from msid.estimation.em import _log_state_densities, hamilton_filter


def test_bootstrap_determinism_and_nondegenerate(fitted_example):
    res = fitted_example
    irf1 = res.irf(horizon=6, n_boot=15, n_jobs=1, random_state=42)
    irf2 = res.irf(horizon=6, n_boot=15, n_jobs=1, random_state=42)
    assert np.allclose(irf1.lo, irf2.lo)
    assert np.allclose(irf1.hi, irf2.hi)
    width = irf1.hi - irf1.lo
    assert (width[1:] > 0).mean() > 0.9  # bands non-degenerate
    assert irf1.draws.shape[1:] == (7, 3, 3)  # full array returned


def test_logsumexp_filter_matches_naive():
    """Log-sum-exp filter agrees with the naive filter on easy data."""
    rng = np.random.default_rng(0)
    T, K = 200, 2
    U = rng.standard_normal((T, K))
    B = np.eye(K)
    lams = [np.array([2.0, 3.0])]
    P = np.array([[0.9, 0.2], [0.1, 0.8]])
    xi0 = np.array([0.5, 0.5])
    logeta = _log_state_densities(U, B, lams)
    xf, _, ll = hamilton_filter(logeta, P, xi0)

    # naive filter with raw densities
    eta = np.exp(logeta)
    xi = xi0.copy()
    ll_naive = 0.0
    for t in range(T):
        pred = P @ xi
        denom = pred @ eta[t]
        ll_naive += np.log(denom)
        xi = pred * eta[t] / denom
    assert np.isclose(ll, ll_naive, rtol=1e-10)
    assert np.allclose(xf[-1], xi)


def test_no_underflow_long_sample_extreme_lambdas():
    """5,000 observations with lambda = 700 (Tether Table 2 magnitudes):
    no NaN/underflow anywhere in the filter or the fit."""
    pars = default_msvecm_params()
    lams = [np.ones(3), np.array([700.0, 0.045, 26.0])]
    y, _ = simulate_msvecm(
        T=5000,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=lams,
        P=pars["P"],
        random_state=1,
    )
    model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2)
    res = model.fit(n_starts=2, n_jobs=1, random_state=0, max_iter=120)
    assert np.isfinite(res.loglik_)
    assert not np.isnan(res.smoothed_probs_.to_numpy()).any()
    assert not np.isnan(res.B_).any()
    lam = np.sort(res.Lambda_[1])
    assert lam[-1] > 100.0  # the extreme regime is found


def test_deterministic_options():
    pars = default_msvecm_params()
    y, _ = simulate_msvecm(
        T=400,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=pars["Lambdas"],
        P=pars["P"],
        random_state=2,
    )
    for det in ("n", "c", "ct"):
        m = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2, deterministic=det)
        res = m.fit(n_starts=1, n_jobs=1, random_state=0, max_iter=60)
        assert np.isfinite(res.loglik_)
    with pytest.raises(ValueError):
        msid.MSVECM(y, lags=1, coint_rank=1, deterministic="bogus")


def test_exog_hook():
    pars = default_msvecm_params()
    y, _ = simulate_msvecm(
        T=400,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=pars["Lambdas"],
        P=pars["P"],
        random_state=4,
    )
    exog = np.random.default_rng(0).standard_normal((len(y), 1))
    m = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2, exog=exog)
    _, Z, _ = m._build_design()
    m0 = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2)
    _, Z0, _ = m0._build_design()
    assert Z.shape[1] == Z0.shape[1] + 1
    res = m.fit(n_starts=1, n_jobs=1, random_state=0, max_iter=60)
    assert np.isfinite(res.loglik_)


def test_pickle_roundtrip(fitted_example, tmp_path):
    p = tmp_path / "res.pkl"
    fitted_example.to_pickle(str(p))
    back = msid.MSVECM.from_pickle(str(p))
    assert np.isclose(back.loglik_, fitted_example.loglik_)
    assert np.allclose(back.B_, fitted_example.B_)
