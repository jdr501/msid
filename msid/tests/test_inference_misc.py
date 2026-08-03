"""Identification diagnostics, stability tests, pretests, plotting smoke."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import msid
from msid.datasets.simulate import default_msvecm_params, simulate_msvecm


def test_identification_block(fitted_example):
    ident = fitted_example.identification_
    assert len(ident.table) == 3  # K=3 pairs for M=2
    assert ident.identified in (True, False)
    assert "Identification" in str(ident) or "WARNING" in ident.verdict


def test_summary_contains_required_blocks(fitted_example):
    txt = fitted_example.summary(print_output=False)
    for token in ("Identification status", "Lambda_2", "Transition matrix", "log L"):
        assert token in txt


def test_invariance_tests_require_m3(fitted_example):
    with pytest.raises(ValueError, match="M >= 3"):
        fitted_example.test_b_invariance()
    with pytest.raises(ValueError, match="M >= 3"):
        fitted_example.test_overidentification(n_boot=10)


def test_m3_invariance_and_jtest():
    pars = default_msvecm_params(M=3)
    y, _ = simulate_msvecm(
        T=800,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=pars["Lambdas"],
        P=pars["P"],
        random_state=7,
    )
    m = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=3)
    res = m.fit(n_starts=3, n_jobs=1, random_state=0, max_iter=250)
    inv = res.test_b_invariance()
    assert inv.df == int(3 * 3 * 4 / 2 - 9 - 2 * 3)  # = 3
    assert 0.0 <= inv.p_value <= 1.0
    j = res.test_overidentification(n_boot=100, n_jobs=1, random_state=2)
    assert j.q == 3
    # the data were generated with a common B: the test should not reject
    assert j.p_value > 0.01


def test_b_stability_guard(fitted_example):
    # absurdly high threshold: no window can have enough regime observations
    with pytest.raises(ValueError, match="not identified within the window"):
        fitted_example.test_b_stability(
            "2020-05-15",
            window_before=(2, 1),
            window_after=(1, 2),
            n_boot=5,
            min_regime_obs=10_000,
            n_jobs=1,
        )


def test_b_stability_runs(fitted_example):
    bs = fitted_example.test_b_stability(
        "2020-05-15",
        window_before=(8, 3),
        window_after=(3, 8),
        n_boot=25,
        min_regime_obs=10,
        n_jobs=1,
        random_state=9,
    )
    assert np.isfinite(bs.statistic)
    assert 0.0 <= bs.p_value <= 1.0
    assert bs.B1.shape == (3, 3)


def test_rolling_stability_and_plots(fitted_example):
    roll = fitted_example.rolling_stability(window=300, lags=2)
    assert len(roll.paths) > 0
    key = next(iter(roll.paths))
    df = roll.paths[key]
    assert {"coef", "lo", "hi"} <= set(df.columns)
    figs = roll.plot()
    assert len(figs) == 3
    fig = fitted_example.plot_regimes(threshold=0.7)
    assert fig is not None


def test_fevd_shares_sum_to_one(fitted_example):
    tab = fitted_example.fevd(horizon=6)
    shares = tab[[c for c in tab.columns if c.startswith("shock")]].to_numpy()
    assert np.allclose(shares.sum(axis=1), 1.0, atol=1e-8)
    assert (shares >= -1e-12).all()


def test_johansen_and_lag_selection(example_data):
    res = msid.johansen(example_data, det_order=1, k_ar_diff=1)
    assert len(res.table) == 6  # K=3: trace + max-eig rows
    assert res.trace_stat[0] > res.trace_crit[0, 1]  # cointegration found
    tab = msid.select_lags(example_data, max_lags=4)
    assert set(tab.columns) == {"AIC", "SC", "HQ"}
    assert tab.attrs["selected"]["SC"] in range(1, 5)


def test_justify_ms(example_data):
    out = msid.justify_ms(example_data, lags=1, print_output=False)
    assert out["arch_lm"]["p-value"] < 0.05  # MS data => ARCH detected
    assert "warranted" in out["recommendation"]


def test_loglik_starts_stored(fitted_example):
    assert fitted_example.loglik_starts_.shape == (3,)
    assert np.isfinite(fitted_example.loglik_starts_).any()


def test_structural_shocks_whitened(fitted_example):
    """In regime-1 periods the structural shocks should have roughly unit
    variance and weak cross-correlation."""
    eps = fitted_example.structural_shocks_.to_numpy()
    probs = fitted_example.smoothed_probs_.to_numpy()
    mask = probs[:, 0] > 0.9
    S = np.cov(eps[mask].T)
    assert np.allclose(np.diag(S), 1.0, atol=0.35)
    off = S[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 0.3
