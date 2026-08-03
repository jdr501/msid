"""Simulation recovery tests (spec Section 12, items 1 and 3)."""

from __future__ import annotations

import numpy as np
import pytest

import msid
from msid.datasets.simulate import default_msvecm_params, simulate_msvecm

from .conftest import closest_signed_permutation

N_SEEDS = 20


@pytest.mark.parametrize("seed", range(N_SEEDS))
def test_msvecm_recovery(seed):
    """K=3, M=2 MSVECM: recover (B, Lambda2, P, beta) within Monte-Carlo
    tolerance; B compared via the closest signed permutation."""
    pars = default_msvecm_params()
    y, _ = simulate_msvecm(
        T=700,
        alpha=pars["alpha"],
        beta=pars["beta"],
        Gammas=pars["Gammas"],
        B=pars["B"],
        Lambdas=pars["Lambdas"],
        P=pars["P"],
        random_state=1000 + seed,
    )
    model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2)
    res = model.fit(n_starts=3, n_jobs=1, random_state=seed, max_iter=300)
    assert np.isfinite(res.loglik_)

    # beta (Phillips-normalized truth: (1, -0.5, -0.2)')
    beta_true = pars["beta"] / pars["beta"][0, 0]
    assert np.allclose(model.beta, beta_true, atol=0.08)

    B_aligned, perm, _ = closest_signed_permutation(res.B_, pars["B"])
    assert np.allclose(
        B_aligned, pars["B"], atol=0.35
    ), f"seed {seed}: B not recovered\n{B_aligned}\nvs\n{pars['B']}"
    lam_est = np.asarray(res.Lambda_[1])[perm]
    lam_true = pars["Lambdas"][1]
    assert np.allclose(
        np.log(lam_est), np.log(lam_true), atol=0.8
    ), f"seed {seed}: Lambda2 {lam_est} vs {lam_true}"
    # transition probabilities
    assert np.allclose(np.diag(res.P_), np.diag(pars["P"]), atol=0.12)


def test_beta_input_equivalence(example_data):
    """Fitting with beta= set to the Johansen estimate from a prior run
    must reproduce the beta=None fit to numerical tolerance (spec 12.3)."""
    m1 = msid.MSVECM(example_data, lags=1, coint_rank=1, n_regimes=2)
    r1 = m1.fit(n_starts=2, n_jobs=1, random_state=0)
    m2 = msid.MSVECM(example_data, lags=1, beta=m1.beta, n_regimes=2)
    assert m2.beta_source == "user"
    assert m2.coint_rank == 1
    r2 = m2.fit(n_starts=2, n_jobs=1, random_state=0)
    assert np.isclose(r1.loglik_, r2.loglik_, rtol=1e-6)
    assert np.allclose(r1.B_, r2.B_, atol=1e-3)


def test_beta_validation(example_data):
    with pytest.raises(ValueError):
        msid.MSVECM(example_data, lags=1, beta=np.ones((2, 1)))  # wrong rows
    with pytest.raises(ValueError):
        msid.MSVECM(example_data, lags=1, beta=np.ones((3, 2)))  # rank deficient
    bad = np.array([[1.0], [np.nan], [0.0]])
    with pytest.raises(ValueError):
        msid.MSVECM(example_data, lags=1, beta=bad)  # non-finite
    with pytest.raises(NotImplementedError):
        msid.MSVECM(example_data, lags=1, coint_rank=1, beta="known_partial")
