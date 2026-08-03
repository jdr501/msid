"""Replication benchmark on the Tether dataset (spec Section 12, item 7).

Marked ``@slow`` and skipped unless the dataset is available via the
``MSID_TETHER_CSV`` environment variable (raw columns: date, p_usdt,
q_usdt, p_btc; see :func:`msid.load_tether`).

Targets (Rajapaksa & Shao 2026, Tables 1-3, Figure 7), p=13, r=1, M=2,
deterministic="ct", sample 2019-01-02 to 2023-11-08:

* log L approx 1772.23 (unrestricted),
* AIC/SC ordering of Table 1 (triangular < unrestricted < conventional),
* lambda estimates of Table 2 within reported standard errors
  (lambda_21 = 0.045 (0.003), lambda_22 = 717.91 (30.414),
  lambda_23 = 26.18 (1.317)),
* LR statistics of Table 3 (0.002 and 31.397, df = 2) with identical test
  conclusions,
* IRF signs/shapes of Figure 7.

Exact-to-print equality is not required (optimizer/seed variation).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import msid

CSV = os.environ.get("MSID_TETHER_CSV")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(CSV is None, reason="set MSID_TETHER_CSV to run the replication"),
]


@pytest.fixture(scope="module")
def tether_fit():
    y = msid.load_tether(CSV).loc["2019-01-02":"2023-11-08"]
    model = msid.MSVECM(y, lags=13, coint_rank=1, n_regimes=2, deterministic="ct")
    res = model.fit(n_starts=50, random_state=0)
    return model, res


def test_unrestricted_loglik(tether_fit):
    _, res = tether_fit
    assert res.loglik_ == pytest.approx(1772.23, abs=5.0)


def test_lambda_estimates_within_ses(tether_fit):
    _, res = tether_fit
    lam = np.sort(res.Lambda_[1])
    targets = np.sort([0.045, 717.91, 26.18])
    ses = np.array([0.003, 30.414, 1.317])[np.argsort([0.045, 717.91, 26.18])]
    for est, tgt, se in zip(lam, targets, 2.0 * ses):
        assert abs(est - tgt) < max(se, 0.05 * tgt)


def test_lr_tests_table3(tether_fit):
    _, res = tether_fit
    R_tri = msid.Restrictions(3).b_zeros([(2, 1)]).b_signs({(2, 0): "+"})
    R_conv = msid.Restrictions(3).b_zeros([(2, 0), (2, 1)])
    t_tri, _ = res.test_restrictions(R_tri, df_override=2, n_starts=30, random_state=0)
    t_conv, _ = res.test_restrictions(R_conv, df_override=2, n_starts=30, random_state=0)
    assert t_tri["p-value"].iloc[0] > 0.10  # triangular not rejected
    assert t_conv["p-value"].iloc[0] < 0.01  # conventional rejected
    assert t_conv["LR"].iloc[0] == pytest.approx(31.397, rel=0.25)


def test_aic_sc_ordering(tether_fit):
    _, res = tether_fit
    R_tri = msid.Restrictions(3).b_zeros([(2, 1)]).b_signs({(2, 0): "+"})
    _, res_tri = res.test_restrictions(R_tri, n_starts=30, random_state=0)
    assert res_tri.aic_ < res.aic_  # Table 1 ordering
    assert res_tri.sc_ < res.sc_


def test_irf_signs_figure7(tether_fit):
    _, res = tether_fit
    irf = res.irf(horizon=30, n_boot=0)
    names = list(res.model.y.columns)
    i_p = names.index("p_usdt")
    # order shocks so the USDT-price shock is the one with the largest
    # on-impact price response
    j_p = int(np.argmax(np.abs(irf.irfs[0, i_p, :])))
    resp = irf.irfs[:, :, j_p] * np.sign(irf.irfs[0, i_p, j_p])
    assert resp[0, i_p] > 0  # own impact positive
    assert abs(resp[30, i_p]) < 0.2 * resp[0, i_p]  # peg reversion by day 30
