"""Fixed-design wild bootstrap for impulse-response confidence bands.

Exactly as in HL (2014, Section 3.2, Eq. 7) and the Tether paper:

1. Conditional on the ML estimates, ``Dy*_t = fitted_t + psi_t u_hat_t``
   with psi_t i.i.d. Rademacher (+-1 w.p. 0.5), regressors held at their
   original values (fixed design).
2. Per replication, theta* and B* are re-estimated by maximizing the
   likelihood **starting from the ML estimates** (warm start), holding
   Lambda_m and P fixed at their ML values -- the relative variances and
   transition probabilities are not resampled.
3. The original restriction set and sign normalization are enforced in
   every replication (columns of B* are aligned with the original B_hat to
   prevent band contamination from column flips), and the same label rule
   applies (labels cannot switch since Lambda and P are held fixed).
4. Percentile intervals, default 68% (16th/84th), 1000 replications;
   parallel via joblib with a spawned SeedSequence per replication.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed

__all__ = ["bootstrap_irf"]


def _one_replication(results, child, horizon, cumulate, max_iter):
    from ..estimation.em import EMConfig, EMState, run_em
    from ..irf import point_irf

    rng = np.random.default_rng(child)
    model, R = results.model, results.restrictions
    DY, Z = results._DY, results._Z
    U = results.residuals_.to_numpy()
    psi = rng.integers(0, 2, size=DY.shape[0]) * 2.0 - 1.0
    DYb = (DY - U) + psi[:, None] * U

    state = EMState(
        Theta=results.theta_.copy(),
        B=results.B_.copy(),
        lams=[l.copy() for l in results.Lambda_[1:]],
        P=results.P_.copy(),
        xi0=results._xi0.copy(),
    )
    # warm starts converge in few iterations; the full-precision EM tail is
    # not needed for percentile bands
    config = EMConfig(
        max_iter=max_iter, tol_ll=1e-6, tol_param=1e-4, struct_maxiter=50, label_order="lambda_sort"
    )
    C_map = model.longrun_map if R.has_xi_restrictions else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            st = run_em(DYb, Z, R, model.M, state, config, C_longrun=C_map, fix_lambda_P=True)
        except (np.linalg.LinAlgError, RuntimeError):
            return None
    if not np.isfinite(st.loglik):
        return None
    Bb = R.normalize_signs(st.B, reference=results.B_)
    try:
        return point_irf(model, st.Theta, Bb, horizon, cumulate=cumulate)
    except (ValueError, np.linalg.LinAlgError):
        return None


def bootstrap_irf(
    results,
    horizon: int = 30,
    n_boot: int = 1000,
    ci: float = 0.68,
    cumulate: str = "levels",
    n_jobs: int = -1,
    random_state=None,
    max_iter: int = 100,
):
    """Bootstrap IRF draws and percentile bands.

    Returns
    -------
    (draws, lo, hi) : draws is the full ``(n_ok, horizon+1, K, K)`` bootstrap
    IRF array (so users can compute other quantiles), lo/hi the percentile
    bands at level ``ci``.
    """
    ss = np.random.SeedSequence(random_state)
    children = ss.spawn(n_boot)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_one_replication)(results, c, horizon, cumulate, max_iter) for c in children
    )
    draws = np.array([d for d in out if d is not None])
    n_fail = n_boot - draws.shape[0]
    if n_fail:
        warnings.warn(
            f"{n_fail}/{n_boot} bootstrap replications failed and were dropped", UserWarning
        )
    if draws.shape[0] < 10:
        raise RuntimeError("too few successful bootstrap replications")
    a = (1.0 - ci) / 2.0
    lo = np.quantile(draws, a, axis=0)
    hi = np.quantile(draws, 1.0 - a, axis=0)
    return draws, lo, hi
