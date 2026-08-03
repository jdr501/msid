"""Temporal-stability diagnostics for the model coefficients and B.

* Rolling-window coefficient estimation and plots (Tether Appendix E):
  per-equation OLS coefficient paths with 95% bands over rolling windows.
* Overlapping-window Wald test for a constant B around a transition date
  (Tether Appendix F.2): W = Delta' V[Delta]^{-1} Delta with V from a
  parametric bootstrap under H0 of common B, drawing
  ``u*_t ~ N(0, sum_m p_{t|T}(m) Sigma_m)`` at each date.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

__all__ = ["BStabilityResult", "RollingStabilityResult", "b_stability_test", "rolling_stability"]


@dataclass
class RollingStabilityResult:
    """Coefficient paths: dict ``(equation, regressor) -> DataFrame``.

    Each DataFrame has columns ``coef``, ``lo``, ``hi`` indexed by window-end
    date.  ``plot()`` renders the Tether Figures 9-11 style fans.
    """

    paths: dict
    window: int
    lags: int
    var_names: list

    def plot(self, equation=None, **kwargs):
        from ..plotting import plot_rolling

        return plot_rolling(self, equation=equation, **kwargs)


def rolling_stability(results, window: int = 400, lags: int = 3) -> RollingStabilityResult:
    """Rolling-window OLS estimates of the (V)ECM slope coefficients.

    Refits the reduced-form model equation by equation with ``lags`` lagged
    differences on windows of ``window`` observations (Tether App. E; the
    paper used 3 lags for legibility versus 13 in the main model).  For the
    MSVECM, beta is held at the full-sample estimate.
    """
    model = results.model
    y = model.y
    K = model.K
    names = list(y.columns)
    is_vecm = hasattr(model, "beta")

    # rebuild a small design with the requested lag order
    cls = type(model)
    if is_vecm:
        sub = cls(
            y,
            lags=lags,
            coint_rank=model.coint_rank,
            beta=model.beta,
            deterministic=model.deterministic,
            n_regimes=model.M,
        )
        prefix = "D"
    else:
        sub = cls(y, lags=lags, deterministic=model.deterministic, n_regimes=model.M)
        prefix = ""
    DY, Z, index = sub._build_design()
    T, q = Z.shape
    if T < window + 1:
        raise ValueError(f"sample ({T}) shorter than rolling window ({window})")

    d = sub.n_det
    reg_names = ["const", "trend"][:d]
    if is_vecm:
        reg_names += [f"ect{j + 1}" for j in range(sub.coint_rank)]
    for i in range(1, lags + 1):
        reg_names += [f"{prefix}{nm}(t-{i})" for nm in names]
    reg_names += [f"exog{j}" for j in range(q - len(reg_names))]

    n_win = T - window + 1
    coefs = np.empty((n_win, K, q))
    ses = np.empty((n_win, K, q))
    for w in range(n_win):
        Zw, Yw = Z[w : w + window], DY[w : w + window]
        XtX_inv = np.linalg.pinv(Zw.T @ Zw)
        beta_hat = XtX_inv @ Zw.T @ Yw  # (q, K)
        resid = Yw - Zw @ beta_hat
        dof = max(window - q, 1)
        s2 = (resid**2).sum(axis=0) / dof  # per equation
        se = np.sqrt(np.outer(np.diag(XtX_inv), s2))  # (q, K)
        coefs[w] = beta_hat.T
        ses[w] = se.T
    dates = index[window - 1 :]
    paths = {}
    for i, eq in enumerate(names):
        for j, rn in enumerate(reg_names):
            df = pd.DataFrame(
                {
                    "coef": coefs[:, i, j],
                    "lo": coefs[:, i, j] - 1.96 * ses[:, i, j],
                    "hi": coefs[:, i, j] + 1.96 * ses[:, i, j],
                },
                index=dates,
            )
            paths[(eq, rn)] = df
    return RollingStabilityResult(paths=paths, window=window, lags=lags, var_names=names)


@dataclass
class BStabilityResult:
    statistic: float
    p_value: float
    n_boot: int
    B1: np.ndarray
    B2: np.ndarray
    windows: tuple

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Overlapping-window Wald test for constant B: W = {self.statistic:.2f}, "
            f"bootstrap p = {self.p_value:.2f} ({self.n_boot} reps)"
        )


def _window_slice(index: pd.Index, center, months_before: int, months_after: int):
    center = pd.Timestamp(center)
    lo = center - pd.DateOffset(months=months_before)
    hi = center + pd.DateOffset(months=months_after)
    mask = (index >= lo) & (index <= hi)
    return np.where(mask)[0]


def _estimate_b_window(results, rows: np.ndarray, DY, Z) -> np.ndarray:
    """Re-estimate the MS variance model on a window (warm start) and
    return B aligned (sign/permutation) with the full-sample estimate."""
    from ..estimation.em import EMConfig, EMState, run_em

    model, R = results.model, results.restrictions
    state = EMState(
        Theta=results.theta_.copy(),
        B=results.B_.copy(),
        lams=[l.copy() for l in results.Lambda_[1:]],
        P=results.P_.copy(),
        xi0=results._xi0.copy(),
    )
    config = EMConfig(
        max_iter=200, tol_ll=1e-6, tol_param=1e-4, struct_maxiter=50, label_order="lambda_sort"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = run_em(DY[rows], Z[rows], R, model.M, state, config)
    return _align_b(st.B, results.B_)


def _align_b(B: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Closest signed column permutation of B to the reference matrix."""
    from itertools import permutations

    K = B.shape[0]
    best, best_err = B, np.inf
    for perm in permutations(range(K)):
        Bp = B[:, list(perm)]
        signs = np.sign(np.sum(Bp * ref, axis=0))
        signs[signs == 0] = 1.0
        Bc = Bp * signs
        err = np.linalg.norm(Bc - ref)
        if err < best_err:
            best, best_err = Bc, err
    return best


def b_stability_test(
    results,
    center_date,
    window_before: tuple[int, int] = (16, 5),
    window_after: tuple[int, int] = (5, 16),
    n_boot: int = 3000,
    min_regime_obs: int = 30,
    prob_threshold: float = 0.7,
    n_jobs: int = -1,
    random_state=None,
) -> BStabilityResult:
    """Overlapping-window Wald test for temporal stability of B (App. F.2).

    Window 1 spans ``window_before = (months before, months after)`` around
    ``center_date`` (paper default 16 before / 5 after); Window 2 spans
    ``window_after`` (5 before / 16 after).  Refuses to run if either window
    holds fewer than ``min_regime_obs`` high-probability observations from
    each regime, since B is not identified within such a window.
    """
    index = results.smoothed_probs_.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("b_stability_test needs a DatetimeIndex on the data")
    DY, Z = results._DY, results._Z
    probs = results.smoothed_probs_.to_numpy()
    rows1 = _window_slice(index, center_date, *window_before)
    rows2 = _window_slice(index, center_date, *window_after)
    for w, rows in (("window 1", rows1), ("window 2", rows2)):
        if rows.size == 0:
            raise ValueError(f"{w} contains no observations")
        counts = (probs[rows] > prob_threshold).sum(axis=0)
        if (counts < min_regime_obs).any():
            raise ValueError(
                f"{w} has regimes with fewer than {min_regime_obs} observations "
                f"at smoothed probability > {prob_threshold} (counts: {counts.tolist()}); "
                "B is not identified within the window"
            )

    B1 = _estimate_b_window(results, rows1, DY, Z)
    B2 = _estimate_b_window(results, rows2, DY, Z)
    delta = (B1 - B2).ravel(order="F")

    # parametric bootstrap under H0 of common B:
    # u*_t ~ N(0, sum_m p_{t|T}(m) Sigma_m)
    T = DY.shape[0]
    sigmas = results.Sigma_
    mix = np.einsum("tm,mij->tij", probs, np.stack(sigmas))
    chols = np.linalg.cholesky(mix)
    fitted = Z @ results.theta_.T
    ss = np.random.SeedSequence(random_state)
    children = ss.spawn(n_boot)

    def _one(child):
        rng = np.random.default_rng(child)
        eps = rng.standard_normal((T, results.K))
        Ub = np.einsum("tij,tj->ti", chols, eps)
        DYb = fitted + Ub
        try:
            B1b = _estimate_b_window(results, rows1, DYb, Z)
            B2b = _estimate_b_window(results, rows2, DYb, Z)
            return (B1b - B2b).ravel(order="F")
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return None

    deltas = Parallel(n_jobs=n_jobs)(delayed(_one)(c) for c in children)
    deltas = np.array([d for d in deltas if d is not None])
    if deltas.shape[0] < max(20, results.K**2 + 1):
        raise RuntimeError("too few successful bootstrap replications for V[Delta]")
    V = np.cov(deltas.T, ddof=1)
    iV = np.linalg.pinv(V)
    W = float(delta @ iV @ delta)
    dc = deltas - deltas.mean(axis=0)
    Wstars = np.einsum("bi,ij,bj->b", dc, iV, dc)
    p = float((Wstars >= W).mean())
    return BStabilityResult(
        statistic=W,
        p_value=p,
        n_boot=deltas.shape[0],
        B1=B1,
        B2=B2,
        windows=(window_before, window_after),
    )
