"""Temporal-stability diagnostics for the model coefficients and B.

* Rolling-window coefficient estimation and plots (Tether Appendix E):
  per-equation OLS coefficient paths with 95% bands over rolling windows.
* Overlapping-window Wald test for a constant B around a transition date
  (Tether Appendix F.2).  The null of a common B is imposed in the bootstrap
  DGP, and by default the DGP keeps each date's realized shock magnitude
  (``null="wild"``); see :func:`b_stability_test` for why that matters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

__all__ = [
    "BStabilityResult",
    "RollingStabilityResult",
    "b_stability_test",
    "b_stability_test_fixed_probs",
    "rolling_stability",
]


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
    """Overlapping-window test for a constant impact matrix B.

    ``statistic`` and ``p_value`` test ALL of vec(B).  The bootstrap
    replications are kept, so any subset of B, or the scale-free version, can
    be tested from the SAME draws with :meth:`subtest` at no extra cost.
    """

    statistic: float
    p_value: float
    n_boot: int
    B1: np.ndarray
    B2: np.ndarray
    windows: tuple
    null: str = "wild"
    centered: bool = True
    draws1: np.ndarray | None = field(default=None, repr=False)
    draws2: np.ndarray | None = field(default=None, repr=False)

    def subtest(self, elements=None, relative: bool = False):
        """Wald test on part of B, from the stored replications.

        Parameters
        ----------
        elements : list of (row, col) pairs, 0-based.  ``[(2, 0), (2, 1)]`` is
            (b31, b32).  ``None`` tests every element.
        relative : divide each column by its own diagonal element first.  That
            removes the SCALE of each shock and leaves only the DIRECTION of its
            impacts, so it tests off-diagonal elements only.

        Returns
        -------
        (W, p_value, df)
        """
        if self.draws1 is None:
            raise ValueError("this result was built without its bootstrap draws")
        return _wald(
            self.B1,
            self.B2,
            self.draws1,
            self.draws2,
            elements,
            relative,
            center=self.centered,
        )

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Overlapping-window Wald test for constant B: W = {self.statistic:.2f}, "
            f"bootstrap p = {self.p_value:.4f} ({self.n_boot} reps, {self.null} null)"
        )


def _pick(B: np.ndarray, elements, relative: bool) -> np.ndarray:
    """The elements of B the test compares, in column-major order."""
    K = B.shape[0]
    if relative:
        B = B / np.diag(B)[None, :]
        pairs = [(i, j) for j in range(K) for i in range(K) if i != j]
    else:
        pairs = [(i, j) for j in range(K) for i in range(K)]
    if elements is not None:
        pairs = [tuple(e) for e in elements]
        if relative and any(i == j for i, j in pairs):
            raise ValueError("relative=True compares off-diagonal elements only")
    return np.array([B[i, j] for i, j in pairs])


def _wald(B1, B2, D1, D2, elements, relative, center: bool = True):
    """Wald statistic and bootstrap p-value, both centred on the NULL mean.

    The bootstrap DGP imposes a common B, so the replications ARE the null
    distribution of the estimated window difference, including any bias the
    window estimator has when the two windows hold different regime mixes.
    Observed and bootstrap differences are therefore both measured from the
    null mean.  Centring only the bootstrap side, as an earlier version did,
    leaves that bias in the observed statistic alone and over-rejects.

    ``center=False`` measures BOTH sides from zero instead.  That is also a
    valid test -- the same transformation is applied to the observed value and
    to every draw -- and it is what the paper's original implementation does.
    It has less power when the null mean is far from zero, since the bias then
    inflates the observed statistic and the reference distribution alike.
    """
    d = _pick(B1, elements, relative) - _pick(B2, elements, relative)
    D = np.array(
        [_pick(a, elements, relative) - _pick(b, elements, relative) for a, b in zip(D1, D2)]
    )
    mu = D.mean(axis=0) if center else np.zeros(D.shape[1])
    iV = np.linalg.pinv(np.atleast_2d(np.cov(D.T, ddof=1)))
    W = float((d - mu) @ iV @ (d - mu))
    Dc = D - mu
    Ws = np.einsum("bi,ij,bj->b", Dc, iV, Dc)
    p = float((1 + (Ws >= W).sum()) / (len(Ws) + 1))
    return W, p, int(d.size)


def _window_slice(index: pd.Index, center, months_before: int, months_after: int):
    center = pd.Timestamp(center)
    lo = center - pd.DateOffset(months=months_before)
    hi = center + pd.DateOffset(months=months_after)
    mask = (index >= lo) & (index <= hi)
    return np.where(mask)[0]


def _wild_draw(fitted: np.ndarray, E: np.ndarray, B0: np.ndarray, rng) -> np.ndarray:
    """One wild-bootstrap sample of DY under a common B.

    ONE Rademacher sign per DATE, not per element.  A date's K structural
    shocks are flipped together, so whatever co-movement the realized shocks
    have at that date survives into the draw, and each date keeps its realized
    magnitude.  Flipping every element separately makes the draws cleaner than
    the data, shrinks the null distribution and over-rejects.  This is the
    Goncalves-Kilian convention and matches msid's IRF bootstrap.
    """
    psi = rng.integers(0, 2, size=(E.shape[0], 1)) * 2.0 - 1.0
    return fitted + (psi * E) @ B0.T


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


def _weighted_sigmas(U: np.ndarray, w: np.ndarray, demean: bool = False) -> list:
    """Smoothed-probability-weighted covariance of the residuals, per regime.

    ``Sigma_m = sum_t p_t(m) u_t u_t' / sum_t p_t(m)`` over the rows supplied.
    With ``demean=True`` each regime is centred on its own weighted mean first;
    the bootstrap draws are zero-mean by construction, so the default keeps the
    observed and the simulated side on the same footing.
    """
    out = []
    for m in range(w.shape[1]):
        wm = w[:, m]
        denom = max(float(wm.sum()), 1e-12)
        Um = U
        if demean:
            Um = U - (wm[:, None] * U).sum(axis=0) / denom
        out.append((Um * wm[:, None]).T @ Um / denom)
    return out


def b_stability_test_fixed_probs(
    results,
    center_date,
    window_before: tuple[int, int] = (16, 5),
    window_after: tuple[int, int] = (5, 16),
    n_boot: int = 999,
    min_regime_obs: int = 30,
    prob_threshold: float = 0.7,
    center: bool = False,
    demean: bool = False,
    n_jobs: int = -1,
    random_state=None,
) -> BStabilityResult:
    """Constant-B test with the regime probabilities HELD FIXED (M = 2 only).

    This is the Tether draft's own implementation of Appendix F.2, kept
    separate from :func:`b_stability_test` because it is a different statistic,
    not a different setting of the same one.  Three things differ.

    1.  B is recovered in closed form inside each window, from the pair of
        smoothed-probability-weighted residual covariances,
        ``Sigma_2 v = lambda Sigma_1 v`` with ``B = (V')^{-1}``.  Nothing is
        re-estimated: Theta, P and the smoothed probabilities stay at their
        pooled values.  :func:`b_stability_test` instead re-runs the whole EM
        inside each window, so it asks whether a window-specific REFIT lands at
        the same B.  This one asks whether one B diagonalizes the two window
        covariances when everything else is held at the pooled fit.
    2.  The null is the Gaussian parametric one, ``u*_t ~ N(0, Sigma*_t)`` with
        ``Sigma*_t = sum_m p_{t|T}(m) Sigma_m``, which keeps the date-specific
        regime composition through the weights rather than assigning each date
        a hard regime.
    3.  The Wald statistic is UNCENTRED by default (``center=False``), matching
        the draft.  Observed and bootstrap values are treated identically, so
        the p-value is valid; it simply has less power than the centred version
        when the window estimator is biased under the null.

    Because no EM runs per replication, this is one to two orders of magnitude
    faster than :func:`b_stability_test`, so ``n_boot = 999`` is cheap.

    Returns the same :class:`BStabilityResult`, so ``subtest`` works on the
    stored draws exactly as it does for the other test.
    """
    M = getattr(results, "M", None) or results.model.M
    if M != 2:
        raise ValueError(
            f"b_stability_test_fixed_probs needs exactly 2 regimes, got M = {M}; "
            "the closed-form pair decomposition is defined for the baseline pair only"
        )
    index = results.smoothed_probs_.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("b_stability_test_fixed_probs needs a DatetimeIndex on the data")

    probs = results.smoothed_probs_.to_numpy()
    # residuals recomputed from theta_, never read from residuals_, which can be
    # stale after a warm-started refit
    U = results._DY - results._Z @ results.theta_.T
    T, K = U.shape

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

    def _b_of(Ux: np.ndarray, rows: np.ndarray) -> np.ndarray:
        from .invariance import _b_from_pair

        S1, S2 = _weighted_sigmas(Ux[rows], probs[rows], demean=demean)
        return _align_b(_b_from_pair(S1, S2), results.B_)

    B1 = _b_of(U, rows1)
    B2 = _b_of(U, rows2)

    # Gaussian parametric null: one common B is imposed through Sigma_1, Sigma_2
    mix = np.einsum("tm,mij->tij", probs, np.stack(results.Sigma_))
    chols = np.linalg.cholesky(mix)

    def _one(child):
        rng = np.random.default_rng(child)
        Ub = np.einsum("tij,tj->ti", chols, rng.standard_normal((T, K)))
        try:
            return _b_of(Ub, rows1), _b_of(Ub, rows2)
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return None

    children = np.random.SeedSequence(random_state).spawn(n_boot)
    out = [o for o in Parallel(n_jobs=n_jobs)(delayed(_one)(c) for c in children) if o is not None]
    if len(out) < max(20, K**2 + 1):
        raise RuntimeError("too few successful bootstrap replications for V[Delta]")
    D1 = np.array([o[0] for o in out])
    D2 = np.array([o[1] for o in out])

    W, p, _ = _wald(B1, B2, D1, D2, elements=None, relative=False, center=center)
    return BStabilityResult(
        statistic=W,
        p_value=p,
        n_boot=len(out),
        B1=B1,
        B2=B2,
        windows=(window_before, window_after),
        null="gaussian_fixed_probs" + ("" if center else ", uncentred"),
        centered=center,
        draws1=D1,
        draws2=D2,
    )


def b_stability_test(
    results,
    center_date,
    window_before: tuple[int, int] = (16, 5),
    window_after: tuple[int, int] = (5, 16),
    n_boot: int = 3000,
    min_regime_obs: int = 30,
    prob_threshold: float = 0.7,
    null: str = "wild",
    n_jobs: int = -1,
    random_state=None,
) -> BStabilityResult:
    """Overlapping-window Wald test for temporal stability of B (App. F.2).

    Window 1 spans ``window_before = (months before, months after)`` around
    ``center_date`` (paper default 16 before / 5 after); Window 2 spans
    ``window_after`` (5 before / 16 after).  Refuses to run if either window
    holds fewer than ``min_regime_obs`` high-probability observations from
    each regime, since B is not identified within such a window.

    Both windows are re-estimated warm started from the full-sample fit and
    aligned to it by the closest signed column permutation, so they share one
    reference state.

    The null of a common B
    ----------------------
    Each window normalizes its state-1 shocks to unit variance WITHIN that
    window.  A window that happens to contain a few very large shocks therefore
    gets a larger impact matrix even when the true B is constant.  The
    bootstrap has to reproduce that, or its variance is too small and the test
    over-rejects.

    ``null="wild"`` (default)
        Rotate the residuals into structural shocks with the full-sample B,
        flip their signs at random, rotate back:
        ``eps_t = B^-1 u_t``, ``u*_t = B (psi_t * eps_t)`` with a SINGLE
        Rademacher draw ``psi_t`` per date.  Every date shares
        ONE B, so the null is imposed, and every date keeps its realized shock
        magnitude, so episodes stay in the windows that contain them.  This is
        the construction msid already uses for the LR bootstrap.
    ``null="gaussian"``
        ``u*_t ~ N(0, sum_m p_{t|T}(m) Sigma_m)``, the earlier behaviour.  Kept
        for comparison.  Under fat-tailed, clustered shocks it cannot produce
        the window-to-window variation the estimator actually has, and it
        over-rejects.

    The statistic tests all of vec(B).  Use ``result.subtest(...)`` for a
    subset or for the scale-free version from the same replications.
    """
    if null not in ("wild", "gaussian"):
        raise ValueError(f'null must be "wild" or "gaussian", got {null!r}')
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

    T, K = DY.shape
    fitted = Z @ results.theta_.T
    if null == "wild":
        # residuals recomputed from theta_, never read from residuals_, which
        # can be stale after a warm-started refit
        E = (DY - fitted) @ np.linalg.inv(results.B_).T
        B0 = results.B_

        def _make(rng):
            return _wild_draw(fitted, E, B0, rng)

    else:
        mix = np.einsum("tm,mij->tij", probs, np.stack(results.Sigma_))
        chols = np.linalg.cholesky(mix)

        def _make(rng):
            return fitted + np.einsum("tij,tj->ti", chols, rng.standard_normal((T, K)))

    def _one(child):
        DYb = _make(np.random.default_rng(child))
        try:
            return (
                _estimate_b_window(results, rows1, DYb, Z),
                _estimate_b_window(results, rows2, DYb, Z),
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return None

    children = np.random.SeedSequence(random_state).spawn(n_boot)
    out = [o for o in Parallel(n_jobs=n_jobs)(delayed(_one)(c) for c in children) if o is not None]
    if len(out) < max(20, K**2 + 1):
        raise RuntimeError("too few successful bootstrap replications for V[Delta]")
    D1 = np.array([o[0] for o in out])
    D2 = np.array([o[1] for o in out])

    W, p, _ = _wald(B1, B2, D1, D2, elements=None, relative=False)
    return BStabilityResult(
        statistic=W,
        p_value=p,
        n_boot=len(out),
        B1=B1,
        B2=B2,
        windows=(window_before, window_after),
        null=null,
        draws1=D1,
        draws2=D2,
    )
