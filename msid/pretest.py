"""Pre-tests: Johansen cointegration, lag selection, heteroskedasticity.

* :func:`johansen` wraps ``statsmodels`` ``coint_johansen`` and formats
  trace / max-eigenvalue statistics with 95% critical values in the layout
  of Tether Table 11.
* :func:`select_lags` produces an AIC/SC(BIC)/HQ table.
* :func:`arch_lm` runs multivariate ARCH-LM and White heteroskedasticity
  tests on residuals -- the motivating evidence for regime switching.
* :func:`justify_ms` fits the no-MS model, runs the tests and prints a
  recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["JohansenResult", "arch_lm", "johansen", "justify_ms", "select_lags", "white_test"]


@dataclass
class JohansenResult:
    trace_stat: np.ndarray
    trace_crit: np.ndarray  # columns: 90 / 95 / 99
    maxeig_stat: np.ndarray
    maxeig_crit: np.ndarray
    evec: np.ndarray  # unnormalized cointegration vectors
    eig: np.ndarray
    table: pd.DataFrame

    def __str__(self) -> str:  # pragma: no cover
        return self.table.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def johansen(y, det_order: int = 1, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace and max-eigenvalue tests (Tether Table 11 layout).

    Parameters
    ----------
    y : (T, K) array or DataFrame.
    det_order : -1 (none), 0 (constant), 1 (constant + linear trend);
        the ``statsmodels`` convention.
    k_ar_diff : number of lagged differences in the test VECM.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    arr = np.asarray(y, dtype=float)
    with np.errstate(all="ignore"):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            res = coint_johansen(arr, det_order, k_ar_diff)
    # eigen-decompositions can return numerically complex output with a
    # negligible imaginary part; keep the real part throughout
    lr1, lr2 = np.real(res.lr1), np.real(res.lr2)
    evec, eig = np.real(res.evec), np.real(res.eig)
    K = arr.shape[1]
    rows = []
    for j in range(K):
        rows.append(
            {
                "Test": f"Trace Statistic (r<={j})" if j else "Trace Statistic (r=0)",
                "Statistic": lr1[j],
                "Critical Value (95%)": res.cvt[j, 1],
            }
        )
    for j in range(K):
        rows.append(
            {
                "Test": (
                    f"Max Eigenvalue Statistic (r<={j})" if j else "Max Eigenvalue Statistic (r=0)"
                ),
                "Statistic": lr2[j],
                "Critical Value (95%)": res.cvm[j, 1],
            }
        )
    table = pd.DataFrame(rows)
    return JohansenResult(
        trace_stat=lr1,
        trace_crit=res.cvt,
        maxeig_stat=lr2,
        maxeig_crit=res.cvm,
        evec=evec,
        eig=eig,
        table=table,
    )


def select_lags(
    y, max_lags: int = 10, criterion: str | None = None, deterministic: str = "ct"
) -> pd.DataFrame:
    """Lag-order selection table (AIC / SC / HQ) for a levels VAR.

    Returns a DataFrame indexed by lag order; the minimizing order per
    criterion is exposed in ``df.attrs['selected']``.
    """
    arr = np.asarray(y, dtype=float)
    T_all, K = arr.shape
    rows = []
    for p in range(1, max_lags + 1):
        T = T_all - max_lags  # common sample across p
        Yt = arr[max_lags:]
        blocks = []
        if deterministic in ("c", "ct"):
            blocks.append(np.ones((T, 1)))
        if deterministic == "ct":
            blocks.append(np.arange(1, T + 1, dtype=float)[:, None])
        for i in range(1, p + 1):
            blocks.append(arr[max_lags - i : max_lags - i + T])
        Z = np.hstack(blocks)
        coef, *_ = np.linalg.lstsq(Z, Yt, rcond=None)
        U = Yt - Z @ coef
        Sig = (U.T @ U) / T
        _, logdet = np.linalg.slogdet(Sig)
        k = Z.shape[1] * K
        rows.append(
            {
                "lags": p,
                "AIC": logdet + 2.0 * k / T,
                "SC": logdet + np.log(T) * k / T,
                "HQ": logdet + 2.0 * np.log(np.log(T)) * k / T,
            }
        )
    df = pd.DataFrame(rows).set_index("lags")
    df.attrs["selected"] = {c: int(df[c].idxmin()) for c in df.columns}
    if criterion is not None:
        df.attrs["choice"] = df.attrs["selected"][criterion.upper()]
    return df


def arch_lm(residuals, lags: int = 5) -> dict:
    """Multivariate ARCH-LM test (Doornik-Hendry style).

    Regresses ``vech(u_t u_t')`` on its own ``lags`` lags; under H0 of no
    ARCH, ``T R^2`` per the system is chi2 with ``lags * n^2`` df where
    ``n = K (K + 1) / 2`` (Luetkepohl 2005, Sec. 16.5).
    """
    U = np.asarray(residuals, dtype=float)
    T, K = U.shape
    n = K * (K + 1) // 2
    iu = np.triu_indices(K)
    V = np.array([np.outer(u, u)[iu] for u in U])  # (T, n) vech series
    Y = V[lags:]
    X = np.hstack([np.ones((T - lags, 1))] + [V[lags - i - 1 : T - i - 1] for i in range(lags)])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ coef
    Y0 = Y - Y.mean(axis=0)
    Om = (E.T @ E) / len(Y)
    Om0 = (Y0.T @ Y0) / len(Y)
    R2m = 1.0 - np.trace(np.linalg.lstsq(Om0, Om, rcond=None)[0]) / n
    stat = 0.5 * len(Y) * n * (n + 1) * R2m
    df = lags * n * n
    return {
        "statistic": float(stat),
        "df": int(df),
        "p-value": float(stats.chi2.sf(stat, df)),
        "test": "multivariate ARCH-LM",
    }


def white_test(residuals, regressors) -> dict:
    """White heteroskedasticity test, equation by equation, aggregated.

    Regresses each squared residual on regressors and their squares; the
    system statistic sums the per-equation ``T R^2`` values (independent
    under H0), chi2 with summed df.
    """
    U = np.asarray(residuals, dtype=float)
    Z = np.asarray(regressors, dtype=float)
    with np.errstate(over="ignore"):
        X = np.hstack([np.ones((len(Z), 1)), Z, Z**2])
    X = X[:, np.std(X, axis=0) > 1e-12]
    X = np.hstack([np.ones((len(Z), 1)), X])
    stat_total, df_total = 0.0, 0
    for k in range(U.shape[1]):
        yv = U[:, k] ** 2
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        e = yv - X @ coef
        ss_res = float(e @ e)
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
        stat_total += len(yv) * r2
        df_total += X.shape[1] - 1
    return {
        "statistic": stat_total,
        "df": df_total,
        "p-value": float(stats.chi2.sf(stat_total, df_total)),
        "test": "White (system)",
    }


def justify_ms(
    y, lags: int = 2, deterministic: str = "ct", arch_lags: int = 5, print_output: bool = True
) -> dict:
    """Fit the no-MS levels VAR, test its residuals for heteroskedasticity,
    and print a recommendation (spec Section 11)."""
    arr = np.asarray(y, dtype=float)
    T_all = arr.shape[0]
    p = lags
    T = T_all - p
    Yt = arr[p:]
    blocks = []
    if deterministic in ("c", "ct"):
        blocks.append(np.ones((T, 1)))
    if deterministic == "ct":
        blocks.append(np.arange(1, T + 1, dtype=float)[:, None])
    for i in range(1, p + 1):
        blocks.append(arr[p - i : p - i + T])
    Z = np.hstack(blocks)
    coef, *_ = np.linalg.lstsq(Z, Yt, rcond=None)
    U = Yt - Z @ coef
    res_arch = arch_lm(U, lags=arch_lags)
    res_white = white_test(U, Z[:, 1:] if deterministic != "n" else Z)
    reject = res_arch["p-value"] < 0.05 or res_white["p-value"] < 0.05
    rec = (
        "Residual heteroskedasticity detected -- a Markov-switching "
        "covariance specification is warranted."
        if reject
        else "No strong evidence of residual heteroskedasticity; MS in the "
        "covariances may not be identified from these data."
    )
    out = {"arch_lm": res_arch, "white": res_white, "recommendation": rec}
    if print_output:
        for r in (res_arch, res_white):
            print(
                f"{r['test']}: stat = {r['statistic']:.2f}, df = {r['df']}, "
                f"p = {r['p-value']:.4f}"
            )
        print(rec)
    return out
