"""Plots: regime-probability shading, IRF grids with bands, stability fans.

Styled after the Tether paper's Figures 4-6 (regime shading), Figure 7 /
HL Figure 4 (IRF grids) and Figures 9-11 (rolling-coefficient fans).
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_irf", "plot_regimes", "plot_rolling"]


def _shade_regime(ax, index, prob, threshold):
    mask = np.asarray(prob) > threshold
    if not mask.any():
        return
    idx = np.asarray(index)
    start = None
    for t in range(len(mask)):
        if mask[t] and start is None:
            start = idx[t]
        elif not mask[t] and start is not None:
            ax.axvspan(start, idx[t], color="lightblue", alpha=0.5, lw=0)
            start = None
    if start is not None:
        ax.axvspan(start, idx[-1], color="lightblue", alpha=0.5, lw=0)


def plot_regimes(results, series=None, threshold: float = 0.7, diff: bool = False, figsize=None):
    """One panel per regime with light-blue shading where the smoothed
    probability exceeds ``threshold`` (Tether Figures 4-6 look).

    Parameters
    ----------
    series : str or None
        Column of the data to plot (default: first variable).
    diff : bool
        Also overlay the first difference of the series in grey.
    """
    import matplotlib.pyplot as plt

    probs = results.smoothed_probs_
    M = results.M
    y = results.model.y
    col = series if series is not None else y.columns[0]
    level = y[col].reindex(probs.index)
    fig, axes = plt.subplots(M, 1, figsize=figsize or (10, 3 * M), sharex=True)
    axes = np.atleast_1d(axes)
    for m in range(M):
        ax = axes[m]
        if diff:
            d = y[col].diff().reindex(probs.index)
            ax.plot(probs.index, d, color="grey", lw=0.6, label=f"d {col}")
            ax2 = ax.twinx()
            ax2.plot(probs.index, level, color="tab:blue", lw=0.9, label=col)
            ax2.set_ylabel(col)
        else:
            ax.plot(probs.index, level, color="tab:blue", lw=0.9, label=col)
        _shade_regime(ax, probs.index, probs.iloc[:, m], threshold)
        ax.set_title(f"Regime {m + 1}")
    fig.tight_layout()
    return fig


def plot_irf(irf_results, figsize=None):
    """K x K grid: solid point estimate, dashed percentile bands, titles
    "shock -> variable" (Tether Figure 7 / HL Figure 4 style)."""
    import matplotlib.pyplot as plt

    irfs = irf_results.irfs
    h, K, _ = irfs.shape
    x = np.arange(h)
    fig, axes = plt.subplots(
        K, K, figsize=figsize or (3.2 * K, 2.6 * K), sharex=True, squeeze=False
    )
    for i in range(K):
        for j in range(K):
            ax = axes[i][j]
            ax.plot(x, irfs[:, i, j], color="black", lw=1.4)
            if irf_results.lo is not None:
                ax.plot(x, irf_results.lo[:, i, j], "--", color="tab:blue", lw=1.0)
                ax.plot(x, irf_results.hi[:, i, j], "--", color="tab:blue", lw=1.0)
            ax.axhline(0.0, color="grey", lw=0.6)
            ax.set_title(
                f"{irf_results.shock_names[j]} $\\rightarrow$ " f"{irf_results.var_names[i]}",
                fontsize=9,
            )
    fig.tight_layout()
    return fig


def plot_rolling(rolling_results, equation=None, figsize=None, ncols: int = 3):
    """Per-equation coefficient paths with 95% shaded bands over rolling
    windows (Tether Figures 9-11 style)."""
    import matplotlib.pyplot as plt

    eqs = rolling_results.var_names if equation is None else [equation]
    figs = []
    for eq in eqs:
        keys = [k for k in rolling_results.paths if k[0] == eq and any("(t-" in k[1] for _ in [0])]
        keys = [k for k in keys if "(t-" in k[1]]
        if not keys:
            continue
        n = len(keys)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize or (3.6 * ncols, 2.4 * nrows), squeeze=False, sharex=True
        )
        for k, key in enumerate(keys):
            ax = axes[k // ncols][k % ncols]
            df = rolling_results.paths[key]
            ax.plot(df.index, df["coef"], color="black", lw=1.0)
            ax.fill_between(df.index, df["lo"], df["hi"], color="tab:blue", alpha=0.25, label="95")
            ax.axhline(0.0, color="grey", lw=0.5)
            ax.set_title(f"{eq} on {key[1]}", fontsize=9)
        for k in range(len(keys), nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.suptitle(f"Rolling-window estimates: {eq} equation", fontsize=11)
        fig.tight_layout()
        figs.append(fig)
    return figs if equation is None else figs[0]
