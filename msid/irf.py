"""Structural impulse responses and conditional (per-state) FEVDs.

* MSVAR: standard MA(infinity) accumulation from the companion form,
  ``Theta_h = Phi_h B`` (Luetkepohl 2005, Ch. 2-3).
* MSVECM: level responses from the equivalent levels-VAR representation of
  the VECM (Granger representation; HL 2014 default).  ``cumulate="diff"``
  returns cumulated-difference responses, which coincide with the level
  responses by construction and are exposed for convenience.

Conditional FEVDs weight shock j by its state-m variance lambda_mj
(lambda_1j = 1), reproducing the HL Table 8 layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["IRFResults", "fevd", "point_irf"]


def _ma_coeffs(A: list[np.ndarray], K: int, horizon: int) -> np.ndarray:
    """Phi_0..Phi_h of the VAR MA representation via the companion form."""
    p = len(A)
    comp = np.zeros((K * p, K * p)) if p else np.zeros((K, K))
    for i, Ai in enumerate(A):
        comp[:K, i * K : (i + 1) * K] = Ai
    if p > 1:
        comp[K:, : K * (p - 1)] = np.eye(K * (p - 1))
    Phis = np.empty((horizon + 1, K, K))
    power = np.eye(max(K * p, K))
    for h in range(horizon + 1):
        Phis[h] = power[:K, :K]
        power = comp @ power if p else power * 0.0
    return Phis


def point_irf(
    model, Theta: np.ndarray, B: np.ndarray, horizon: int, cumulate: str = "levels"
) -> np.ndarray:
    """(horizon+1, K, K) array; entry [h, i, j] = response of variable i to
    a one-standard-deviation (state 1) shock j after h periods."""
    from .model import MSVECM

    K = model.K
    if isinstance(model, MSVECM):
        A = model.levels_var_coefs(Theta)
    else:
        A = model.slope_blocks(Theta)["A"]
    Phis = _ma_coeffs(A, K, horizon)
    irfs = np.einsum("hik,kj->hij", Phis, B)
    if cumulate == "diff":
        # difference responses cumulated back to levels (identical by
        # construction; see module docstring)
        diffs = np.diff(irfs, axis=0, prepend=np.zeros((1, K, K)))
        irfs = np.cumsum(diffs, axis=0)
    elif cumulate != "levels":
        raise ValueError("cumulate must be 'levels' or 'diff'")
    return irfs


def fevd(
    model,
    Theta: np.ndarray,
    B: np.ndarray,
    Lambdas: list[np.ndarray],
    horizon: int,
    by_state: bool = True,
) -> pd.DataFrame:
    """Conditional-on-state forecast error variance decompositions.

    In state m the structural shock j has variance lambda_mj, so the
    contribution of shock j to variable i's forecast error variance at
    horizon h is ``sum_{s<h} Theta_{s,ij}^2 lambda_mj`` normalized across j
    (HL 2014, Table 8 layout).
    """
    K = model.K
    irfs = point_irf(model, Theta, B, horizon)
    sq = irfs**2  # (h+1, K, K)
    csum = np.cumsum(sq, axis=0)  # cumulative through h
    states = range(len(Lambdas)) if by_state else [0]
    rows = []
    names = list(model.y.columns)
    for m in states:
        lam = np.asarray(Lambdas[m])
        for h in range(1, horizon + 1):
            contrib = csum[h - 1] * lam[None, :]
            total = contrib.sum(axis=1, keepdims=True)
            share = contrib / np.maximum(total, 1e-300)
            for i in range(K):
                row = {"state": m + 1, "horizon": h, "variable": names[i]}
                for j in range(K):
                    row[f"shock {j + 1}"] = share[i, j]
                rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class IRFResults:
    """Point IRFs with optional bootstrap percentile bands."""

    irfs: np.ndarray  # (h+1, K, K)
    var_names: list
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    draws: np.ndarray | None = None  # full bootstrap array
    ci: float | None = None
    shock_names: list = field(default_factory=list)

    def __post_init__(self):
        if not self.shock_names:
            self.shock_names = [f"shock {j + 1}" for j in range(self.irfs.shape[2])]

    @property
    def horizon(self) -> int:
        return self.irfs.shape[0] - 1

    def plot(self, **kwargs):
        from .plotting import plot_irf

        return plot_irf(self, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        h, K, _ = self.irfs.shape
        rows = []
        for t in range(h):
            for i in range(K):
                for j in range(K):
                    rows.append(
                        {
                            "horizon": t,
                            "variable": self.var_names[i],
                            "shock": self.shock_names[j],
                            "irf": self.irfs[t, i, j],
                            "lo": np.nan if self.lo is None else self.lo[t, i, j],
                            "hi": np.nan if self.hi is None else self.hi[t, i, j],
                        }
                    )
        return pd.DataFrame(rows)
