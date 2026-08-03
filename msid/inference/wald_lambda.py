"""Identification diagnostics: Wald tests for equality of Lambda diagonals.

Statistical identification hinges on distinctness of the lambda_mi (HL 2014,
Section 3.1; Lanne et al. 2010, Proposition 1):

* M = 2: B is unique (up to sign/permutation) iff all lambda_2i are
  distinct -> all pairwise tests ``H0: lambda_2i = lambda_2j`` (HL Table 4).
* M >= 3: uniqueness needs, for each pair (k, l), SOME regime j with
  ``lambda_jk != lambda_jl`` -> joint pairwise tests
  ``H0: lambda_2k = lambda_2l AND lambda_3k = lambda_3l [AND ...]`` for every
  pair, plus per-regime all-equal tests (HL Table 5).

Wald statistics use the OPG covariance of the lambdas (delta method from the
log-parameterization; see :mod:`msid.inference.std_errors`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["IdentificationReport", "wald_lambda_tests"]


@dataclass
class IdentificationReport:
    table: pd.DataFrame
    verdict: str
    identified: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "Identification status\n"
            + "-" * 60
            + "\n"
            + self.table.to_string(index=False, float_format=lambda v: f"{v:.3f}")
            + "\n"
            + self.verdict
        )


def _wald(Rmat: np.ndarray, lam: np.ndarray, V: np.ndarray) -> tuple[float, int, float]:
    d = Rmat @ lam
    Vd = Rmat @ V @ Rmat.T
    try:
        stat = float(d @ np.linalg.solve(Vd, d))
    except np.linalg.LinAlgError:
        stat = float(d @ np.linalg.pinv(Vd) @ d)
    df = int(np.linalg.matrix_rank(Rmat))
    p = float(stats.chi2.sf(stat, df))
    return stat, df, p


def wald_lambda_tests(
    lams: list[np.ndarray], V_lambda: np.ndarray, alpha: float = 0.10
) -> IdentificationReport:
    """Run the full identification test battery (spec Section 5).

    Parameters
    ----------
    lams : list of (K,) arrays
        Diagonals of Lambda_2, ..., Lambda_M.
    V_lambda : ((M-1)K, (M-1)K)
        OPG covariance of the stacked lambdas (regime-major:
        ``[lambda_2', lambda_3', ...]``).
    alpha : float
        Level for the plain-language verdict (default 10%, as in HL's
        discussion of weak identification).
    """
    Mm1 = len(lams)
    K = len(lams[0])
    lam = np.concatenate([np.asarray(l) for l in lams])
    rows = []
    worst_p, worst_pair = -1.0, None
    for k in range(K):
        for l in range(k + 1, K):
            Rm = np.zeros((Mm1, Mm1 * K))
            for m in range(Mm1):
                Rm[m, m * K + k] = 1.0
                Rm[m, m * K + l] = -1.0
            stat, df, p = _wald(Rm, lam, V_lambda)
            if Mm1 == 1:
                h0 = f"l{2}{k + 1} = l{2}{l + 1}"
            else:
                h0 = " and ".join(f"l{m + 2}{k + 1} = l{m + 2}{l + 1}" for m in range(Mm1))
            rows.append({"H0": h0, "stat": stat, "df": df, "p-value": p})
            if p > worst_p:
                worst_p, worst_pair = p, (k + 1, l + 1)
    if Mm1 > 1:  # per-regime all-equal joint tests (HL Table 5, last rows)
        for m in range(Mm1):
            Rm = np.zeros((K - 1, Mm1 * K))
            for k in range(K - 1):
                Rm[k, m * K + k] = 1.0
                Rm[k, m * K + k + 1] = -1.0
            stat, df, p = _wald(Rm, lam, V_lambda)
            h0 = " = ".join(f"l{m + 2}{k + 1}" for k in range(K))
            rows.append({"H0": h0, "stat": stat, "df": df, "p-value": p})
    table = pd.DataFrame(rows)

    pair_p = table["p-value"].iloc[: K * (K - 1) // 2]
    identified = bool((pair_p < alpha).all())
    if identified:
        verdict = f"Identification supported at {int(alpha * 100)}% for all pairs."
    else:
        verdict = (
            f"WARNING: cannot reject equality for pair {worst_pair} "
            f"(p={worst_p:.2f}); B is only set-identified -- interpret "
            "restricted-model tests as upper bounds on p-values (HL Sec. 4.2)."
        )
    return IdentificationReport(table=table, verdict=verdict, identified=identified)
