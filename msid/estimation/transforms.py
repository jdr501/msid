"""Parameter transforms: log-Lambda parameterization, packing/unpacking.

The Tether paper's Online Appendix A replaces HL's hard lower bound of 0.01
on the diagonal elements of Lambda_m with an unconstrained optimization over
``log lambda_mi`` -- the objective exponentiates internally, so every step of
the line search satisfies positivity by construction.
"""

from __future__ import annotations

import numpy as np

from ..restrictions import Restrictions

__all__ = ["lambda_trace_order", "pack_struct", "unpack_struct"]


def pack_struct(B: np.ndarray, Lambdas: list[np.ndarray], R: Restrictions) -> np.ndarray:
    """Pack (B, Lambda_2..Lambda_M) into the M-step optimization vector.

    Layout: ``[b_free (vec order), log(diag Lambda_2), ..., log(diag Lambda_M)]``.
    """
    parts = [R.pack_b(B)]
    for lam in Lambdas:
        d = np.diag(lam) if lam.ndim == 2 else np.asarray(lam)
        parts.append(np.log(d))
    return np.concatenate(parts)


def unpack_struct(x: np.ndarray, R: Restrictions, M: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Inverse of :func:`pack_struct`; returns ``(B, [diag Lambda_m for m>=2])``."""
    K = R.K
    nb = R.n_free_b
    B = R.unpack_b(x[:nb])
    lams = [np.exp(x[nb + i * K : nb + (i + 1) * K]) for i in range(M - 1)]
    return B, lams


def lambda_trace_order(Lambdas: list[np.ndarray]) -> np.ndarray:
    """State ordering by ``tr(Lambda_m)`` with state 1 (Lambda_1 = I) included.

    Returns the permutation of states ``0..M-1`` that sorts states by
    increasing total relative variance.  This is the deterministic
    ``label_order="lambda_sort"`` rule (spec Section 3.2).
    """
    traces = [float(np.sum(np.diag(L) if np.ndim(L) == 2 else L)) for L in Lambdas]
    return np.argsort(traces, kind="stable")
