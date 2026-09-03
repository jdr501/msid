"""Deterministic shock labeling for statistically identified SVAR/SVECM models.

Under Markov-switching (statistical) identification the columns of the
impact matrix ``B`` are pinned down only up to permutation and sign (HL
2014, Sec. 3.1): nothing in the likelihood ties column j to variable j.
Every likelihood-ratio statistic in this package is invariant to that
permutation, so no test can validate a labeling -- it has to be fixed
beforehand, by a rule stated in advance and reported with its own
diagnostics.

This module implements one such rule, the **scale-normalized maximum
own-impact assignment**:

1. Standardize each element of B by the scale of the corresponding
   variable, ``A_ij = |B_ij| / sigma_i`` with ``sigma_i`` the residual
   standard deviation of variable i, making the entries unit-free and
   comparable across rows.
2. Convert to impact *shares*, ``S_ij = A_ij^2 / sum_k A_kj^2``, so column
   j reports where shock j's impact lands as fractions summing to one.
3. Score every assignment of shocks to variables.  For an assignment c
   (variable i receives column ``c(i)``) the *total own-variable impact*
   is ``Q(c) = sum_i S[i, c(i)]``.  All K! assignments are enumerated when
   K is small; for larger K the optimum comes from a Hungarian assignment
   and the runner-up from forced-exclusion re-solves.
4. Select ``argmax Q``, then orient signs so every own impact ``B_jj`` is
   positive (done by the caller, see ``results.normalize_own_signs``).

``Q`` runs from about 1 (each shock spread evenly across variables -- no
variable-specific structure at all) to K (a perfect one-to-one match), so
the gap between the best and second-best assignment measures directly how
sharply the labeling is determined.

A rule needs a declared failure branch.  When the gap falls below
``min_gap`` the labels are flagged ambiguous: the statistical shocks are
then likely mixtures of economic shocks, and the caller should fall back
to volatility ordering (``results.sort_shocks``) and refrain from
reporting variable-indexed restrictions such as ``b_32 = 0``.  Pre-commit
a threshold appropriate to the application and report the realized gap
either way -- the default below is a floor, not a target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

__all__ = [
    "DEFAULT_MIN_GAP",
    "MAX_ENUMERATE_K",
    "LabelingReport",
    "assignment_scores",
    "impact_share_matrix",
    "max_own_impact_order",
]

#: Default minimum best-minus-second-best gap in total own-variable impact.
DEFAULT_MIN_GAP = 0.1

#: Largest K for which all K! assignments are enumerated (7! = 5040).
MAX_ENUMERATE_K = 7

_TINY = 1e-300


def impact_share_matrix(B, scale) -> np.ndarray:
    """Scale-normalized impact shares of each shock across the variables.

    Parameters
    ----------
    B : (K, K) array
        Estimated impact matrix; rows are variables, columns are shocks.
    scale : (K,) array
        Per-variable scale, normally the residual standard deviations.

    Returns
    -------
    (K, K) array ``S`` with columns summing to one; ``S[i, j]`` is the
    share of shock j's scale-adjusted impact that falls on variable i.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square, got shape {B.shape}")
    if not np.all(np.isfinite(B)):
        raise ValueError("B contains non-finite entries")
    scale = np.asarray(scale, dtype=float).ravel()
    if scale.shape[0] != B.shape[0]:
        raise ValueError(f"scale must have length K={B.shape[0]}, got {scale.shape[0]}")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale must be finite and strictly positive for every variable")

    A2 = (np.abs(B) / scale[:, None]) ** 2
    denom = A2.sum(axis=0, keepdims=True)
    if np.any(denom <= _TINY):
        warnings.warn(
            "B has a numerically zero column; the shock labeling is undefined for it",
            UserWarning,
        )
    return A2 / np.maximum(denom, _TINY)


def assignment_scores(S) -> list[tuple[tuple[int, ...], float]]:
    """All K! assignments and their total own-variable impact, best first.

    Each entry is ``(order, Q)`` where ``order[i]`` is the column assigned
    to variable i -- the permutation accepted by
    ``results.reorder_shocks``.  Only for K <= ``MAX_ENUMERATE_K``.
    """
    S = np.asarray(S, dtype=float)
    K = S.shape[0]
    if K > MAX_ENUMERATE_K:
        raise ValueError(
            f"enumerating {K}! assignments is infeasible; "
            "use max_own_impact_order, which falls back to Hungarian assignment"
        )
    rows = np.arange(K)
    scored = [(cand, float(S[rows, cand].sum())) for cand in permutations(range(K))]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def _best_two_large_k(S) -> tuple[tuple[int, ...], float, tuple[int, ...], float]:
    """Best and second-best assignment without enumeration.

    The optimum is a Hungarian assignment; the runner-up is the best
    assignment that forbids one of the optimal pairs, maximized over the K
    pairs that can be forbidden.  Both are exact.
    """
    from scipy.optimize import linear_sum_assignment

    S = np.asarray(S, dtype=float)
    K = S.shape[0]
    rows = np.arange(K)
    _, best = linear_sum_assignment(-S)
    best_score = float(S[rows, best].sum())

    runner, runner_score = None, -np.inf
    penalty = float(np.abs(S).sum()) + 1.0
    for i in range(K):
        blocked = S.copy()
        blocked[i, best[i]] -= penalty
        _, cand = linear_sum_assignment(-blocked)
        score = float(S[rows, cand].sum())
        if score > runner_score:
            runner, runner_score = cand, score
    return tuple(int(c) for c in best), best_score, tuple(int(c) for c in runner), runner_score


@dataclass
class LabelingReport:
    """Outcome and diagnostics of the max own-impact labeling rule."""

    order: list[int]
    score: float
    runner_up_order: list[int]
    runner_up_score: float
    min_gap: float
    shares: np.ndarray
    var_names: list[str] = field(default_factory=list)
    all_scores: list[tuple[tuple[int, ...], float]] | None = None
    enumerated: bool = True

    @property
    def K(self) -> int:
        return len(self.order)

    @property
    def gap(self) -> float:
        """Best minus second-best total own-variable impact."""
        return self.score - self.runner_up_score

    @property
    def ambiguous(self) -> bool:
        """True when the gap falls below the declared threshold."""
        return self.gap < self.min_gap

    @property
    def own_shares(self) -> np.ndarray:
        """Impact share of each variable's assigned shock, by variable."""
        return np.array([self.shares[i, self.order[i]] for i in range(self.K)])

    def _name(self, i: int) -> str:
        return self.var_names[i] if i < len(self.var_names) else f"var{i + 1}"

    def __str__(self) -> str:
        lines = ["Shock labeling (scale-normalized max own-impact assignment)"]
        lines.append(f"  order (new column j <- old column): {list(self.order)}")
        own = self.own_shares
        for i in range(self.K):
            lines.append(
                f"    {self._name(i):<12s} <- column {self.order[i]}   "
                f"own impact share {own[i]:.3f}"
            )
        lines.append(
            f"  best Q = {self.score:.4f}    second-best Q = {self.runner_up_score:.4f}"
            f"    gap = {self.gap:.4f}  (threshold {self.min_gap:.4f})"
        )
        lines.append(f"  Q ranges from about 1 (no variable-specific structure) to K = {self.K}")
        if self.ambiguous:
            lines.append(
                "  status: AMBIGUOUS -- gap below threshold; the shocks are likely "
                "mixtures.\n          Do not report variable-indexed restrictions; "
                "fall back to sort_shocks()."
            )
        else:
            lines.append("  status: labels separated -- variable-indexed restrictions are readable")
        if self.all_scores is not None:
            listed = ", ".join(f"{tuple(o)} {q:.4f}" for o, q in self.all_scores[:6])
            more = "" if len(self.all_scores) <= 6 else f", ... ({len(self.all_scores)} total)"
            lines.append(f"  all assignments: {listed}{more}")
        else:
            lines.append("  (K too large to enumerate; best and runner-up computed exactly)")
        return "\n".join(lines)


def max_own_impact_order(
    B,
    scale,
    min_gap: float = DEFAULT_MIN_GAP,
    strict: bool = False,
    var_names=None,
    max_enumerate: int = MAX_ENUMERATE_K,
) -> LabelingReport:
    """Apply the max own-impact labeling rule to an impact matrix.

    Parameters
    ----------
    B : (K, K) array
        Estimated impact matrix (rows variables, columns shocks).
    scale : (K,) array
        Per-variable scale, normally residual standard deviations.
    min_gap : float
        Declared threshold on the best-minus-second-best gap in total
        own-variable impact.  Below it the labeling is flagged ambiguous.
        Pre-commit this value; the default is a floor, not a target.
    strict : bool
        Raise ``ValueError`` instead of warning when the gap is below
        ``min_gap``.  Use in scripted pipelines that must not silently
        produce variable-labeled output.
    var_names : sequence of str, optional
        Variable names, for readable reporting.
    max_enumerate : int
        Enumerate all K! assignments up to this K; above it, use the
        Hungarian/forced-exclusion route (same optimum, no full table).

    Returns
    -------
    LabelingReport
        ``report.order`` is the permutation to hand to
        ``results.reorder_shocks``; the rest are the diagnostics.
    """
    S = impact_share_matrix(B, scale)
    K = S.shape[0]
    names = [str(n) for n in var_names] if var_names is not None else []

    if K <= max_enumerate:
        scored = assignment_scores(S)
        best, best_score = scored[0]
        runner, runner_score = scored[1] if len(scored) > 1 else (best, best_score)
        all_scores, enumerated = scored, True
    else:
        best, best_score, runner, runner_score = _best_two_large_k(S)
        all_scores, enumerated = None, False

    report = LabelingReport(
        order=[int(c) for c in best],
        score=float(best_score),
        runner_up_order=[int(c) for c in runner],
        runner_up_score=float(runner_score),
        min_gap=float(min_gap),
        shares=S,
        var_names=names,
        all_scores=all_scores,
        enumerated=enumerated,
    )
    if report.ambiguous:
        msg = (
            f"ambiguous shock labeling: best assignment scores {report.score:.4f} "
            f"against {report.runner_up_score:.4f} for the runner-up, a gap of "
            f"{report.gap:.4f} < {report.min_gap:.4f}; the statistical shocks are "
            "likely mixtures of economic shocks. Prefer lambda-based ordering "
            "(sort_shocks) and avoid variable-indexed restrictions."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning)
    return report
