"""Restrictions on the structural impact matrix B and the long-run matrix Xi.

Zero restrictions on B enter the M-step exactly, by optimizing only over the
free elements of B (a linear reparameterization ``vec(B) = S @ b_free``).

Long-run (Xi) zeros are nonlinear in the *model* parameters but, holding the
slope parameters theta fixed within an M-step, ``Xi = C(theta) @ B`` is linear
in B (HL 2014, Eq. 2 for the VECM; ``Xi = (I - sum A_i)^{-1} B`` for the
stationary VAR).  They are enforced through a heavily weighted quadratic
penalty ``w * sum(Xi_restricted**2)`` whose weight ``w`` is escalated across
EM iterations until the constraint holds to ``xi_tol`` (default 1e-8); the
fit errors out if the constraint is still violated at convergence.  This is
the penalty route described in the handoff specification, Section 4.

Sign restrictions carry zero degrees of freedom in LR tests: they are pure
normalizations implemented by post-hoc column sign flips (spec Section 3.3,
item 2; see also HL 2014, Section 3.1 on the sign non-uniqueness of B).
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["Restrictions"]


class Restrictions:
    """Container for identification restrictions on ``B`` and ``Xi``.

    Parameters
    ----------
    K : int
        System dimension.
    b_pattern : ndarray, optional
        ``(K, K)`` array with ``np.nan`` for free elements and ``0.0`` for
        excluded elements of ``B``.
    xi_pattern : ndarray, optional
        Same convention for the long-run matrix ``Xi``.

    Examples
    --------
    >>> R = Restrictions(K=3)
    >>> R.b_zeros([(2, 0), (2, 1)])       # b31 = b32 = 0   (0-indexed)
    >>> R.b_signs({(2, 0): "+"})          # b31 > 0 -- normalization, df = 0
    >>> R.xi_zeros([(1, 2)])              # long-run zero on Xi
    """

    def __init__(
        self,
        K: int,
        b_pattern: np.ndarray | None = None,
        xi_pattern: np.ndarray | None = None,
    ) -> None:
        if K < 2:
            raise ValueError("K must be >= 2")
        self.K = int(K)
        self._b_zero: set[tuple[int, int]] = set()
        self._xi_zero: set[tuple[int, int]] = set()
        self._b_sign: dict[tuple[int, int], int] = {}
        if b_pattern is not None:
            self._from_pattern(np.asarray(b_pattern, dtype=float), target="b")
        if xi_pattern is not None:
            self._from_pattern(np.asarray(xi_pattern, dtype=float), target="xi")

    # ------------------------------------------------------------------ setup
    def _check_idx(self, idx: tuple[int, int]) -> tuple[int, int]:
        i, j = int(idx[0]), int(idx[1])
        if not (0 <= i < self.K and 0 <= j < self.K):
            raise ValueError(f"index {idx} out of bounds for K={self.K} (0-indexed)")
        return (i, j)

    def _from_pattern(self, pat: np.ndarray, target: str) -> None:
        if pat.shape != (self.K, self.K):
            raise ValueError(f"pattern must be ({self.K}, {self.K}), got {pat.shape}")
        zeros = [(i, j) for i in range(self.K) for j in range(self.K) if pat[i, j] == 0.0]
        bad = np.isfinite(pat) & (pat != 0.0)
        if bad.any():
            raise ValueError("pattern entries must be np.nan (free) or 0.0 (excluded)")
        if target == "b":
            self.b_zeros(zeros)
        else:
            self.xi_zeros(zeros)

    def b_zeros(self, indices: list[tuple[int, int]]) -> Restrictions:
        """Impose ``b_ij = 0`` for each (i, j) in *indices* (0-indexed)."""
        for idx in indices:
            self._b_zero.add(self._check_idx(idx))
        self._warn_degenerate()
        return self

    def xi_zeros(self, indices: list[tuple[int, int]]) -> Restrictions:
        """Impose long-run zeros ``xi_ij = 0`` (0-indexed), HL-style."""
        for idx in indices:
            self._xi_zero.add(self._check_idx(idx))
        return self

    def b_signs(self, signs: dict[tuple[int, int], str]) -> Restrictions:
        """Sign normalizations such as ``{(2, 0): "+"}`` meaning b31 > 0.

        Sign restrictions are normalizations only and contribute zero degrees
        of freedom to LR tests.  They are enforced by post-hoc column flips.
        """
        for idx, s in signs.items():
            idx = self._check_idx(idx)
            if s not in ("+", "-"):
                raise ValueError(f"sign must be '+' or '-', got {s!r}")
            if idx in self._b_zero:
                raise ValueError(f"element {idx} is restricted to zero; cannot sign-restrict it")
            self._b_sign[idx] = 1 if s == "+" else -1
        return self

    def _warn_degenerate(self) -> None:
        for j in range(self.K):
            n_col = sum(1 for (i, jj) in self._b_zero if jj == j)
            if n_col > self.K - 1:
                warnings.warn(
                    f"column {j} of B carries {n_col} zero restrictions (> K-1 = "
                    f"{self.K - 1}); the column may be degenerate",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------- properties
    @property
    def b_zero_indices(self) -> list[tuple[int, int]]:
        return sorted(self._b_zero)

    @property
    def xi_zero_indices(self) -> list[tuple[int, int]]:
        return sorted(self._xi_zero)

    @property
    def sign_restrictions(self) -> dict[tuple[int, int], int]:
        return dict(self._b_sign)

    @property
    def n_free_b(self) -> int:
        """Number of free (unrestricted) elements of B."""
        return self.K * self.K - len(self._b_zero)

    @property
    def n_zero_restrictions(self) -> int:
        """Total zero restrictions on B and Xi -- the LR degrees of freedom."""
        return len(self._b_zero) + len(self._xi_zero)

    @property
    def has_xi_restrictions(self) -> bool:
        return len(self._xi_zero) > 0

    # ------------------------------------------------------------- machinery
    def selection_matrix(self) -> np.ndarray:
        """``(K^2, n_free_b)`` matrix S with ``vec(B) = S @ b_free``.

        vec() stacks columns (column-major), matching HL's conventions.
        """
        K = self.K
        free = [(i, j) for j in range(K) for i in range(K) if (i, j) not in self._b_zero]
        S = np.zeros((K * K, len(free)))
        for col, (i, j) in enumerate(free):
            S[j * K + i, col] = 1.0
        return S

    def free_b_indices(self) -> list[tuple[int, int]]:
        """(row, col) of free elements, in vec (column-major) order."""
        K = self.K
        return [(i, j) for j in range(K) for i in range(K) if (i, j) not in self._b_zero]

    def pack_b(self, B: np.ndarray) -> np.ndarray:
        """Extract the free elements of ``B`` in vec order."""
        return np.array([B[i, j] for (i, j) in self.free_b_indices()])

    def unpack_b(self, b_free: np.ndarray) -> np.ndarray:
        """Rebuild ``B`` (with exact zeros) from its free elements."""
        K = self.K
        return (self.selection_matrix() @ np.asarray(b_free)).reshape((K, K), order="F")

    def normalize_signs(self, B: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        """Flip column signs of ``B`` to satisfy the sign normalizations.

        Without explicit sign restrictions, columns are normalized to have a
        positive diagonal element; if ``reference`` is given (e.g. the
        original ML estimate inside the bootstrap), columns are instead
        aligned with the reference to prevent band contamination from flips
        (spec Section 9, item 3).
        """
        B = np.array(B, dtype=float, copy=True)
        flipped = np.ones(self.K)
        if reference is not None:
            for j in range(self.K):
                if B[:, j] @ reference[:, j] < 0:
                    flipped[j] = -1.0
        else:
            for (i, j), s in self._b_sign.items():
                if s * B[i, j] < 0:
                    flipped[j] = -1.0
            for j in range(self.K):
                if not any(jj == j for (_, jj) in self._b_sign) and B[j, j] < 0:
                    flipped[j] = -1.0
        return B * flipped

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Restrictions(K={self.K}, b_zeros={self.b_zero_indices}, "
            f"xi_zeros={self.xi_zero_indices}, signs={self._b_sign}, "
            f"n_free_b={self.n_free_b})"
        )
