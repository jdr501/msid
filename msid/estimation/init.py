"""Starting values and the multi-start manager (spec Section 3.1).

Initialization follows Tether Online Appendix A:

1. ``P <- M^{-1} 1_M 1'_M``
2. ``theta_hat <- OLS`` on the pooled model,
3. ``B <- (T^{-1} sum u u')^{1/2} + B0`` with B0 small random numbers,
4. ``Lambda_m <- I_K x Lambda0_m`` with Lambda0_m positive random draws
   (LogUniform(0.1, 10) by default) -- the Tether paper's improvement over
   HL's plain identity start, which tends to collapse to the single-regime
   OLS solution,
5. ``xi_{0|0} <- 1_M / M``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from ..restrictions import Restrictions
from .em import EMState

__all__ = ["make_starts", "ols_start"]


def ols_start(DY: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pooled OLS: Theta_hat (K, q) and residuals U (T, K)."""
    Theta, *_ = np.linalg.lstsq(Z, DY, rcond=None)
    Theta = Theta.T
    U = DY - Z @ Theta.T
    return Theta, U


def make_starts(
    DY: np.ndarray,
    Z: np.ndarray,
    R: Restrictions,
    M: int,
    n_starts: int = 50,
    b0_scale: float = 0.1,
    lambda_init: str = "random",
    lambda_range: tuple[float, float] = (0.1, 10.0),
    random_state=None,
) -> list[EMState]:
    """Generate ``n_starts`` independent EM initializations.

    ``lambda_init="random"`` (default) draws LogUniform(*lambda_range*)
    diagonals; ``"identity"`` reproduces HL's plain identity start (both are
    implemented per the spec; identity is available for comparison).
    """
    rng = np.random.default_rng(random_state)
    K = DY.shape[1]
    T = DY.shape[0]
    Theta, U = ols_start(DY, Z)
    S = (U.T @ U) / T
    Broot = np.real(sqrtm(S))
    scale = b0_scale * np.std(U)
    P0 = np.full((M, M), 1.0 / M)
    xi0 = np.full(M, 1.0 / M)
    starts = []
    for s in range(n_starts):
        B0 = Broot + rng.normal(scale=scale, size=(K, K))
        # zero restrictions are honored from the start
        for i, j in R.b_zero_indices:
            B0[i, j] = 0.0
        if lambda_init == "identity" and s == 0 or lambda_init == "identity":
            lams = [np.ones(K) for _ in range(M - 1)]
        else:
            lo, hi = np.log(lambda_range[0]), np.log(lambda_range[1])
            lams = [np.exp(rng.uniform(lo, hi, size=K)) for _ in range(M - 1)]
        starts.append(
            EMState(
                Theta=Theta.copy(), B=R.normalize_signs(B0), lams=lams, P=P0.copy(), xi0=xi0.copy()
            )
        )
    return starts
