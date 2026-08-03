"""Closed-form xi-weighted GLS update for the slope parameters theta.

Implements the formula of HL (2014, Appendix, "Estimate theta") / Tether
Online Appendix A:

    theta_hat = [ sum_m ( sum_t xi_mt|T Z_{t-1} Z'_{t-1} ) kron Sigma_m^{-1} ]^{-1}
                x sum_t ( sum_m xi_mt|T Z_{t-1} kron Sigma_m^{-1} ) Dy_t

with theta = vec(Theta), Theta the (K, q) matrix [nu0, nu1, alpha, Gamma_1..Gamma_p]
(or [nu0, nu1, A_1..A_p] for the levels VAR) and Dy_t = Theta Z_{t-1} + u_t.
"""

from __future__ import annotations

import numpy as np

__all__ = ["update_theta"]


def update_theta(
    DY: np.ndarray,
    Z: np.ndarray,
    xi_smooth: np.ndarray,
    sigmas: list[np.ndarray],
) -> np.ndarray:
    """GLS update; returns the (K, q) slope matrix Theta.

    Parameters
    ----------
    DY : (T, K) left-hand-side observations.
    Z : (T, q) regressors ``Z_{t-1}``.
    xi_smooth : (T, M) smoothed state probabilities ``xi_{t|T}``.
    sigmas : list of M state covariance matrices.
    """
    K = DY.shape[1]
    q = Z.shape[1]
    A = np.zeros((K * q, K * q))
    b = np.zeros(K * q)
    for m, Sig in enumerate(sigmas):
        iS = np.linalg.inv(Sig)
        w = xi_smooth[:, m]
        Wzz = (Z * w[:, None]).T @ Z  # sum_t xi Z Z'
        A += np.kron(Wzz, iS)
        Cyz = (DY * w[:, None]).T @ Z  # sum_t xi Dy Z'
        b += (iS @ Cyz).ravel(order="F")  # vec(iS sum xi Dy Z')
    theta = np.linalg.solve(A, b)
    return theta.reshape((K, q), order="F")
