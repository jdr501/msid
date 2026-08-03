"""Outer-product-of-gradients standard errors (HL 2014, Appendix).

``S = sum_t (dl_t/dgamma)(dl_t/dgamma')`` with per-observation contributions
``l_t = log(xi'_{t|t-1} eta_t)``, computed block-diagonally in

* ``gamma1`` = theta parameters, and
* ``gamma2`` = (free elements of vec B, log-lambdas mapped back to lambdas by
  the delta method, free elements of P),

exactly as in HL's appendix ("Estimation of standard errors").  Scores are
exact autograd derivatives -- no finite differences.  Parameters at the
boundary of the parameter space (e.g. ``p_11 = 1``) get ``nan`` standard
errors, reported as "na" as in HL Table 2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
from autograd.differential_operators import make_jvp_reversemode

from ..estimation.likelihood import filter_loglik_contributions
from ..restrictions import Restrictions

__all__ = ["OPGStandardErrors", "compute_opg_se"]

_BOUNDARY_TOL = 1e-6


@dataclass
class OPGStandardErrors:
    """Standard errors and (gamma2) covariance blocks from the OPG estimator."""

    se_theta: np.ndarray  # (K, q), nan where undefined
    se_B: np.ndarray  # (K, K), nan at restricted zeros
    se_lambda: list[np.ndarray]  # per regime m >= 2, delta-method SEs
    V_lambda: np.ndarray  # ((M-1)K, (M-1)K) covariance of lambdas
    se_P: np.ndarray  # (M, M), nan in dependent row / at boundary
    boundary: list[str]  # names of parameters at the boundary


def _pack_P_free(P: np.ndarray) -> np.ndarray:
    """Free transition parameters: all M(M-1) elements above the last row,
    column-major (columns sum to one, so the last row is dependent)."""
    return P[:-1, :].ravel(order="F")


def _unpack_P_free(pfree, M: int):
    top = anp.reshape(pfree, (M - 1, M), order="F")
    last = 1.0 - anp.sum(top, axis=0, keepdims=True)
    return anp.concatenate([top, last], axis=0)


def compute_opg_se(
    DY: np.ndarray,
    Z: np.ndarray,
    R: Restrictions,
    B: np.ndarray,
    lams: list[np.ndarray],
    P: np.ndarray,
    xi0: np.ndarray,
    Theta: np.ndarray,
) -> OPGStandardErrors:
    """Block-diagonal OPG standard errors for the fitted model."""
    K = DY.shape[1]
    M = P.shape[0]
    q = Z.shape[1]
    S_sel = R.selection_matrix()
    nb = S_sel.shape[1]
    U0 = DY - Z @ Theta.T

    loglam = np.concatenate([np.log(np.asarray(l)) for l in lams]) if M > 1 else np.array([])
    pfree = _pack_P_free(P)
    x2 = np.concatenate([R.pack_b(B), loglam, pfree])

    def lt_gamma2(x):
        Bm = anp.reshape(S_sel @ x[:nb], (K, K), order="F")
        lam_list = [anp.exp(x[nb + i * K : nb + (i + 1) * K]) for i in range(M - 1)]
        Pm = _unpack_P_free(x[nb + (M - 1) * K :], M)
        return filter_loglik_contributions(U0, Bm, lam_list, Pm, xi0)

    def lt_gamma1(th):
        Th = anp.reshape(th, (K, q), order="F")
        U = DY - Z @ Th.T
        lam_list = [anp.asarray(l) for l in lams]
        return filter_loglik_contributions(U, anp.asarray(B), lam_list, anp.asarray(P), xi0)

    def _jac_forward(fun, x):
        """(T, n) Jacobian via JVPs (reverse-over-reverse): one linear-cost
        pass per parameter, instead of one reverse pass per observation --
        the filter recursion makes the row-wise route O(T^2)."""
        jvp = make_jvp_reversemode(fun)(x)
        cols = []
        for k in range(x.size):
            v = np.zeros(x.size)
            v[k] = 1.0
            cols.append(np.asarray(jvp(v)))
        return np.column_stack(cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J2 = _jac_forward(lt_gamma2, x2)  # (T, n2)
        J1 = _jac_forward(lt_gamma1, Theta.ravel(order="F"))  # (T, Kq)

    boundary: list[str] = []
    # detect transition probabilities at the boundary (HL report "na")
    pf_at_bound = (pfree < _BOUNDARY_TOL) | (pfree > 1.0 - _BOUNDARY_TOL)
    n2 = x2.size
    keep2 = np.ones(n2, dtype=bool)
    off = nb + (M - 1) * K
    for k, bnd in enumerate(pf_at_bound):
        if bnd:
            keep2[off + k] = False
            i, j = k % (M - 1), k // (M - 1)
            boundary.append(f"p{i + 1}{j + 1}")

    def _opg_cov(J: np.ndarray, keep: np.ndarray) -> np.ndarray:
        n = J.shape[1]
        V = np.full((n, n), np.nan)
        Jk = J[:, keep]
        Sk = Jk.T @ Jk
        try:
            Vk = np.linalg.inv(Sk)
        except np.linalg.LinAlgError:
            Vk = np.linalg.pinv(Sk)
            warnings.warn("OPG matrix is singular; using pseudo-inverse", UserWarning)
        idx = np.where(keep)[0]
        V[np.ix_(idx, idx)] = Vk
        return V

    V2 = _opg_cov(J2, keep2)
    V1 = _opg_cov(J1, np.ones(J1.shape[1], dtype=bool))

    se2 = np.sqrt(np.abs(np.diag(V2)))
    se_theta = np.sqrt(np.abs(np.diag(V1))).reshape((K, q), order="F")

    se_B = np.full((K, K), np.nan)
    for k, (i, j) in enumerate(R.free_b_indices()):
        se_B[i, j] = se2[k]

    # delta method: lambda = exp(loglam) => se_lam = lam * se_loglam,
    # V_lambda = D V_loglam D with D = diag(lambda)
    nlam = (M - 1) * K
    lam_all = np.concatenate([np.asarray(l) for l in lams]) if nlam else np.array([])
    V_ll = V2[nb : nb + nlam, nb : nb + nlam]
    D = np.diag(lam_all)
    V_lambda = D @ V_ll @ D
    se_lambda = [
        lam_all[i * K : (i + 1) * K] * se2[nb + i * K : nb + (i + 1) * K] for i in range(M - 1)
    ]

    se_P = np.full((M, M), np.nan)
    seP_free = se2[off:]
    for j in range(M):
        for i in range(M - 1):
            se_P[i, j] = seP_free[j * (M - 1) + i]
    return OPGStandardErrors(
        se_theta=se_theta,
        se_B=se_B,
        se_lambda=se_lambda,
        V_lambda=V_lambda,
        se_P=se_P,
        boundary=boundary,
    )
