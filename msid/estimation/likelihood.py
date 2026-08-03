"""Regime-conditional Gaussian log-likelihoods (autograd-differentiable).

Implements

* the M-step objective ``l(B, Lambda_2, ..., Lambda_M)`` of HL (2014,
  Appendix, "Estimate B and Lambda_m") in the log-lambda parameterization of
  the Tether paper's Online Appendix A, and
* the per-observation filtered log-likelihood contributions
  ``l_t = log(xi'_{t|t-1} eta_t)`` (HL 2014, Appendix, "Likelihood function")
  used for outer-product-of-gradients standard errors.

All functions are written with :mod:`autograd.numpy` so that exact reverse-
mode gradients are available -- no finite differences anywhere (spec 3.3).
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np

from ..restrictions import Restrictions

__all__ = [
    "LOG2PI",
    "filter_loglik_contributions",
    "make_struct_objective",
    "neg_loglik_struct",
]

LOG2PI = float(np.log(2.0 * np.pi))


def neg_loglik_struct(B, lam_list, W, Tm, xi_penalty=None):
    """HL's M-step objective l(B, Lambda_2, ..., Lambda_M).

    Parameters
    ----------
    B : (K, K) array (autograd-traceable)
    lam_list : list of (K,) arrays
        Diagonals of Lambda_2, ..., Lambda_M (positive).
    W : (M, K, K) array
        State-weighted residual moment matrices ``W_m = sum_t xi_mt|T u_t u_t'``.
    Tm : (M,) array
        Weighted sample sizes ``T_m = sum_t xi_mt|T``; ``sum(Tm) = T``.
    xi_penalty : callable or None
        Optional penalty term ``xi_penalty(B)`` added to the objective (used
        to enforce long-run zeros; see :mod:`msid.restrictions`).

    Notes
    -----
    l(B, L2..LM) = T log|det B| + 1/2 tr(B'^{-1} B^{-1} W_1)
                   + sum_m [ T_m/2 log det(Lambda_m)
                             + 1/2 tr(B'^{-1} Lambda_m^{-1} B^{-1} W_m) ]
    (HL 2014, Appendix).
    """
    T = anp.sum(Tm)
    iB = anp.linalg.inv(B)
    _, logabsdet = anp.linalg.slogdet(B)
    val = T * logabsdet
    S1 = iB @ W[0] @ iB.T
    val = val + 0.5 * anp.trace(S1)
    for m, lam in enumerate(lam_list, start=1):
        Sm = iB @ W[m] @ iB.T
        val = val + 0.5 * Tm[m] * anp.sum(anp.log(lam))
        val = val + 0.5 * anp.sum(anp.diag(Sm) / lam)
    if xi_penalty is not None:
        val = val + xi_penalty(B)
    return val


def make_struct_objective(W, Tm, R: Restrictions, M: int, xi_penalty=None):
    """Return ``f(x)`` over the packed vector [b_free, log-lambdas].

    The returned callable is autograd-differentiable; ``x`` follows the
    layout of :func:`msid.estimation.transforms.pack_struct`.
    """
    K = R.K
    S = R.selection_matrix()
    nb = S.shape[1]
    W = anp.asarray(W)
    Tm = anp.asarray(Tm)

    def objective(x):
        B = anp.reshape(S @ x[:nb], (K, K), order="F")
        lam_list = [anp.exp(x[nb + i * K : nb + (i + 1) * K]) for i in range(M - 1)]
        return neg_loglik_struct(B, lam_list, W, Tm, xi_penalty=xi_penalty)

    return objective


def _log_state_densities(U, B, lam_list):
    """(T, M) matrix of log conditional densities log f(y_t | s_t = m)."""
    _, K = U.shape
    iB = anp.linalg.inv(B)
    _, logdetB = anp.linalg.slogdet(B)
    E = U @ iB.T  # structural residuals eps_t' rows
    cols = []
    # state 1: Sigma_1 = BB'
    q1 = anp.sum(E * E, axis=1)
    cols.append(-0.5 * K * LOG2PI - logdetB - 0.5 * q1)
    for lam in lam_list:
        qm = anp.sum((E * E) / lam[None, :], axis=1)
        logdet = 2.0 * logdetB + anp.sum(anp.log(lam))
        cols.append(-0.5 * K * LOG2PI - 0.5 * logdet - 0.5 * qm)
    return anp.stack(cols, axis=1)


def filter_loglik_contributions(U, B, lam_list, P, xi0):
    """Per-observation log-likelihood contributions from the Hamilton filter.

    Runs the filter ``xi_{t|t-1} = P xi_{t-1|t-1}`` in an autograd-friendly
    way using the log-sum-exp trick, and returns the length-``T`` vector of
    ``l_t = log(xi'_{t|t-1} eta_t)`` (HL 2014, Appendix, likelihood
    evaluation with *given*, not smoothed, parameters).

    Parameters
    ----------
    U : (T, K) reduced-form residuals.
    B : (K, K); lam_list : list of (K,) diagonals of Lambda_m, m >= 2.
    P : (M, M) transition matrix, columns sum to one.
    xi0 : (M,) initial state probabilities ``xi_{0|0}``.
    """
    logeta = _log_state_densities(U, B, lam_list)
    T = U.shape[0]
    lt = []
    xi = xi0
    for t in range(T):
        pred = P @ xi
        w = logeta[t]
        c = anp.max(w)
        et = anp.exp(w - c)
        denom = anp.dot(pred, et)
        lt.append(anp.log(denom) + c)
        xi = (pred * et) / denom
    return anp.stack(lt)
