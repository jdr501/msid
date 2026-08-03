"""Regime-invariance tests for the impact matrix B (spec Section 6).

6.1  LR test for state-invariant B (M >= 3 only): fully unrestricted
     unstructured MS covariances versus the common-B decomposition, with
     df = M K (K+1)/2 - K^2 - (M-1) K  (HL 2014, Eq. 6; Lanne et al. 2010).
     Reproduces the "State-invariant B" row of HL Table 6.

6.2  Bootstrap overidentification J-test for common diagonalization
     (Tether Appendix F.1), generalized to any M >= 3:
     q = (M-2) K (K-1)/2 overidentifying restrictions from the extra
     regime covariances; Omega from a parametric bootstrap under the null.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg, stats

from ..estimation.em import _update_P, hamilton_filter, kim_smoother

__all__ = ["JTestResult", "LRInvarianceResult", "j_test_overidentification", "lr_state_invariance"]


@dataclass
class LRInvarianceResult:
    statistic: float
    df: int
    p_value: float
    loglik_unstructured: float
    loglik_common_b: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"LR test for state-invariant B: LR = {self.statistic:.3f}, "
            f"df = {self.df}, p = {self.p_value:.3f}"
        )


@dataclass
class JTestResult:
    statistic: float
    p_value: float
    q: int
    n_boot: int
    g: np.ndarray

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Overidentification J-test for constant B: J = {self.statistic:.2f}, "
            f"bootstrap p = {self.p_value:.2f} (q = {self.q}, {self.n_boot} reps)"
        )


def _em_unstructured(DY, Z, M, Theta0, sigmas0, P0, xi00, max_iter=500, tol=1e-8):
    """EM for the unstructured MS model (separate Sigma_m, no B decomposition).

    The M-step for the covariances is closed form: Sigma_m = W_m / T_m.
    """
    from ..estimation.mstep_theta import update_theta

    Theta, sigmas, P, xi0 = Theta0.copy(), [S.copy() for S in sigmas0], P0.copy(), xi00.copy()
    prev = -np.inf
    ll = -np.inf
    for _ in range(max_iter):
        U = DY - Z @ Theta.T
        logeta = _log_dens_sigmas(U, sigmas)
        xi_filt, _, ll = hamilton_filter(logeta, P, xi0)
        if not np.isfinite(ll):
            return -np.inf, None
        if np.isfinite(prev) and abs(ll - prev) / (abs(prev) + 1e-12) < tol:
            break
        prev = ll
        xi_smooth, joint = kim_smoother(xi_filt, P, xi0)
        P = _update_P(joint, xi_smooth)
        W = np.einsum("tm,ti,tj->mij", xi_smooth[1:], U, U)
        Tm = xi_smooth[1:].sum(axis=0)
        sigmas = [W[m] / max(Tm[m], 1e-8) for m in range(M)]
        Theta = update_theta(DY, Z, xi_smooth[1:], sigmas)
        xi0 = xi_smooth[0]
    return ll, sigmas


def _log_dens_sigmas(U, sigmas):
    T, K = U.shape
    cols = []
    for S in sigmas:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return np.full((T, len(sigmas)), -np.inf)
        iS = np.linalg.inv(S)
        q = np.einsum("ti,ij,tj->t", U, iS, U)
        cols.append(-0.5 * K * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * q)
    return np.column_stack(cols)


def lr_state_invariance(results) -> LRInvarianceResult:
    """LR test of the common-B decomposition against unstructured Sigma_m.

    Only defined for M >= 3: with two states the decomposition
    Sigma_1 = BB', Sigma_2 = B Lambda_2 B' is exactly identified and there
    is nothing to test (HL 2014, Section 3.1).
    """
    M, K = results.M, results.K
    if M < 3:
        raise ValueError(
            "the state-invariance LR test requires M >= 3; with M = 2 the "
            "common-B decomposition is exactly identified (HL 2014, Sec. 3.1)"
        )
    df = int(M * K * (K + 1) / 2 - K * K - (M - 1) * K)
    ll_u, _ = _em_unstructured(
        results._DY,
        results._Z,
        M,
        results.theta_,
        results.Sigma_,
        results.P_,
        results._xi0,
    )
    if ll_u < results.loglik_ - 1e-6:
        warnings.warn(
            "unstructured model likelihood below the common-B likelihood; "
            "the unstructured EM may have found a local optimum",
            UserWarning,
        )
    lr = max(2.0 * (ll_u - results.loglik_), 0.0)
    return LRInvarianceResult(
        statistic=lr,
        df=df,
        p_value=float(stats.chi2.sf(lr, df)),
        loglik_unstructured=ll_u,
        loglik_common_b=results.loglik_,
    )


def _b_from_pair(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """B from the baseline pair via generalized eigendecomposition.

    Solves ``S2 v = lambda S1 v`` with ``V' S1 V = I``; then ``B = (V')^{-1}``
    satisfies ``S1 = BB'`` and ``S2 = B diag(lambda) B'`` (Tether App. F.1).
    """
    _, V = linalg.eigh(S2, S1)
    return np.linalg.inv(V.T)


def _g_stat(B: np.ndarray, sigmas_extra: list[np.ndarray]) -> np.ndarray:
    iB = np.linalg.inv(B)
    gs = []
    for S in sigmas_extra:
        D = iB @ S @ iB.T
        K = D.shape[0]
        iu = np.triu_indices(K, k=1)
        gs.append(((D + D.T) / 2.0)[iu])
    return np.concatenate(gs)


def j_test_overidentification(
    results, n_boot: int = 3000, n_jobs: int = -1, random_state=None
) -> JTestResult:
    """Bootstrap overidentification J-test (Tether Appendix F.1), M >= 3.

    B_hat is estimated from the baseline pair (Sigma_1, Sigma_2); the
    stacked off-diagonals of ``B^{-1} Sigma_m B^{-1'}`` for m >= 3 give
    q = (M-2) K (K-1)/2 restrictions.  Omega is estimated by a parametric
    bootstrap under the null, simulating from
    ``Sigma^(0)_m = B diag(B^{-1} Sigma_m B^{-1'}) B'`` with regimes drawn
    from the smoothed probabilities (held fixed across replications), and
    the p-value is read off the bootstrap distribution of J*.
    """
    M, K = results.M, results.K
    if M < 3:
        raise ValueError(
            "the J-test needs M >= 3 (with M = 2 there are no " "overidentifying restrictions)"
        )
    q = (M - 2) * K * (K - 1) // 2
    xi = results.smoothed_probs_.to_numpy()
    U = results.residuals_.to_numpy()
    T = U.shape[0]
    Tm = xi.sum(axis=0)
    sig_hat = [np.einsum("t,ti,tj->ij", xi[:, m], U, U) / max(Tm[m], 1e-8) for m in range(M)]
    B = _b_from_pair(sig_hat[0], sig_hat[1])
    g = _g_stat(B, sig_hat[2:])

    # null-implied covariances
    sig0 = []
    iB = np.linalg.inv(B)
    for S in sig_hat:
        sig0.append(B @ np.diag(np.diag(iB @ S @ iB.T)) @ B.T)
    chols = [np.linalg.cholesky(S) for S in sig0]

    ss = np.random.SeedSequence(random_state)
    children = ss.spawn(n_boot)

    def _one(child):
        rng = np.random.default_rng(child)
        # draw regimes from the smoothed probabilities, then u* ~ N(0, Sigma0_m)
        cdf = np.cumsum(xi, axis=1)
        u = rng.random(T)
        regimes = (u[:, None] > cdf).sum(axis=1)
        eps = rng.standard_normal((T, K))
        Ub = np.empty((T, K))
        for m in range(M):
            mask = regimes == m
            Ub[mask] = eps[mask] @ chols[m].T
        sig_b = [np.einsum("t,ti,tj->ij", xi[:, m], Ub, Ub) / max(Tm[m], 1e-8) for m in range(M)]
        try:
            Bb = _b_from_pair(sig_b[0], sig_b[1])
            return _g_stat(Bb, sig_b[2:])
        except np.linalg.LinAlgError:
            return None

    gstars = Parallel(n_jobs=n_jobs)(delayed(_one)(c) for c in children)
    gstars = np.array([gs for gs in gstars if gs is not None])
    Omega = T * np.cov(gstars.T, ddof=1).reshape(q, q)
    iOm = np.linalg.pinv(Omega)
    J = float(T * g @ iOm @ g)
    gc = gstars - gstars.mean(axis=0)
    Jstars = T * np.einsum("bi,ij,bj->b", gc, iOm, gc)
    p = float((Jstars >= J).mean())
    return JTestResult(statistic=J, p_value=p, q=q, n_boot=len(gstars), g=g)
