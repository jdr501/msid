"""EM driver: E-step (Hamilton filter + Kim smoother), M-step orchestration.

Implements the algorithm of HL (2014, Appendix) with the modifications of
the Tether paper's Online Appendix A (Algorithm 1):

* log-density arithmetic with the log-sum-exp trick in the filter,
* log-parameterized Lambda diagonals optimized with autograd gradients,
* randomized Lambda initializations and a multi-start manager,
* closed-form xi-weighted GLS update for theta,
* column-rescaling guard on the transition-matrix update.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..restrictions import Restrictions
from .likelihood import LOG2PI
from .mstep_struct import update_struct
from .mstep_theta import update_theta
from .transforms import pack_struct, unpack_struct

__all__ = ["EMConfig", "EMState", "em_estimate", "hamilton_filter", "kim_smoother", "run_em"]


@dataclass
class EMConfig:
    """Tuning knobs for the EM iterations (defaults from the handoff spec)."""

    max_iter: int = 500
    tol_ll: float = 1e-8
    tol_param: float = 1e-6
    label_order: str = "lambda_sort"  # or "terminal_prob"
    cond_limit: float = 1e12
    xi_tol: float = 1e-8
    xi_penalty_w0: float = 1e4
    xi_penalty_growth: float = 10.0
    xi_penalty_wmax: float = 1e12
    struct_maxiter: int = 200


@dataclass
class EMState:
    """Current parameter values inside the EM loop."""

    Theta: np.ndarray  # (K, q) slope parameters
    B: np.ndarray  # (K, K)
    lams: list[np.ndarray]  # diagonals of Lambda_2..Lambda_M
    P: np.ndarray  # (M, M), columns sum to 1
    xi0: np.ndarray  # (M,)
    loglik: float = -np.inf
    converged: bool = False
    n_iter: int = 0
    smoothed: np.ndarray | None = None  # (T+1, M) incl. t=0
    filtered: np.ndarray | None = None  # (T, M)
    history: list = field(default_factory=list)

    def sigmas(self) -> list[np.ndarray]:
        out = [self.B @ self.B.T]
        for lam in self.lams:
            out.append(self.B @ np.diag(lam) @ self.B.T)
        return out


def _log_state_densities(U: np.ndarray, B: np.ndarray, lams: list[np.ndarray]) -> np.ndarray:
    """(T, M) log Gaussian densities log f(y_t | s_t = m, Y_{t-1})."""
    _, K = U.shape
    iB = np.linalg.inv(B)
    _, logdetB = np.linalg.slogdet(B)
    E = U @ iB.T
    cols = [-0.5 * K * LOG2PI - logdetB - 0.5 * np.sum(E * E, axis=1)]
    for lam in lams:
        logdet = 2.0 * logdetB + np.sum(np.log(lam))
        cols.append(-0.5 * K * LOG2PI - 0.5 * logdet - 0.5 * np.sum(E * E / lam[None, :], axis=1))
    return np.column_stack(cols)


def hamilton_filter(
    logeta: np.ndarray, P: np.ndarray, xi0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward filter in log-density space (spec Section 3.2).

    Returns ``(xi_filt (T, M), xi_pred (T, M), loglik)`` where
    ``loglik = sum_t log(xi'_{t|t-1} eta_t)`` (HL 2014, Appendix).
    """
    T, M = logeta.shape
    xi_filt = np.empty((T, M))
    xi_pred = np.empty((T, M))
    xi = np.asarray(xi0, dtype=float)
    ll = 0.0
    for t in range(T):
        pred = P @ xi
        w = logeta[t]
        c = w.max()
        et = np.exp(w - c)
        denom = pred @ et
        if not np.isfinite(denom) or denom <= 0.0:
            return xi_filt, xi_pred, -np.inf
        ll += np.log(denom) + c
        xi = (pred * et) / denom
        xi_pred[t] = pred
        xi_filt[t] = xi
    return xi_filt, xi_pred, ll


def kim_smoother(
    xi_filt: np.ndarray, P: np.ndarray, xi0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backward smoother (HL 2014, Appendix; Tether Online App. A).

    Returns ``(xi_smooth (T+1, M), joint (T, M, M))`` where
    ``xi_smooth[0]`` is ``xi_{0|T}`` and ``joint[t][i, j]`` is the pairwise
    smoothed probability ``Pr(s_{t+1}=i, s_t=j | Y_T)`` built from the
    ``xi^{(2)}_{t|T}`` formula.
    """
    T, M = xi_filt.shape
    eps = 1e-300
    xi_all = np.vstack([xi0[None, :], xi_filt])  # index t = 0..T
    xi_smooth = np.empty((T + 1, M))
    joint = np.empty((T, M, M))
    xi_smooth[T] = xi_filt[T - 1]
    for t in range(T - 1, -1, -1):
        pred = P @ xi_all[t]
        ratio = xi_smooth[t + 1] / np.maximum(pred, eps)
        xi_smooth[t] = (P.T @ ratio) * xi_all[t]
        joint[t] = P * np.outer(ratio, xi_all[t])
        s = xi_smooth[t].sum()
        if s > 0:
            xi_smooth[t] /= s
    return xi_smooth, joint


def _update_P(joint: np.ndarray, xi_smooth: np.ndarray) -> np.ndarray:
    """Closed-form transition matrix update with column-rescaling guard."""
    num = joint.sum(axis=0)  # [i, j]: into i, out of j
    den = xi_smooth[1:].sum(axis=0)  # sum_{t=1..T} xi_{t|T}
    P = num / np.maximum(den[None, :], 1e-300)
    colsum = P.sum(axis=0)
    colsum[colsum <= 0] = 1.0
    return P / colsum[None, :]  # rescale exactly (App. A guard)


def _relabel(state: EMState, xi_smooth: np.ndarray, order: str) -> EMState:
    """Apply the label-switching rule; renormalize if state 1 changes.

    ``"lambda_sort"`` orders states by total state variance tr(Sigma_m) =
    tr(B Lambda_m B') -- a deterministic rule that is invariant to which
    regime the EM run happened to use as the Lambda_1 = I baseline (robust
    for M > 2); ``"terminal_prob"`` enforces HL's xi_{iT|T} <= xi_{jT|T}
    for i < j.  When the new state 1 is not the old state 1, the
    normalization Lambda_1 = I is restored via B <- B diag(Lambda_new1)^{1/2}.
    """
    M = state.P.shape[0]
    K = state.B.shape[0]
    lam_full = [np.ones(K)] + [np.asarray(l) for l in state.lams]
    if order == "terminal_prob":
        perm = np.argsort(xi_smooth[-1], kind="stable")
    else:
        traces = [float(np.trace(state.B @ np.diag(l) @ state.B.T)) for l in lam_full]
        perm = np.argsort(traces, kind="stable")
    if np.array_equal(perm, np.arange(M)):
        return state
    base = lam_full[perm[0]]
    newB = state.B @ np.diag(np.sqrt(base))
    new_lams = [lam_full[perm[m]] / base for m in range(1, M)]
    newP = state.P[np.ix_(perm, perm)]
    state.B = newB
    state.lams = new_lams
    state.P = newP
    state.xi0 = state.xi0[perm]
    return state


def _pack_all(state: EMState) -> np.ndarray:
    return np.concatenate(
        [state.Theta.ravel(), state.B.ravel()]
        + [np.asarray(l) for l in state.lams]
        + [state.P.ravel(), state.xi0]
    )


def run_em(
    DY: np.ndarray,
    Z: np.ndarray,
    R: Restrictions,
    M: int,
    state: EMState,
    config: EMConfig,
    C_longrun=None,
    fix_lambda_P: bool = False,
) -> EMState:
    """Iterate E/M steps until convergence from the given starting state.

    Parameters
    ----------
    DY, Z : design matrices with ``DY_t = Theta @ Z_t + u_t``.
    C_longrun : callable or None
        ``C_longrun(Theta) -> (K, K)`` matrix with ``Xi = C @ B`` for the
        current slope parameters; required when ``R`` has Xi zeros.
    fix_lambda_P : bool
        Bootstrap mode (spec Section 9): hold Lambda_m and P at their
        current values, update only theta and (free elements of) B.
    """
    w_xi = config.xi_penalty_w0
    prev_ll = -np.inf
    prev_par = _pack_all(state)
    for it in range(1, config.max_iter + 1):
        U = DY - Z @ state.Theta.T
        logeta = _log_state_densities(U, state.B, state.lams)
        xi_filt, _, ll = hamilton_filter(logeta, state.P, state.xi0)
        if not np.isfinite(ll):
            warnings.warn("likelihood became non-finite; stopping this EM run", UserWarning)
            state.loglik = -np.inf
            state.converged = False
            state.n_iter = it
            return state
        xi_smooth, joint = kim_smoother(xi_filt, state.P, state.xi0)

        # convergence check (both criteria, spec Section 3.4)
        par = _pack_all(state)
        rel_ll = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12) if np.isfinite(prev_ll) else np.inf
        # scale-aware hybrid: absolute for near-zero parameters (tiny
        # cross-lag coefficients jitter at noise level and never pass a pure
        # relative test), relative for large ones (e.g. lambda ~ 700)
        rel_par = np.max(np.abs(par - prev_par) / (1.0 + np.abs(prev_par)))
        state.loglik = ll
        state.history.append(ll)
        state.smoothed = xi_smooth
        state.filtered = xi_filt
        state.n_iter = it
        if it > 1 and rel_ll < config.tol_ll and rel_par < config.tol_param:
            state.converged = True
            break
        prev_ll, prev_par = ll, par

        # ---- M step ----
        if not fix_lambda_P:
            state.P = _update_P(joint, xi_smooth)
        Wm = np.einsum("tm,ti,tj->mij", xi_smooth[1:], U, U)
        Tm = xi_smooth[1:].sum(axis=0)

        xi_pen = None
        if R.has_xi_restrictions:
            if C_longrun is None:
                raise ValueError("Xi restrictions require a long-run map C(theta)")
            C = C_longrun(state.Theta)
            idx = R.xi_zero_indices
            wloc = w_xi

            def xi_pen(B, C=C, idx=idx, w=wloc):
                Xi = C @ B
                return w * sum(Xi[i, j] ** 2 for (i, j) in idx)

        x0 = pack_struct(state.B, [np.diag(l) for l in state.lams], R)
        x1, ok = update_struct(
            x0,
            Wm,
            Tm,
            R,
            M,
            xi_penalty=xi_pen,
            cond_limit=config.cond_limit,
            maxiter=config.struct_maxiter,
            fix_lambda=fix_lambda_P,
        )
        if ok:
            state.B, new_lams = unpack_struct(x1, R, M)
            if not fix_lambda_P:
                state.lams = new_lams
        state.B = R.normalize_signs(state.B)

        state.Theta = update_theta(DY, Z, xi_smooth[1:], state.sigmas())
        state.xi0 = xi_smooth[0]
        if not fix_lambda_P:
            state = _relabel(state, xi_smooth, config.label_order)
        if R.has_xi_restrictions:
            w_xi = min(w_xi * config.xi_penalty_growth, config.xi_penalty_wmax)
    else:
        warnings.warn(
            f"EM did not converge in {config.max_iter} iterations "
            f"(rel. loglik change {rel_ll:.2e}, rel. param change {rel_par:.2e})",
            UserWarning,
        )
        state.converged = False

    # verify long-run restrictions at convergence (spec Section 4)
    if R.has_xi_restrictions and C_longrun is not None:
        Xi = C_longrun(state.Theta) @ state.B
        viol = max(abs(Xi[i, j]) for (i, j) in R.xi_zero_indices)
        if viol > max(config.xi_tol, 1e-6) * max(1.0, np.abs(Xi).max()):
            raise RuntimeError(
                f"long-run zero restrictions violated at convergence "
                f"(max |Xi_restricted| = {viol:.3e}); increase xi_penalty_wmax"
            )
    return state


def em_estimate(
    DY: np.ndarray,
    Z: np.ndarray,
    R: Restrictions,
    M: int,
    starts: list[EMState],
    config: EMConfig,
    C_longrun=None,
    n_jobs: int = -1,
) -> tuple[EMState, np.ndarray]:
    """Multi-start manager: run EM from every start, keep the best.

    Returns ``(best_state, loglik_starts)`` where ``loglik_starts`` collects
    every converged log-likelihood so the local-optima landscape can be
    inspected (``results.loglik_starts_``).
    """
    from joblib import Parallel, delayed

    def _one(st: EMState) -> EMState:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return run_em(DY, Z, R, M, st, config, C_longrun=C_longrun)
            except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
                st.loglik = -np.inf
                return st

    if len(starts) == 1:
        results = [run_em(DY, Z, R, M, starts[0], config, C_longrun=C_longrun)]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(st) for st in starts)
    logliks = np.array([r.loglik for r in results])
    if not np.isfinite(logliks).any():
        raise RuntimeError("all EM starts failed; try more starts or rescale the data")
    best = results[int(np.nanargmax(logliks))]
    return best, logliks
