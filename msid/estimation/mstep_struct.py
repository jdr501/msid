"""Numerical M-step for (B, Lambda_2, ..., Lambda_M).

Minimizes HL's negative log-likelihood ``l(B, Lambda_2..Lambda_M)`` (HL 2014,
Appendix) over the *free* elements of B and ``log lambda_mi`` using exact
autograd gradients fed to L-BFGS-B -- no finite differences (Tether Online
Appendix A; spec Section 3.3).  Steps producing an ill-conditioned state
covariance (condition number above ``cond_limit``) are rejected.
"""

from __future__ import annotations

import autograd
import autograd.numpy as anp
import numpy as np
from scipy.optimize import minimize

from ..restrictions import Restrictions
from .likelihood import make_struct_objective

__all__ = ["update_struct"]


def _sigma_condition_ok(x: np.ndarray, R: Restrictions, M: int, cond_limit: float) -> bool:
    from .transforms import unpack_struct

    B, lams = unpack_struct(x, R, M)
    sigmas = [B @ B.T] + [B @ np.diag(l) @ B.T for l in lams]
    for S in sigmas:
        if not np.all(np.isfinite(S)):
            return False
        c = np.linalg.cond(S)
        if not np.isfinite(c) or c > cond_limit:
            return False
    return True


def update_struct(
    x0: np.ndarray,
    W: np.ndarray,
    Tm: np.ndarray,
    R: Restrictions,
    M: int,
    xi_penalty=None,
    cond_limit: float = 1e12,
    maxiter: int = 200,
    fix_lambda: bool = False,
) -> tuple[np.ndarray, bool]:
    """One structural M-step from warm start ``x0``.

    Parameters
    ----------
    x0 : packed vector ``[b_free, log lambda_2, ..., log lambda_M]``.
    W : (M, K, K) weighted moment matrices; Tm : (M,) weighted sample sizes.
    xi_penalty : optional callable ``penalty(B)`` for long-run zeros.
    fix_lambda : bool
        Bootstrap mode: optimize only over the free elements of B, holding
        the log-lambdas at their values in ``x0`` (spec Section 9, item 2).

    Returns
    -------
    (x_new, accepted) : the updated packed vector and whether the step was
    accepted (rejected steps leave the parameters unchanged).
    """
    nb = R.n_free_b
    full_obj = make_struct_objective(W, Tm, R, M, xi_penalty=xi_penalty)

    if fix_lambda:
        tail = np.asarray(x0[nb:], dtype=float)

        def obj(b):
            return full_obj(anp.concatenate([b, tail]))

        grad = autograd.grad(obj)
        res = minimize(
            obj,
            np.asarray(x0[:nb], float),
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        x1 = np.concatenate([res.x, tail])
    else:
        grad = autograd.grad(full_obj)
        res = minimize(
            full_obj,
            np.asarray(x0, float),
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        x1 = res.x

    if not np.all(np.isfinite(x1)) or not _sigma_condition_ok(x1, R, M, cond_limit):
        return np.asarray(x0, float), False
    # accept if it does not worsen the objective (L-BFGS-B can terminate
    # on a line-search failure with a slightly worse point)
    try:
        if full_obj(x1) > full_obj(np.asarray(x0, float)) + 1e-8:
            return np.asarray(x0, float), False
    except FloatingPointError:
        return np.asarray(x0, float), False
    return x1, True
