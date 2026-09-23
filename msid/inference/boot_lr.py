"""Bootstrap null distributions for likelihood-ratio tests of restrictions on B.

:func:`msid.compare_models` refers ``LR = 2 (log L_1 - log L_0)`` to a
chi-squared distribution.  That reference is asymptotically correct for the
interior zero restrictions of :mod:`msid.restrictions`: B is identified from the
variance shift, so the restrictions are overidentifying rather than identifying,
and the restricted values are not on a boundary of the parameter space.  In
finite samples with regime-dependent volatility the chi-squared quantiles can
still be off, and a *non*-rejection carries little weight unless the test is
known to be correctly sized.  This module simulates the null instead.

The null data-generating process is a fixed-design wild bootstrap carried out in
**structural** space.  Writing ``u0`` for the restricted model's residuals and
``B0`` for its impact matrix,

    eps0 = u0 B0^{-1}'          DY* = Z Theta0' + (psi .* eps0) B0'

with ``psi`` Rademacher, drawn independently per element.  Flipping the
reduced-form residuals directly does **not** impose the null: their sample
second moments are a property of the data rather than of the restricted model,
so each draw retains whatever the restriction sets to zero and the bootstrap LR
merely reproduces the observed one.  Rotating through ``B0`` forces
``Cov(u*) = B0 diag(.) B0'``, which satisfies the restriction by construction,
while the empirical distribution of the structural shocks and its
regime-dependent scale are preserved.  Nothing is assumed Gaussian -- which
matters when the data carry the heavy tails these models are usually fitted to.

Each replication warm-starts both models at the null parameters.  The restricted
optimum is a feasible point of the unrestricted parameter space and the EM is
monotone in the likelihood, so this makes ``LR* >= 0`` by construction.  The
same device is applied to the observed data when ``polish=True``: the
unrestricted fit is re-run from the restricted solution.  That removes the
nesting violations multi-start EM otherwise produces when a restricted model's
inner optimisation is better conditioned than its parent's -- pinning a nearly
flat direction can let the restricted fit converge *further* and report the
higher likelihood.

The warm start isolates the chi-squared approximation for a well-behaved
optimiser.  It does not reproduce a multi-start search, so it says nothing about
how often that search lands in a bad optimum; raise ``n_extra_starts`` to probe
that, at proportional cost.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from ..estimation.em import EMConfig, EMState, run_em

__all__ = ["BootLRResult", "bootstrap_lr_test", "polish_from"]

_FAILURES = (np.linalg.LinAlgError, RuntimeError, ValueError)


@dataclass
class BootLRResult:
    """Outcome of a bootstrap LR test of a restricted model."""

    statistic: float
    df: int
    p_value: float
    p_chi2: float
    n_boot: int
    n_ok: int
    loglik_unrestricted: float
    loglik_restricted: float
    crit: dict = field(default_factory=dict)
    chi2_crit: dict = field(default_factory=dict)
    draws: np.ndarray | None = None
    polish_gain: float = 0.0
    h0: str = ""

    @property
    def rejected(self) -> bool:
        """Rejection at the 5% bootstrap level."""
        return self.p_value < 0.05

    @property
    def size_ratio(self) -> float:
        """Bootstrap over chi-squared 95% critical value.

        Above one means the chi-squared reference is liberal here, so the
        asymptotic test over-rejects and a non-rejection is more secure than its
        chi-squared p-value suggests.  Below one means the reverse.
        """
        ref = self.chi2_crit.get(95, np.nan)
        return float(self.crit.get(95, np.nan) / ref) if ref else float("nan")

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Bootstrap LR test of {self.h0 or 'the restriction'}: "
            f"LR = {self.statistic:.4f} (df = {self.df}), bootstrap p = {self.p_value:.4f} "
            f"from {self.n_ok}/{self.n_boot} draws; chi2 p = {self.p_chi2:.4f}. "
            f"crit95 bootstrap {self.crit.get(95, float('nan')):.3f} against chi2 "
            f"{self.chi2_crit.get(95, float('nan')):.3f} "
            f"(ratio {self.size_ratio:.2f})"
        )


def _state_of(res) -> EMState:
    """An EM starting state carrying a fitted model's parameters."""
    return EMState(
        Theta=res.theta_.copy(),
        B=res.B_.copy(),
        lams=[np.asarray(lam).copy() for lam in res.Lambda_[1:]],
        P=res.P_.copy(),
        xi0=res._xi0.copy(),
    )


def _copy_state(state: EMState) -> EMState:
    return EMState(
        Theta=state.Theta.copy(),
        B=state.B.copy(),
        lams=[lam.copy() for lam in state.lams],
        P=state.P.copy(),
        xi0=state.xi0.copy(),
    )


def _config(max_iter: int) -> EMConfig:
    return EMConfig(max_iter=max_iter, tol_ll=1e-8, tol_param=1e-6, struct_maxiter=200)


def _longrun(model, R):
    return model.longrun_map if R.has_xi_restrictions else None


def _run(DY, Z, R, M, state, config, C_longrun):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = run_em(DY, Z, R, M, state, config, C_longrun=C_longrun)
        except _FAILURES:
            return None
    return out if np.isfinite(out.loglik) else None


def polish_from(target, source, max_iter: int = 500):
    """Re-run ``target``'s EM warm-started at ``source``'s solution.

    ``source`` must be at least as restricted as ``target``, so that its
    parameter point is feasible for ``target``.  Returns the new log-likelihood
    and the terminal state, or ``(target.loglik_, None)`` when nothing improves.

    This exists because a restricted fit can report a *higher* likelihood than
    the model nesting it: removing a nearly flat direction from B conditions the
    inner optimisation better, so the restricted EM converges further.  Starting
    the parent from the child's solution cannot do worse, the EM being monotone.
    """
    R = target.restrictions
    state = _state_of(source)
    state.B = R.unpack_b(R.pack_b(state.B))
    out = _run(
        target._DY,
        target._Z,
        R,
        target.M,
        state,
        _config(max_iter),
        _longrun(target.model, R),
    )
    if out is None or out.loglik <= target.loglik_ + 1e-10:
        return float(target.loglik_), None
    return float(out.loglik), out


def _one_draw(child, DY, Z, M, R_res, R_un, state0, fitted0, eps0, B0, cfg, C_res, C_un, n_extra):
    """One LR draw from the null."""
    rng = np.random.default_rng(child)
    psi = rng.integers(0, 2, size=eps0.shape) * 2.0 - 1.0
    DYb = fitted0 + (psi * eps0) @ B0.T

    def best(R, C):
        starts = [_copy_state(state0)]
        for _ in range(n_extra):
            s = _copy_state(state0)
            scale = 0.05 * np.abs(s.B).max()
            s.B = R.unpack_b(R.pack_b(s.B + scale * rng.standard_normal(s.B.shape)))
            starts.append(s)
        lls = [out.loglik for out in (_run(DYb, Z, R, M, s, cfg, C) for s in starts) if out]
        return max(lls) if lls else None

    ll_un, ll_res = best(R_un, C_un), best(R_res, C_res)
    if ll_un is None or ll_res is None:
        return None
    return 2.0 * (ll_un - ll_res)


def _check_nested(unrestricted, restricted) -> int:
    """Validate nesting and return the degrees of freedom."""
    if (
        unrestricted.model is not restricted.model
        and unrestricted._DY.shape != restricted._DY.shape
    ):
        raise ValueError("the two fits are not on the same data")
    Ru, Rr = unrestricted.restrictions, restricted.restrictions
    if not set(Ru.b_zero_indices) <= set(Rr.b_zero_indices):
        raise ValueError(
            "the restricted fit must impose every zero the unrestricted one does; "
            f"got b-zeros {Rr.b_zero_indices} against {Ru.b_zero_indices}"
        )
    if not set(Ru.xi_zero_indices) <= set(Rr.xi_zero_indices):
        raise ValueError("the restricted fit must impose every Xi zero the unrestricted one does")
    df = Rr.n_zero_restrictions - Ru.n_zero_restrictions
    if df <= 0:
        raise ValueError(
            "the two fits impose the same restrictions, so there is nothing to test "
            "(sign restrictions carry zero degrees of freedom)"
        )
    return df


def bootstrap_lr_test(
    unrestricted,
    restricted,
    n_boot: int = 499,
    max_iter: int = 400,
    n_extra_starts: int = 0,
    polish: bool = True,
    df_override: int | None = None,
    n_jobs: int = -1,
    random_state=None,
) -> BootLRResult:
    """LR test of ``restricted`` against ``unrestricted`` with a simulated null.

    Parameters
    ----------
    unrestricted, restricted : results objects
        Fits of the same model on the same data, with ``restricted`` imposing a
        superset of ``unrestricted``'s zero restrictions.
    n_boot : int
        Null replications.  The p-value is ``(1 + #{LR* >= LR}) / (n_ok + 1)``,
        so its floor is ``1 / (n_ok + 1)``; 499 gives a floor of 0.002.
    max_iter : int
        EM cap per replication.  Warm starts converge in far fewer iterations
        than a cold multi-start fit, so this is a safety cap rather than a
        target.
    n_extra_starts : int
        Random restarts per replication in addition to the warm start.  Zero
        tests the chi-squared approximation itself; larger values also pick up
        optimiser variability, at proportional cost.
    polish : bool
        Re-run the unrestricted fit from the restricted solution first, so the
        observed ``LR >= 0``.  See :func:`polish_from`.
    df_override : int, optional
        Override the automatic degrees of freedom (number of zero restrictions
        in H0 minus those in H1).  The chi-squared column uses this; the
        bootstrap p-value does not depend on it.

    Returns
    -------
    BootLRResult
    """
    df = df_override if df_override is not None else _check_nested(unrestricted, restricted)
    model, M = restricted.model, restricted.M
    DY, Z = restricted._DY, restricted._Z
    R_res, R_un = restricted.restrictions, unrestricted.restrictions

    ll_un = float(unrestricted.loglik_)
    gain = 0.0
    if polish:
        polished, _ = polish_from(unrestricted, restricted, max_iter=max_iter)
        gain = polished - ll_un
        ll_un = polished
    ll_res = float(restricted.loglik_)
    lr_obs = 2.0 * (ll_un - ll_res)

    B0 = restricted.B_.copy()
    fitted0 = Z @ restricted.theta_.T
    eps0 = (DY - fitted0) @ np.linalg.inv(B0).T
    state0 = _state_of(restricted)
    cfg = _config(max_iter)
    C_res, C_un = _longrun(model, R_res), _longrun(model, R_un)

    children = np.random.SeedSequence(random_state).spawn(n_boot)
    raw = Parallel(n_jobs=n_jobs)(
        delayed(_one_draw)(
            c, DY, Z, M, R_res, R_un, state0, fitted0, eps0, B0, cfg, C_res, C_un, n_extra_starts
        )
        for c in children
    )
    draws = np.array([d for d in raw if d is not None and np.isfinite(d)], dtype=float)
    if draws.size < 10:
        raise RuntimeError(f"only {draws.size} usable bootstrap draws; raise n_boot")

    levels = (90, 95, 99)
    crit = {q: float(np.percentile(draws, q)) for q in levels}
    chi2_crit = {q: float(stats.chi2.ppf(q / 100.0, df)) for q in levels}
    p_boot = float((1.0 + np.sum(draws >= lr_obs)) / (draws.size + 1.0))

    added = sorted(set(R_res.b_zero_indices) - set(R_un.b_zero_indices))
    parts = [f"b{i + 1}{j + 1}=0" for (i, j) in added]
    return BootLRResult(
        statistic=lr_obs,
        df=int(df),
        p_value=p_boot,
        p_chi2=float(stats.chi2.sf(lr_obs, df)),
        n_boot=int(n_boot),
        n_ok=int(draws.size),
        loglik_unrestricted=ll_un,
        loglik_restricted=ll_res,
        crit=crit,
        chi2_crit=chi2_crit,
        draws=draws,
        polish_gain=float(gain),
        h0=", ".join(parts) if parts else "the restriction",
    )
