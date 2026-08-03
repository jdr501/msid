"""Results containers for fitted MSVAR / MSVECM models.

Trailing-underscore attributes are estimated quantities (spec Section 10).
Standard errors are computed lazily (first access of ``std_errors_`` or
``summary()``) via the OPG estimator of HL's appendix.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

from .restrictions import Restrictions

__all__ = ["MSVARResults", "MSVECMResults"]


class _ResultsBase:
    def __init__(self, model, state, R: Restrictions, DY, Z, index, logliks, config):
        self.model = model
        self.restrictions = R
        self.K = model.K
        self.M = model.M
        self._DY = DY
        self._Z = Z
        self._index = index
        self._config = config
        self._xi0 = state.xi0

        self.theta_ = state.Theta
        self.B_ = state.B
        self.Lambda_ = [np.ones(self.K)] + [np.asarray(l) for l in state.lams]
        self.P_ = state.P
        self.Sigma_ = state.sigmas()
        self.loglik_ = state.loglik
        self.converged_ = state.converged
        self.n_iter_ = state.n_iter
        self.loglik_starts_ = np.asarray(logliks)
        self.loglik_history_ = np.asarray(state.history)

        T = DY.shape[0]
        self.nobs_ = T
        k_struct = R.n_free_b - len(R.xi_zero_indices)
        self.n_free_params_ = (
            self.theta_.size + k_struct + (self.M - 1) * self.K + self.M * (self.M - 1)
        )
        self.aic_ = -2.0 * self.loglik_ + 2.0 * self.n_free_params_
        self.sc_ = -2.0 * self.loglik_ + np.log(T) * self.n_free_params_

        cols = [f"regime {m + 1}" for m in range(self.M)]
        self.smoothed_probs_ = pd.DataFrame(state.smoothed[1:], index=index, columns=cols)
        self.filtered_probs_ = pd.DataFrame(state.filtered, index=index, columns=cols)

        U = DY - Z @ self.theta_.T
        names = list(model.y.columns)
        self.residuals_ = pd.DataFrame(U, index=index, columns=names)
        eps = U @ np.linalg.inv(self.B_).T
        self.structural_shocks_ = pd.DataFrame(
            eps, index=index, columns=[f"eps{j + 1}" for j in range(self.K)]
        )
        self._se = None
        self._ident = None

    # ------------------------------------------------------------ lazy SEs
    @property
    def std_errors_(self):
        """OPG standard errors (HL 2014, Appendix); computed on first access."""
        if self._se is None:
            from .inference.std_errors import compute_opg_se

            self._se = compute_opg_se(
                self._DY,
                self._Z,
                self.restrictions,
                self.B_,
                self.Lambda_[1:],
                self.P_,
                self._xi0,
                self.theta_,
            )
        return self._se

    @property
    def identification_(self):
        """Automatic identification diagnostics (spec Section 5)."""
        if self._ident is None:
            from .inference.wald_lambda import wald_lambda_tests

            self._ident = wald_lambda_tests(self.Lambda_[1:], self.std_errors_.V_lambda)
        return self._ident

    @property
    def Xi_(self) -> np.ndarray:
        """Long-run effect matrix Xi (HL 2014, Eq. 2 / spec 1.2)."""
        return self.model.longrun_map(self.theta_) @ self.B_

    # ------------------------------------------------------------- methods
    def irf(
        self,
        horizon: int = 30,
        ci: float = 0.68,
        n_boot: int = 1000,
        cumulate: str = "levels",
        n_jobs: int = -1,
        random_state=None,
    ):
        """Structural IRFs with fixed-design wild bootstrap bands (spec 9).

        Set ``n_boot=0`` for point estimates only.
        """
        from .inference.bootstrap import bootstrap_irf
        from .irf import IRFResults, point_irf

        irfs = point_irf(self.model, self.theta_, self.B_, horizon, cumulate=cumulate)
        names = list(self.model.y.columns)
        if n_boot and n_boot > 0:
            draws, lo, hi = bootstrap_irf(
                self,
                horizon=horizon,
                n_boot=n_boot,
                ci=ci,
                cumulate=cumulate,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            return IRFResults(irfs=irfs, var_names=names, lo=lo, hi=hi, draws=draws, ci=ci)
        return IRFResults(irfs=irfs, var_names=names)

    def fevd(self, horizon: int = 24, by_state: bool = True) -> pd.DataFrame:
        """Conditional-on-state FEVDs (HL Table 8 layout)."""
        from .irf import fevd as _fevd

        return _fevd(self.model, self.theta_, self.B_, self.Lambda_, horizon, by_state)

    def plot_regimes(self, series=None, threshold: float = 0.7, diff: bool = False, **kwargs):
        """Regime-probability shading plots (Tether Figures 4-6 style)."""
        from .plotting import plot_regimes

        return plot_regimes(self, series=series, threshold=threshold, diff=diff, **kwargs)

    def test_restrictions(
        self,
        restrictions: Restrictions,
        df_override=None,
        n_starts: int = 20,
        random_state=None,
        n_jobs: int = -1,
    ):
        """Fit the restricted model and LR-test it against this fit (spec 7)."""
        from .inference.lr_tests import compare_models

        restricted = self.model.fit(
            restrictions=restrictions,
            n_starts=n_starts,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        table = compare_models(self, restricted, df_override=df_override)
        return table, restricted

    def test_b_invariance(self):
        """LR test for state-invariant B, M >= 3 (HL Eq. 6; spec 6.1)."""
        from .inference.invariance import lr_state_invariance

        return lr_state_invariance(self)

    def test_overidentification(self, n_boot: int = 3000, n_jobs: int = -1, random_state=None):
        """Bootstrap overidentification J-test (Tether App. F.1; spec 6.2)."""
        from .inference.invariance import j_test_overidentification

        return j_test_overidentification(
            self, n_boot=n_boot, n_jobs=n_jobs, random_state=random_state
        )

    def test_b_stability(
        self,
        center_date,
        window_before=(16, 5),
        window_after=(5, 16),
        n_boot: int = 3000,
        min_regime_obs: int = 30,
        prob_threshold: float = 0.7,
        n_jobs: int = -1,
        random_state=None,
    ):
        """Overlapping-window Wald test for constant B (Tether App. F.2; spec 6.3)."""
        from .inference.stability import b_stability_test

        return b_stability_test(
            self,
            center_date,
            window_before=window_before,
            window_after=window_after,
            n_boot=n_boot,
            min_regime_obs=min_regime_obs,
            prob_threshold=prob_threshold,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def rolling_stability(self, window: int = 400, lags: int = 3):
        """Rolling-window coefficient stability diagnostics (Tether App. E)."""
        from .inference.stability import rolling_stability

        return rolling_stability(self, window=window, lags=lags)

    # ------------------------------------------------- shock ordering
    def reorder_shocks(self, order):
        """Permute the structural shock columns in place and return self.

        B is identified only up to column permutation and sign (HL 2014,
        Sec. 3.1), so any permutation of the shock columns describes the
        same model; this method applies one consistently across ``B_``,
        ``Lambda_``, ``structural_shocks_`` and cached standard errors.

        Parameters
        ----------
        order : sequence of int
            ``order[j]`` is the (0-indexed) current column to place at new
            position j, e.g. ``[1, 2, 0]``.
        """
        order = list(order)
        if sorted(order) != list(range(self.K)):
            raise ValueError(f"order must be a permutation of 0..{self.K - 1}, got {order}")
        R = self.restrictions
        if R.n_zero_restrictions or R.sign_restrictions:
            warnings.warn(
                "reordering shocks of a restricted model: the positions in "
                "the Restrictions object no longer match the new column "
                "order; interpret restricted positions with care",
                UserWarning,
            )
        self.B_ = self.B_[:, order]
        self.Lambda_ = [np.asarray(lam)[order] for lam in self.Lambda_]
        eps = self.structural_shocks_.to_numpy()[:, order]
        self.structural_shocks_ = pd.DataFrame(
            eps, index=self.structural_shocks_.index,
            columns=[f"eps{j + 1}" for j in range(self.K)],
        )
        if self._se is not None:
            se = self._se
            se.se_B = se.se_B[:, order]
            se.se_lambda = [s[order] for s in se.se_lambda]
            perm_full = [m * self.K + j for m in range(self.M - 1) for j in order]
            se.V_lambda = se.V_lambda[np.ix_(perm_full, perm_full)]
        self._ident = None  # H0 labels are index-based; recompute on demand
        return self

    def sort_shocks(self, regime: int = 2, ascending: bool = False):
        """Order shocks by their relative variance in ``regime`` (HL's
        volatility-labeling device).

        With ``ascending=False`` (default), shock 1 becomes the one with the
        largest lambda in the chosen regime.  Only meaningful when the
        lambdas are statistically distinct -- check the identification block
        of ``summary()`` first.
        """
        if not 2 <= regime <= self.M:
            raise ValueError(f"regime must be in 2..{self.M}")
        lam = np.asarray(self.Lambda_[regime - 1])
        order = np.argsort(lam, kind="stable")
        if not ascending:
            order = order[::-1]
        return self.reorder_shocks(order.tolist())

    def order_shocks_by_variables(self, warn_margin: float = 0.1):
        """Match each shock column to "its" variable (Cholesky-style labels).

        Assigns the columns of B to variables by maximizing the total
        scale-normalized impact share (Hungarian assignment), so that after
        reordering, shock j is the one whose impact falls mainly on
        variable j -- the ordering convention readers of conventional SVARs
        expect.  Impacts are normalized by each variable's residual
        standard deviation, making the assignment unit-free.

        Only meaningful when B is roughly diagonal-dominant after
        normalization, i.e. each statistical shock loads mainly on one
        distinct variable.  When the assignment is ambiguous (the assigned
        variable's impact share exceeds the best alternative by less than
        ``warn_margin``), a warning is raised: the statistical shocks are
        then likely mixtures and variable labels should not be trusted --
        prefer ``sort_shocks`` (lambda-magnitude ordering) or explicit
        restrictions in that case.
        """
        from scipy.optimize import linear_sum_assignment

        sd = self.residuals_.std(axis=0).to_numpy()
        Bn = np.abs(self.B_) / np.maximum(sd[:, None], 1e-300)
        share = Bn**2 / np.maximum((Bn**2).sum(axis=0, keepdims=True), 1e-300)
        rows, cols = linear_sum_assignment(-share)
        for i, j in zip(rows, cols):
            margin = share[i, j] - np.delete(share[:, j], i).max()
            if margin < warn_margin:
                warnings.warn(
                    f"ambiguous shock-to-variable assignment: column {j} is "
                    f"matched to variable '{self.residuals_.columns[i]}' by a "
                    f"margin of only {margin:.3f} in impact share; the shock "
                    "may be a mixture -- consider lambda-based ordering or "
                    "explicit restrictions instead",
                    UserWarning,
                )
        return self.reorder_shocks(cols.tolist())

    def plot_convergence(self, log_scale: bool = True, figsize=(9, 4)):
        """EM convergence diagnostics for the winning start.

        Left panel: log-likelihood path over EM iterations.  Right panel:
        per-iteration log-likelihood improvement (log scale by default) --
        a long flat tail here means the fit plateaued well before the
        stopping rule triggered.  The dashed line marks the best converged
        log-likelihood across all multi-start runs (``loglik_starts_``).
        """
        import matplotlib.pyplot as plt

        ll = np.asarray(self.loglik_history_)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.plot(np.arange(1, ll.size + 1), ll, color="black", lw=1.2)
        finite = self.loglik_starts_[np.isfinite(self.loglik_starts_)]
        if finite.size:
            ax1.axhline(finite.max(), color="tab:blue", ls="--", lw=0.8,
                        label="best start")
            ax1.legend(fontsize=8)
        ax1.set_xlabel("EM iteration")
        ax1.set_ylabel("log-likelihood")
        ax1.set_title(f"EM path ({'converged' if self.converged_ else 'NOT converged'}"
                      f" in {self.n_iter_} iterations)")
        gain = np.diff(ll)
        ax2.plot(np.arange(2, ll.size + 1), np.maximum(gain, 0.0), color="black", lw=1.0)
        if log_scale:
            ax2.set_yscale("log")
        ax2.set_xlabel("EM iteration")
        ax2.set_ylabel("log-likelihood gain per iteration")
        ax2.set_title("Improvement per iteration")
        fig.tight_layout()
        return fig

    def to_pickle(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def from_pickle(path: str):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------- summary
    def summary(self, print_output: bool = True) -> str:
        se = None
        try:
            se = self.std_errors_
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:  # degenerate fits
            warnings.warn(f"standard errors unavailable: {exc}", UserWarning)
        lines = []
        name = type(self.model).__name__
        lines.append("=" * 72)
        lines.append(
            f"{name} results -- K={self.K}, M={self.M}, lags={self.model.lags}, "
            f"deterministic='{self.model.deterministic}'"
        )
        lines.append("=" * 72)
        lines.append(f"Observations: {self.nobs_}    log L: {self.loglik_:.4f}")
        lines.append(
            f"AIC: {self.aic_:.2f}    SC: {self.sc_:.2f}    "
            f"free parameters: {self.n_free_params_}"
        )
        lines.append(
            f"Converged: {self.converged_} in {self.n_iter_} EM iterations "
            f"({np.isfinite(self.loglik_starts_).sum()} successful starts)"
        )
        if hasattr(self.model, "beta"):
            lines.append(f"beta source: {self.model.beta_source} (r={self.model.coint_rank})")
            lines.append("beta':")
            lines.append(np.array_str(self.model.beta.T, precision=4))
        lines.append("-" * 72)
        lines.append("B (instantaneous effects; state-1 shocks have unit variance):")
        lines.append(np.array_str(self.B_, precision=4, suppress_small=True))
        if se is not None:
            lines.append("B standard errors:")
            lines.append(np.array_str(se.se_B, precision=4, suppress_small=True))
        for m in range(1, self.M):
            lam = self.Lambda_[m]
            row = ", ".join(f"l{m + 1}{i + 1}={v:.4g}" for i, v in enumerate(lam))
            if se is not None:
                ses = se.se_lambda[m - 1]
                row += "   (se: " + ", ".join(f"{v:.3g}" for v in ses) + ")"
            lines.append(f"Lambda_{m + 1}: {row}")
        lines.append("Transition matrix P (columns sum to 1):")
        lines.append(np.array_str(self.P_, precision=4, suppress_small=True))
        if se is not None and se.boundary:
            lines.append(f"Parameters at boundary (SE = na): {', '.join(se.boundary)}")
        lines.append("-" * 72)
        if se is not None:
            try:
                ident = self.identification_
                lines.append(str(ident))
            except (np.linalg.LinAlgError, ValueError) as exc:
                lines.append(f"Identification tests unavailable: {exc}")
        lines.append("=" * 72)
        out = "\n".join(lines)
        if print_output:
            print(out)
        return out


class MSVARResults(_ResultsBase):
    """Results for the levels MSVAR."""


class MSVECMResults(_ResultsBase):
    """Results for the MSVECM; adds beta and the Johansen pre-test table."""

    def __init__(self, model, state, R, DY, Z, index, logliks, config):
        super().__init__(model, state, R, DY, Z, index, logliks, config)
        self.beta_ = model.beta
        self.beta_source = model.beta_source
        self.johansen_table_ = model.johansen_table_


def _make_results(model, state, R, DY, Z, index, logliks, config):
    from .model import MSVECM

    cls = MSVECMResults if isinstance(model, MSVECM) else MSVARResults
    return cls(model, state, R, DY, Z, index, logliks, config)
