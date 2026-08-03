"""Model classes: MSVAR (levels VAR) and MSVECM with MS heteroskedasticity.

Both share the EM loop, likelihood, structural M-step, bootstrap, tests and
plots; they differ only in the regressor block ``Z_{t-1}`` and the long-run
effect matrix ``Xi`` (spec Section 1.2):

* MSVECM:  ``Dy_t = nu0 + nu1 t + alpha beta' y_{t-1} + sum Gamma_i Dy_{t-i} + u_t``
  with ``Xi = beta_perp [alpha_perp' (I - sum Gamma_i) beta_perp]^{-1}
  alpha_perp' B`` (HL 2014, Eq. 2).
* MSVAR:   ``y_t = nu0 + nu1 t + sum A_i y_{t-i} + u_t`` with
  ``Xi = (I - sum A_i)^{-1} B`` (stationarity required).

The MS covariance structure is ``u_t | s_t ~ N(0, Sigma_{s_t})`` with
``Sigma_1 = BB'`` and ``Sigma_m = B Lambda_m B'`` (HL 2014, Eq. 4-5).
"""

from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.linalg import null_space

from .estimation.em import EMConfig, em_estimate
from .estimation.init import make_starts
from .restrictions import Restrictions

__all__ = ["MSVAR", "MSVECM"]

_DET_CHOICES = ("n", "c", "ct")


def _as_frame(y) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y
    y = np.asarray(y, dtype=float)
    return pd.DataFrame(y, columns=[f"y{i + 1}" for i in range(y.shape[1])])


class _MSBase(ABC):
    """Abstract base holding data, design construction and the fit loop."""

    def __init__(
        self,
        y,
        lags: int,
        n_regimes: int = 2,
        deterministic: str = "ct",
        exog=None,
    ) -> None:
        if deterministic not in _DET_CHOICES:
            raise ValueError(f"deterministic must be one of {_DET_CHOICES}")
        self.y = _as_frame(y)
        if self.y.isna().any().any():
            raise ValueError("y contains missing values")
        self.K = self.y.shape[1]
        if self.K < 2:
            raise ValueError("need at least K = 2 variables")
        self.lags = int(lags)
        if not 2 <= int(n_regimes) <= 4:
            raise ValueError("n_regimes must be in {2, 3, 4}")
        self.M = int(n_regimes)
        self.deterministic = deterministic
        self.exog = None if exog is None else np.asarray(exog, dtype=float)

    # ------------------------------------------------------------ design
    @abstractmethod
    def _build_design(self) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        """Return ``(DY, Z, index)`` for the effective sample."""

    @abstractmethod
    def longrun_map(self, Theta: np.ndarray) -> np.ndarray:
        """``C`` with ``Xi = C @ B`` for the current slope parameters."""

    def _det_block(self, T: int) -> np.ndarray:
        cols = []
        if self.deterministic in ("c", "ct"):
            cols.append(np.ones(T))
        if self.deterministic == "ct":
            cols.append(np.arange(1, T + 1, dtype=float))
        return np.column_stack(cols) if cols else np.empty((T, 0))

    def _exog_block(self, T: int, offset: int) -> np.ndarray:
        if self.exog is None:
            return np.empty((T, 0))
        X = self.exog
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != len(self.y):
            raise ValueError("exog must have the same number of rows as y")
        return X[offset : offset + T]

    @property
    def n_det(self) -> int:
        return {"n": 0, "c": 1, "ct": 2}[self.deterministic]

    # --------------------------------------------------------------- fit
    def fit(
        self,
        restrictions: Restrictions | None = None,
        n_starts: int = 50,
        max_iter: int = 500,
        tol_ll: float = 1e-8,
        tol_param: float = 1e-6,
        b0_scale: float = 0.1,
        lambda_init: str = "random",
        lambda_range: tuple[float, float] = (0.1, 10.0),
        label_order: str = "lambda_sort",
        struct_maxiter: int = 200,
        n_jobs: int = -1,
        random_state=None,
    ):
        """Estimate by EM with multi-start (spec Section 3).

        ``n_starts`` defaults to 50; hard problems (small samples, many
        regimes) may need far more -- HL used >= 10,000 starts for their
        small quarterly sample.  All converged log-likelihoods are stored in
        ``results.loglik_starts_``; the EM path of the winning start is in
        ``results.loglik_history_``.  ``struct_maxiter`` caps the inner
        L-BFGS iterations of each structural M-step.
        """
        from .results import _make_results

        R = restrictions if restrictions is not None else Restrictions(self.K)
        if R.K != self.K:
            raise ValueError(f"Restrictions built for K={R.K}, model has K={self.K}")
        DY, Z, index = self._build_design()
        config = EMConfig(
            max_iter=max_iter,
            tol_ll=tol_ll,
            tol_param=tol_param,
            label_order=label_order,
            struct_maxiter=struct_maxiter,
        )
        starts = make_starts(
            DY,
            Z,
            R,
            self.M,
            n_starts=n_starts,
            b0_scale=b0_scale,
            lambda_init=lambda_init,
            lambda_range=lambda_range,
            random_state=random_state,
        )
        C_map = self.longrun_map if R.has_xi_restrictions else None
        best, logliks = em_estimate(
            DY, Z, R, self.M, starts, config, C_longrun=C_map, n_jobs=n_jobs
        )
        return _make_results(self, best, R, DY, Z, index, logliks, config)

    @classmethod
    def from_pickle(cls, path: str):
        """Load a results object previously stored with ``results.to_pickle``."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj


class MSVAR(_MSBase):
    """Markov-switching structural VAR in levels (first-class, spec 1.2).

    ``y_t = nu0 + nu1 t + sum_{i=1}^p A_i y_{t-i} + u_t``, ``u_t = B eps_t``.
    """

    def _build_design(self):
        Y = self.y.to_numpy()
        p, K = self.lags, self.K
        T = Y.shape[0] - p
        if T < K * (p + 2):
            warnings.warn("very short effective sample relative to model size", UserWarning)
        DY = Y[p:]
        blocks = [self._det_block(T)]
        for i in range(1, p + 1):
            blocks.append(Y[p - i : p - i + T])
        blocks.append(self._exog_block(T, p))
        Z = np.column_stack(blocks)
        return DY, Z, self.y.index[p:]

    def slope_blocks(self, Theta: np.ndarray) -> dict:
        """Split Theta into deterministic terms and A_1..A_p."""
        d = self.n_det
        out = {"det": Theta[:, :d], "A": []}
        for i in range(self.lags):
            out["A"].append(Theta[:, d + i * self.K : d + (i + 1) * self.K])
        return out

    def companion(self, Theta: np.ndarray) -> np.ndarray:
        A = self.slope_blocks(Theta)["A"]
        K, p = self.K, self.lags
        comp = np.zeros((K * p, K * p))
        for i, Ai in enumerate(A):
            comp[:K, i * K : (i + 1) * K] = Ai
        if p > 1:
            comp[K:, : K * (p - 1)] = np.eye(K * (p - 1))
        return comp

    def longrun_map(self, Theta: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        A = self.slope_blocks(Theta)["A"]
        comp = self.companion(Theta)
        rho = np.max(np.abs(np.linalg.eigvals(comp)))
        if rho >= 1.0 - tol:
            raise ValueError(
                f"the levels VAR is not stationary (companion spectral radius "
                f"{rho:.6f} >= 1 - {tol:g}); the long-run matrix Xi = "
                f"(I - sum A_i)^{{-1}} B is undefined. Consider an MSVECM."
            )
        return np.linalg.inv(np.eye(self.K) - sum(A))


class MSVECM(_MSBase):
    """Markov-switching structural VECM (primary specification, spec 1.1).

    Parameters
    ----------
    y : array-like or DataFrame, shape (T, K)
    lags : int
        Number of *lagged differences* p in the VECM.
    coint_rank : int or None
        Cointegration rank r.  ``None`` selects r by the Johansen trace test
        at level ``rank_alpha`` (table stored in ``johansen_table_``).
    beta : ndarray (K, r) or None
        ``None`` (default): estimate beta by Johansen reduced-rank
        regression and hold it fixed through the EM iterations (the
        two-step approach of the Tether paper).  An array: use the supplied
        cointegration matrix directly (e.g. HL's beta = (0,0,0,1)').
    deterministic : {"n", "c", "ct"}
    n_regimes : int, 2 to 4.
    """

    def __init__(
        self,
        y,
        lags: int,
        coint_rank: int | None = None,
        beta: np.ndarray | None = None,
        deterministic: str = "ct",
        n_regimes: int = 2,
        rank_alpha: float = 0.05,
        exog=None,
        alpha_restrictions=None,
    ) -> None:
        super().__init__(y, lags, n_regimes=n_regimes, deterministic=deterministic, exog=exog)
        if isinstance(beta, str):
            raise NotImplementedError(
                "mixed known/estimated beta ('known_partial') is out of scope; "
                "supply either beta=None (Johansen) or a full (K, r) array"
            )
        if alpha_restrictions is not None:
            raise NotImplementedError(
                "alpha (weak-exogeneity) restrictions are not implemented yet; "
                "the hook is reserved for future use"
            )
        self.rank_alpha = float(rank_alpha)
        self.johansen_table_ = None
        if beta is not None:
            beta = np.asarray(beta, dtype=float)
            if beta.ndim == 1:
                beta = beta[:, None]
            if beta.shape[0] != self.K:
                raise ValueError(f"beta must have K={self.K} rows, got shape {beta.shape}")
            r = beta.shape[1]
            if coint_rank is not None and coint_rank != r:
                raise ValueError(f"beta has {r} columns but coint_rank={coint_rank}")
            if not np.all(np.isfinite(beta)):
                raise ValueError("beta contains non-finite entries")
            if np.linalg.matrix_rank(beta) < r:
                raise ValueError(f"beta must have full column rank {r}")
            self.beta = beta
            self.coint_rank = r
            self.beta_source = "user"
        else:
            self.beta, self.coint_rank, self.johansen_table_ = self._johansen_beta(coint_rank)
            self.beta_source = "johansen"

    # ------------------------------------------------------------ Johansen
    def _johansen_beta(self, coint_rank: int | None):
        from .pretest import johansen

        det_order = {"n": -1, "c": 0, "ct": 1}[self.deterministic]
        res = johansen(self.y, det_order=det_order, k_ar_diff=self.lags)
        table = res.table
        if coint_rank is None:
            idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(self.rank_alpha)
            if idx is None:
                raise ValueError("rank_alpha must be one of 0.10, 0.05, 0.01")
            r = self.K
            for j in range(self.K):
                if res.trace_stat[j] < res.trace_crit[j, idx]:
                    r = j
                    break
            if r == 0:
                raise ValueError(
                    "Johansen trace test finds no cointegration (r = 0) at "
                    f"level {self.rank_alpha}; use MSVAR on differences or "
                    "supply beta/coint_rank explicitly"
                )
        else:
            r = int(coint_rank)
            if not 1 <= r < self.K:
                raise ValueError(f"coint_rank must satisfy 1 <= r < K = {self.K}")
        beta = res.evec[:, :r]
        # Phillips triangular normalization: beta = [I_r, beta*']'
        top = beta[:r, :r]
        if np.linalg.matrix_rank(top) < r:
            warnings.warn(
                "leading (r x r) block of beta is singular; skipping Phillips " "normalization",
                UserWarning,
            )
        else:
            beta = beta @ np.linalg.inv(top)
        return beta, r, table

    # ------------------------------------------------------------ design
    def _build_design(self):
        Y = self.y.to_numpy()
        p, K = self.lags, self.K
        dY = np.diff(Y, axis=0)  # dY[t] = y_{t+1} - y_t
        T = dY.shape[0] - p  # effective sample
        if T < K * (p + 2):
            warnings.warn("very short effective sample relative to model size", UserWarning)
        DY = dY[p:]
        ect = Y[p : p + T] @ self.beta  # beta' y_{t-1}
        blocks = [self._det_block(T), ect]
        for i in range(1, p + 1):
            blocks.append(dY[p - i : p - i + T])
        blocks.append(self._exog_block(T, p + 1))
        Z = np.column_stack(blocks)
        return DY, Z, self.y.index[p + 1 :]

    def slope_blocks(self, Theta: np.ndarray) -> dict:
        d, r, K = self.n_det, self.coint_rank, self.K
        out = {"det": Theta[:, :d], "alpha": Theta[:, d : d + r], "Gamma": []}
        for i in range(self.lags):
            out["Gamma"].append(Theta[:, d + r + i * K : d + r + (i + 1) * K])
        return out

    def longrun_map(self, Theta: np.ndarray) -> np.ndarray:
        """C with Xi = C B from the Granger representation (HL 2014, Eq. 2)."""
        blocks = self.slope_blocks(Theta)
        alpha, Gammas = blocks["alpha"], blocks["Gamma"]
        K = self.K
        b_perp = null_space(self.beta.T)
        a_perp = null_space(alpha.T)
        if b_perp.shape[1] == 0 or a_perp.shape[1] == 0:
            raise ValueError("beta or alpha has full rank K; Xi is not defined")
        G = np.eye(K) - sum(Gammas) if Gammas else np.eye(K)
        core = a_perp.T @ G @ b_perp
        return b_perp @ np.linalg.solve(core, a_perp.T)

    def levels_var_coefs(self, Theta: np.ndarray) -> list[np.ndarray]:
        """A_1..A_{p+1} of the equivalent levels VAR (for IRFs).

        ``A_1 = I + alpha beta' + Gamma_1``, ``A_i = Gamma_i - Gamma_{i-1}``,
        ``A_{p+1} = -Gamma_p`` (Luetkepohl 2005, Ch. 6).
        """
        blocks = self.slope_blocks(Theta)
        alpha, G = blocks["alpha"], blocks["Gamma"]
        K, p = self.K, self.lags
        Pi = alpha @ self.beta.T
        A = [np.eye(K) + Pi + (G[0] if p > 0 else 0.0)]
        for i in range(1, p):
            A.append(G[i] - G[i - 1])
        if p > 0:
            A.append(-G[p - 1])
        return A
