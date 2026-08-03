"""Simulation of MSVAR / MSVECM processes (for tests, examples, docs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["default_msvecm_params", "simulate_msvar", "simulate_msvecm"]


def _simulate_regimes(T: int, P: np.ndarray, rng) -> np.ndarray:
    M = P.shape[0]
    # stationary distribution as the starting law
    w, V = np.linalg.eig(P)
    pi = np.real(V[:, np.argmax(np.real(w))])
    pi = np.abs(pi) / np.abs(pi).sum()
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(M, p=pi)
    for t in range(1, T):
        s[t] = rng.choice(M, p=P[:, s[t - 1]])
    return s


def _draw_errors(T, B, Lambdas, states, rng):
    K = B.shape[0]
    eps = rng.standard_normal((T, K))
    scale = np.array([np.sqrt(np.asarray(Lambdas[m])) for m in states])
    return (eps * scale) @ B.T


def simulate_msvar(T, A, B, Lambdas, P, nu0=None, nu1=None, burn=200, random_state=None):
    """Simulate a stationary levels MSVAR; returns (DataFrame y, states)."""
    rng = np.random.default_rng(random_state)
    K = B.shape[0]
    p = len(A)
    nu0 = np.zeros(K) if nu0 is None else np.asarray(nu0)
    nu1 = np.zeros(K) if nu1 is None else np.asarray(nu1)
    total = T + burn
    states = _simulate_regimes(total, P, rng)
    U = _draw_errors(total, B, Lambdas, states, rng)
    Y = np.zeros((total + p, K))
    for t in range(total):
        i = t + p
        acc = nu0 + nu1 * (t + 1) + U[t]
        for lag, Ai in enumerate(A, start=1):
            acc = acc + Ai @ Y[i - lag]
        Y[i] = acc
    y = Y[p + burn :]
    idx = pd.date_range("2019-01-01", periods=T, freq="D")
    return pd.DataFrame(y, index=idx, columns=[f"y{i + 1}" for i in range(K)]), states[burn:]


def simulate_msvecm(T, alpha, beta, Gammas, B, Lambdas, P, nu0=None, burn=200, random_state=None):
    """Simulate an MSVECM ``Dy_t = nu0 + alpha beta' y_{t-1} + sum Gamma_i
    Dy_{t-i} + B eps_t``; returns (DataFrame y, states)."""
    rng = np.random.default_rng(random_state)
    K = B.shape[0]
    p = len(Gammas)
    nu0 = np.zeros(K) if nu0 is None else np.asarray(nu0)
    total = T + burn
    states = _simulate_regimes(total, P, rng)
    U = _draw_errors(total, B, Lambdas, states, rng)
    Y = np.zeros((total + p + 1, K))
    dY = np.zeros((total + p + 1, K))
    for t in range(total):
        i = t + p + 1
        acc = nu0 + alpha @ (beta.T @ Y[i - 1]) + U[t]
        for lag, Gi in enumerate(Gammas, start=1):
            acc = acc + Gi @ dY[i - lag]
        dY[i] = acc
        Y[i] = Y[i - 1] + acc
    y = Y[p + 1 + burn :]
    idx = pd.date_range("2019-01-01", periods=T, freq="D")
    return pd.DataFrame(y, index=idx, columns=[f"y{i + 1}" for i in range(K)]), states[burn:]


def default_msvecm_params(K: int = 3, M: int = 2):
    """A well-behaved K=3, M=2 MSVECM parameterization used in the tests
    and the bundled example dataset."""
    if K != 3:
        raise ValueError("default parameters are provided for K = 3")
    beta = np.array([[1.0], [-0.5], [-0.2]])
    alpha = np.array([[-0.30], [0.10], [0.05]])
    Gammas = [np.array([[0.20, 0.00, 0.05], [0.05, 0.15, 0.00], [0.00, 0.05, 0.10]])]
    B = np.array([[1.00, 0.00, 0.30], [0.50, 1.20, 0.00], [0.20, 0.30, 0.90]])
    if M == 2:
        Lambdas = [np.ones(3), np.array([6.0, 0.3, 15.0])]
        P = np.array([[0.95, 0.08], [0.05, 0.92]])
    elif M == 3:
        Lambdas = [np.ones(3), np.array([6.0, 0.3, 15.0]), np.array([0.2, 9.0, 2.5])]
        P = np.array([[0.92, 0.05, 0.05], [0.05, 0.90, 0.05], [0.03, 0.05, 0.90]])
    else:
        Lambdas = [
            np.ones(3),
            np.array([6.0, 0.3, 15.0]),
            np.array([0.2, 9.0, 2.5]),
            np.array([25.0, 2.0, 0.5]),
        ]
        P = 0.80 * np.eye(4) + 0.05  # columns sum to exactly 1
    return {"alpha": alpha, "beta": beta, "Gammas": Gammas, "B": B, "Lambdas": Lambdas, "P": P}
