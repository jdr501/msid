"""Shared fixtures and helpers for the msid test suite."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest


def closest_signed_permutation(B_est: np.ndarray, B_true: np.ndarray):
    """Align columns of ``B_est`` to ``B_true`` by the closest signed
    permutation (structural shocks are unique only up to column sign and
    permutation; HL 2014, Section 3.1).

    Returns ``(B_aligned, perm, signs)``.
    """
    K = B_true.shape[0]
    best = None
    best_err = np.inf
    for perm in permutations(range(K)):
        Bp = B_est[:, list(perm)]
        signs = np.sign(np.sum(Bp * B_true, axis=0))
        signs[signs == 0] = 1.0
        Bc = Bp * signs
        err = np.linalg.norm(Bc - B_true)
        if err < best_err:
            best, best_err = (Bc, list(perm), signs), err
    return best


@pytest.fixture(scope="session")
def example_data():
    import msid

    return msid.load_example()


@pytest.fixture(scope="session")
def fitted_example(example_data):
    """One shared K=3, M=2 MSVECM fit reused across tests."""
    import msid

    model = msid.MSVECM(example_data, lags=1, coint_rank=1, n_regimes=2)
    return model.fit(n_starts=3, n_jobs=1, random_state=0)
