"""Restrictions API and LR degrees of freedom (spec Section 12, item 5)."""

from __future__ import annotations

import numpy as np
import pytest

from msid.inference.lr_tests import compare_models
from msid.restrictions import Restrictions


def test_free_parameter_counting():
    R = Restrictions(K=3)
    assert R.n_free_b == 9
    assert R.n_zero_restrictions == 0
    R.b_zeros([(2, 0), (2, 1)])
    assert R.n_free_b == 7
    assert R.n_zero_restrictions == 2
    R.b_signs({(0, 0): "+"})
    assert R.n_free_b == 7  # sign restrictions leave the count unchanged
    assert R.n_zero_restrictions == 2
    R.xi_zeros([(1, 2)])
    assert R.n_zero_restrictions == 3


def test_pattern_matrix_input():
    pat = np.array([[np.nan, 0.0, np.nan], [np.nan, np.nan, np.nan], [0.0, np.nan, np.nan]])
    R = Restrictions(K=3, b_pattern=pat)
    assert set(R.b_zero_indices) == {(0, 1), (2, 0)}
    B = R.unpack_b(R.pack_b(np.arange(9, dtype=float).reshape(3, 3)))
    assert B[0, 1] == 0.0 and B[2, 0] == 0.0


def test_selection_matrix_roundtrip():
    R = Restrictions(K=3).b_zeros([(2, 0)])
    B = np.arange(1, 10, dtype=float).reshape(3, 3)
    B[2, 0] = 0.0
    assert np.allclose(R.unpack_b(R.pack_b(B)), B)


def test_over_restriction_warns():
    with pytest.warns(UserWarning, match="degenerate"):
        Restrictions(K=3).b_zeros([(0, 0), (1, 0), (2, 0)])


def test_sign_on_zero_element_raises():
    R = Restrictions(K=3).b_zeros([(2, 0)])
    with pytest.raises(ValueError):
        R.b_signs({(2, 0): "+"})


def test_sign_normalization():
    R = Restrictions(K=2).b_signs({(0, 0): "+"})
    B = np.array([[-1.0, 0.5], [2.0, -0.7]])
    Bn = R.normalize_signs(B)
    assert Bn[0, 0] > 0
    # reference-based alignment (bootstrap mode)
    ref = np.array([[1.0, 0.5], [-2.0, -0.7]])
    Bn2 = R.normalize_signs(B, reference=ref)
    assert np.sum(Bn2[:, 0] * ref[:, 0]) > 0


def test_restricted_fit_reduces_free_params(fitted_example):
    res_u = fitted_example
    K = res_u.K
    R = Restrictions(K).b_zeros([(2, 0), (2, 1)]).b_signs({(0, 0): "+"})
    table, res_r = res_u.test_restrictions(R, n_starts=2, random_state=0, n_jobs=1)
    assert res_r.n_free_params_ == res_u.n_free_params_ - 2
    assert res_r.B_[2, 0] == 0.0 and res_r.B_[2, 1] == 0.0
    assert table["df"].iloc[0] == 2
    assert table["LR"].iloc[0] >= 0.0
    # df_override follows the paper's printed convention when requested
    t2 = compare_models(res_u, res_r, df_override=2)
    assert t2["df"].iloc[0] == 2


def test_xi_zero_restriction(fitted_example):
    """Long-run zeros are honored to tolerance at convergence."""
    res_u = fitted_example
    R = Restrictions(3).xi_zeros([(0, 1)])
    table, res_r = res_u.test_restrictions(R, n_starts=2, random_state=0, n_jobs=1)
    Xi = res_r.Xi_
    scale = np.abs(Xi).max()
    assert abs(Xi[0, 1]) < 1e-4 * max(scale, 1.0)
    assert res_r.loglik_ <= res_u.loglik_ + 1e-6
    assert table["df"].iloc[0] == 1
