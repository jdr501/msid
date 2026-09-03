"""The deterministic max own-impact shock-labeling rule (msid.labeling)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from msid.labeling import (
    DEFAULT_MIN_GAP,
    assignment_scores,
    impact_share_matrix,
    max_own_impact_order,
)

# Impact shares with columns summing to one.  Scored by hand, the six
# assignments are 2.35, 1.05, 0.98, 0.97, 0.33, 0.32 -- so the rule picks
# (1, 0, 2) with a best-minus-second-best gap of 1.30.
S_EXAMPLE = np.array(
    [
        [0.10, 0.75, 0.08],
        [0.80, 0.15, 0.12],
        [0.10, 0.10, 0.80],
    ]
)


def _diag_dominant(rng, K):
    """A B whose scaled impacts load mainly on one variable per shock."""
    B = 0.15 * rng.standard_normal((K, K))
    B[np.arange(K), np.arange(K)] += 3.0 * rng.uniform(0.8, 1.2, size=K)
    return B


# ------------------------------------------------------------- share matrix


def test_share_matrix_columns_sum_to_one():
    rng = np.random.default_rng(0)
    S = impact_share_matrix(rng.standard_normal((4, 4)), rng.uniform(0.5, 3.0, size=4))
    assert np.allclose(S.sum(axis=0), 1.0)
    assert np.all(S >= 0.0)


def test_share_matrix_is_unit_free():
    """Rescaling a variable's units must not move any impact share."""
    rng = np.random.default_rng(1)
    B = rng.standard_normal((3, 3))
    scale = rng.uniform(0.5, 2.0, size=3)
    S = impact_share_matrix(B, scale)

    B2, scale2 = B.copy(), scale.copy()
    B2[1, :] *= 1000.0
    scale2[1] *= 1000.0
    assert np.allclose(impact_share_matrix(B2, scale2), S)


def test_share_matrix_validates_input():
    with pytest.raises(ValueError, match="square"):
        impact_share_matrix(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="length K"):
        impact_share_matrix(np.eye(3), np.ones(2))
    with pytest.raises(ValueError, match="positive"):
        impact_share_matrix(np.eye(3), np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        impact_share_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]]), np.ones(2))


def test_zero_column_warns():
    B = np.array([[1.0, 0.0], [0.5, 0.0]])
    with pytest.warns(UserWarning, match="zero column"):
        impact_share_matrix(B, np.ones(2))


# ------------------------------------------------------------------ scoring


def test_hand_checked_assignment_scores():
    scored = assignment_scores(S_EXAMPLE)
    assert len(scored) == 6
    assert scored[0][0] == (1, 0, 2)
    assert scored[0][1] == pytest.approx(2.35)
    assert scored[1][0] == (0, 1, 2)
    assert scored[1][1] == pytest.approx(1.05)
    assert [q for _, q in scored] == sorted((q for _, q in scored), reverse=True)


def test_assignment_scores_refuses_large_k():
    with pytest.raises(ValueError, match="infeasible"):
        assignment_scores(np.eye(9))


def test_score_bounds():
    """Q runs from about 1 (no structure) to K (perfect one-to-one)."""
    K = 4
    diffuse = np.full((K, K), 1.0 / K)
    assert assignment_scores(diffuse)[0][1] == pytest.approx(1.0)
    assert assignment_scores(np.eye(K))[0][1] == pytest.approx(float(K))


# -------------------------------------------------------------- the rule


def test_recovers_planted_permutation():
    rng = np.random.default_rng(2)
    for _ in range(20):
        K = int(rng.integers(2, 6))
        scale = rng.uniform(0.5, 4.0, size=K)
        B = _diag_dominant(rng, K) * scale[:, None]
        p = [int(j) for j in rng.permutation(K)]
        report = max_own_impact_order(B[:, p], scale, min_gap=0.0)
        assert report.order == [p.index(i) for i in range(K)]


def test_selection_invariant_to_variable_scale():
    rng = np.random.default_rng(3)
    scale = rng.uniform(0.5, 3.0, size=3)
    B = _diag_dominant(rng, 3) * scale[:, None]
    base = max_own_impact_order(B[:, [2, 0, 1]], scale, min_gap=0.0)

    B2, scale2 = B.copy(), scale.copy()
    B2[0, :] *= 500.0
    scale2[0] *= 500.0
    rescaled = max_own_impact_order(B2[:, [2, 0, 1]], scale2, min_gap=0.0)
    assert rescaled.order == base.order
    assert rescaled.score == pytest.approx(base.score)


def test_agrees_with_hungarian_assignment():
    """Enumeration and the Hungarian optimum select the same assignment."""
    rng = np.random.default_rng(4)
    for _ in range(15):
        K = int(rng.integers(2, 6))
        B = rng.standard_normal((K, K))
        scale = rng.uniform(0.5, 2.0, size=K)
        report = max_own_impact_order(B, scale, min_gap=0.0)
        _, cols = linear_sum_assignment(-impact_share_matrix(B, scale))
        assert report.order == [int(c) for c in cols]


def test_large_k_path_matches_enumeration():
    """Forced-exclusion runner-up equals the brute-force second best."""
    rng = np.random.default_rng(5)
    B = _diag_dominant(rng, 5)
    scale = np.ones(5)
    enumerated = max_own_impact_order(B, scale, min_gap=0.0)
    hungarian = max_own_impact_order(B, scale, min_gap=0.0, max_enumerate=0)

    assert enumerated.enumerated is True
    assert hungarian.enumerated is False
    assert hungarian.all_scores is None
    assert hungarian.order == enumerated.order
    assert hungarian.score == pytest.approx(enumerated.score)
    assert hungarian.runner_up_score == pytest.approx(enumerated.runner_up_score)


def test_own_shares_track_the_chosen_assignment():
    report = max_own_impact_order(np.eye(3) * 2.0, np.ones(3), min_gap=0.0)
    assert report.order == [0, 1, 2]
    assert np.allclose(report.own_shares, 1.0)
    assert report.gap == pytest.approx(report.score - report.runner_up_score)


# ------------------------------------------------------- the failure branch


def test_ambiguous_labeling_warns():
    """Two indistinguishable shocks leave the assignment undetermined."""
    B = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.warns(UserWarning, match="ambiguous shock labeling"):
        report = max_own_impact_order(B, np.ones(3))
    assert report.ambiguous
    assert report.gap == pytest.approx(0.0)
    assert "AMBIGUOUS" in str(report)


def test_strict_mode_raises_on_ambiguity():
    B = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="ambiguous shock labeling"):
        max_own_impact_order(B, np.ones(3), strict=True)


def test_clear_labeling_does_not_warn():
    import warnings

    B = _diag_dominant(np.random.default_rng(6), 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = max_own_impact_order(B, np.ones(3), min_gap=DEFAULT_MIN_GAP)
    assert not report.ambiguous
    assert "labels separated" in str(report)


# ------------------------------------------------------ results integration


def test_results_stores_and_prints_the_report(fitted_example):
    res = copy.deepcopy(fitted_example).order_shocks_by_max_own_impact(min_gap=0.0)
    report = res.labeling_report_

    assert report is not None
    assert sorted(report.order) == list(range(res.K))
    assert len(report.all_scores) == 6  # K = 3
    assert list(res.residuals_.columns) == report.var_names
    assert np.all(np.diag(res.B_) > 0)  # own impacts oriented positive
    assert "Shock labeling" in res.summary(print_output=False)


def test_legacy_variables_rule_selects_the_same_permutation(fitted_example):
    """The new rule and the legacy Hungarian form must not disagree."""
    legacy = copy.deepcopy(fitted_example).order_shocks_by_variables(warn_margin=0.0)
    new = copy.deepcopy(fitted_example).order_shocks_by_max_own_impact(min_gap=0.0)
    assert np.allclose(legacy.B_, new.B_)
    assert legacy.labeling_report_ is None
    assert new.labeling_report_ is not None


def test_reordering_again_clears_a_stale_report(fitted_example):
    res = copy.deepcopy(fitted_example).order_shocks_by_max_own_impact(min_gap=0.0)
    assert res.labeling_report_ is not None
    res.sort_shocks(regime=2)
    assert res.labeling_report_ is None


def test_labeling_leaves_the_likelihood_untouched(fitted_example):
    """Column permutation is a relabeling: no LR statistic can move."""
    res = copy.deepcopy(fitted_example).order_shocks_by_max_own_impact(min_gap=0.0)
    assert res.loglik_ == pytest.approx(fitted_example.loglik_)
    assert np.allclose(res.Sigma_[0], fitted_example.Sigma_[0])
