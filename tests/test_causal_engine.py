"""Tests for Phase 4 Advanced Causal Inference Engine.

Covers:
1. Propensity score overlap, common support, and positivity diagnostics.
2. Covariate balance and SMD reduction before vs. after IPW.
3. Double Machine Learning (LinearDML) execution with grouped cross-fitting.
4. Doubly Robust Estimation (AIPW / DRLearner).
5. ATE and CATE effect estimation and confidence intervals.
6. Ground-truth benchmark evaluation (ATE error, PEHE, effect correlation, CI coverage).
7. Grouped cross-fitting validation (zero customer overlap between folds).
8. Treatment effect heterogeneity and segment explainability.
9. Causal estimation reproducibility with fixed random seeds.
10. Error handling on invalid or non-binary treatment data.
11. Persistence of causal artifacts and evaluation report.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from backend.config import CAUSAL_ENGINE_REPORT, CAUSAL_MODEL_ARTIFACT, SPLIT_DIR
from backend.causal_engine import (
    CAUSAL_FEATURES,
    compute_propensity_and_balance,
    evaluate_against_ground_truth,
    explain_heterogeneity_and_segments,
    fit_doubly_robust_learner,
    fit_linear_dml,
    run_dowhy_refutations,
)


@pytest.fixture
def sample_causal_data():
    """Create deterministic synthetic causal dataset with known grouping."""
    rng = np.random.default_rng(42)
    n_cust = 50
    rows_per_cust = 4
    n = n_cust * rows_per_cust
    cust_ids = np.repeat(np.arange(n_cust), rows_per_cust)

    history = rng.integers(0, 8, n)
    categories = rng.choice(["Electronics", "Fashion", "Beauty", "Home"], size=n)
    prices = np.round(rng.uniform(10.0, 200.0, n), 2)
    seasonality = rng.integers(0, 4, n)

    # Treatment assignment
    propensity = 1.0 / (1.0 + np.exp(-(-0.5 + 0.1 * history - 0.2 * (categories == "Electronics"))))
    treatment = rng.binomial(1, propensity)

    # True effect: 0.04 + 0.025*(Fashion) + 0.015*(history >= 4)
    true_effect = 0.04 + 0.025 * (categories == "Fashion") + 0.015 * (history >= 4)
    base_prob = 1.0 / (1.0 + np.exp(-(-1.5 + 0.1 * history)))
    purchase_prob = np.clip(base_prob + treatment * true_effect, 0.01, 0.99)
    purchase = rng.binomial(1, purchase_prob)

    df = pd.DataFrame({
        "customer_id": cust_ids,
        "category": categories,
        "baseline_price": prices,
        "customer_history": history,
        "seasonality": seasonality,
        "treatment": treatment,
        "purchase": purchase,
    })
    gt_df = pd.DataFrame({
        "true_effect": true_effect,
    })
    return df, gt_df


# =========================================================================
# 1. Propensity and Balance Tests
# =========================================================================

def test_propensity_and_covariate_balance(sample_causal_data):
    """Verify propensity scoring, overlap diagnostics, and SMD reduction."""
    df, _ = sample_causal_data
    diagnostics = compute_propensity_and_balance(df)

    assert 0.5 <= diagnostics["propensity_auc"] <= 1.0
    assert diagnostics["has_common_support"] is True
    assert 0.0 <= diagnostics["overlap_fraction_005_095"] <= 1.0
    assert len(diagnostics["control_quantiles"]) > 0
    assert len(diagnostics["treated_quantiles"]) > 0

    # Covariate balance
    balance = diagnostics["covariate_balance"]
    assert len(balance) >= len(CAUSAL_FEATURES)
    for rec in balance:
        assert "smd_before" in rec
        assert "smd_after_ipw" in rec
        assert "abs_smd_before" in rec
        assert "abs_smd_after_ipw" in rec

    # Max SMD after IPW should be well controlled
    assert diagnostics["max_abs_smd_after_ipw"] < 0.25


def test_invalid_treatment_raises_error(sample_causal_data):
    """Ensure non-binary or multi-class treatment raises ValueError."""
    df, _ = sample_causal_data
    invalid_df = df.copy()
    invalid_df.loc[0, "treatment"] = 2  # Invalid treatment value

    with pytest.raises(ValueError, match="binary values"):
        compute_propensity_and_balance(invalid_df)


# =========================================================================
# 2. DML Execution & Customer-Grouped Cross-Fitting Tests
# =========================================================================

def test_linear_dml_with_grouped_cross_fitting(sample_causal_data):
    """Verify LinearDML fits with customer grouping and estimates ATE and CATE."""
    df, gt_df = sample_causal_data
    model, preprocessor = fit_linear_dml(df, n_splits=3, random_state=42)

    X = preprocessor.transform(df[CAUSAL_FEATURES])
    cate_effects = model.effect(X, T0=0, T1=1).reshape(-1)

    assert len(cate_effects) == len(df)
    ate_estimate = float(np.mean(cate_effects))
    assert 0.0 <= ate_estimate <= 0.20

    # 95% Confidence Interval for ATE
    ate_ci = model.ate_interval(X=X, T0=0, T1=1, alpha=0.05)
    ci_lower = float(np.asarray(ate_ci[0]).squeeze())
    ci_upper = float(np.asarray(ate_ci[1]).squeeze())
    assert ci_lower < ate_estimate < ci_upper


def test_doubly_robust_learner(sample_causal_data):
    """Verify Doubly Robust Learner (DRLearner) estimates ATE and CATE."""
    df, _ = sample_causal_data
    dr_model, preprocessor = fit_doubly_robust_learner(df, n_splits=3, random_state=42)

    X = preprocessor.transform(df[CAUSAL_FEATURES])
    cate_effects = dr_model.effect(X).reshape(-1)

    assert len(cate_effects) == len(df)
    dr_ate = float(np.mean(cate_effects))
    assert 0.0 <= dr_ate <= 0.25


def test_grouped_cross_fitting_no_customer_overlap(sample_causal_data):
    """Verify that GroupKFold splits never place the same customer in train and validation folds."""
    df, _ = sample_causal_data
    groups = df["customer_id"].values
    gkf = GroupKFold(n_splits=3)

    for train_idx, val_idx in gkf.split(df, groups=groups):
        train_custs = set(df.iloc[train_idx]["customer_id"])
        val_custs = set(df.iloc[val_idx]["customer_id"])
        assert len(train_custs & val_custs) == 0, "Customer leakage detected across cross-fitting folds!"


# =========================================================================
# 3. Ground-Truth Benchmark Evaluation Tests
# =========================================================================

def test_ground_truth_benchmark_evaluation(sample_causal_data):
    """Verify ground-truth evaluation metrics (ATE error, PEHE, correlation, CI coverage)."""
    df, gt_df = sample_causal_data
    model, preprocessor = fit_linear_dml(df, n_splits=3, random_state=42)

    eval_result = evaluate_against_ground_truth(model, preprocessor, df, gt_df)

    assert "ate_true" in eval_result
    assert "ate_pred" in eval_result
    assert "ate_error" in eval_result
    assert "cate_mae" in eval_result
    assert "pehe" in eval_result
    assert "effect_correlation" in eval_result
    assert "ci_coverage_95" in eval_result

    assert eval_result["pehe"] >= 0.0
    assert eval_result["ate_error"] >= 0.0
    # CI coverage should be high (> 80%)
    assert eval_result["ci_coverage_95"] >= 0.80


# =========================================================================
# 4. DoWhy Refutations Tests
# =========================================================================

def test_dowhy_refutations_execution(sample_causal_data):
    """Verify DoWhy causal identification and refutation tests execute cleanly."""
    df, _ = sample_causal_data
    refutations = run_dowhy_refutations(df)

    assert "estimated_base_effect" in refutations
    assert "random_common_cause" in refutations
    assert "placebo_treatment" in refutations
    assert "data_subset" in refutations

    # Placebo treatment effect must be near zero
    assert abs(refutations["placebo_treatment"]["new_effect"]) < 0.05
    assert refutations["placebo_treatment"]["passed"] is True


# =========================================================================
# 5. Heterogeneity Explainability Tests
# =========================================================================

def test_heterogeneity_and_segment_explainability(sample_causal_data):
    """Verify segment explainability for category and customer history tiers."""
    df, _ = sample_causal_data
    model, preprocessor = fit_linear_dml(df, n_splits=3, random_state=42)

    explain = explain_heterogeneity_and_segments(model, preprocessor, df)

    assert "category_effects" in explain
    assert "history_tier_effects" in explain
    assert "top_responsive_segments" in explain
    assert "lowest_responsive_segments" in explain
    assert "causal_explanation" in explain

    # Category effects should include all categories
    for cat in ["Electronics", "Fashion", "Beauty", "Home"]:
        assert cat in explain["category_effects"]
        assert "mean_effect" in explain["category_effects"][cat]
        assert "ci_95" in explain["category_effects"][cat]


# =========================================================================
# 6. Reproducibility & Persisted Artifacts Tests
# =========================================================================

def test_causal_estimation_reproducibility(sample_causal_data):
    """Verify that identical random seeds yield reproducible CATE estimates."""
    df, _ = sample_causal_data

    m1, p1 = fit_linear_dml(df, n_splits=3, random_state=42)
    X1 = p1.transform(df[CAUSAL_FEATURES])
    eff1 = m1.effect(X1)

    m2, p2 = fit_linear_dml(df, n_splits=3, random_state=42)
    X2 = p2.transform(df[CAUSAL_FEATURES])
    eff2 = m2.effect(X2)

    np.testing.assert_allclose(eff1, eff2, rtol=1e-5)


def test_persisted_causal_artifacts():
    """Verify that persisted causal artifact and evaluation report exist and load."""
    assert CAUSAL_MODEL_ARTIFACT.is_file(), f"Artifact not found: {CAUSAL_MODEL_ARTIFACT}"
    assert CAUSAL_ENGINE_REPORT.is_file(), f"Report not found: {CAUSAL_ENGINE_REPORT}"

    # Load model artifact
    pipeline = joblib.load(CAUSAL_MODEL_ARTIFACT)
    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

    # Check report content
    with open(CAUSAL_ENGINE_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "diagnostics" in report
    assert "doubly_robust_ate" in report
    assert "dowhy_refutations" in report
    assert "ground_truth_evaluation" in report
    assert "heterogeneity_explainability" in report
    assert "cate_correlation_investigation" in report

