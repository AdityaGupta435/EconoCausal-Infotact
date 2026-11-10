import json
import joblib
import numpy as np
import pandas as pd
import pytest

from backend.config import (
    UPLIFT_DECILE_REPORT_PATH,
    UPLIFT_EVALUATION_PATH,
    UPLIFT_MODEL_ARTIFACT,
    UPLIFT_REPORT,
    UPLIFT_SCORED_CUSTOMERS_PATH,
)
from backend.uplift_analysis import (
    DMLUplift,
    SLearner,
    TLearner,
    XLearner,
    classify_customer_segments,
    compute_auuc_curve,
    compute_decile_analysis,
    compute_uplift_metrics,
    qini_area,
    qini_curve,
)


def test_qini_curve_returns_finite_values():
    outcomes = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    coverage, gains = qini_curve(outcomes, treatment, scores)

    assert len(coverage) == len(gains)
    assert len(coverage) >= 2
    assert np.isfinite(coverage).all()
    assert np.isfinite(gains).all()


def test_qini_area_is_finite():
    outcomes = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    assert np.isfinite(qini_area(outcomes, treatment, scores))


def test_qini_rejects_data_without_both_groups():
    outcomes = np.array([1, 0, 1, 0])
    treatment = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    with pytest.raises(ValueError, match="Insufficient"):
        qini_curve(outcomes, treatment, scores)


def test_compute_auuc_curve():
    outcomes = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    cov, rate_diff = compute_auuc_curve(outcomes, treatment, scores)
    assert len(cov) == len(rate_diff)
    assert len(cov) >= 2
    assert cov[-1] == 1.0
    assert np.isfinite(rate_diff).all()


def test_compute_uplift_metrics_structure():
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.3, size=n)
    t = np.random.binomial(1, 0.5, size=n)
    pred_uplift = np.random.normal(0.05, 0.02, size=n)
    true_tau = np.random.normal(0.05, 0.01, size=n)

    metrics = compute_uplift_metrics(y, t, pred_uplift, ground_truth_effects=true_tau)

    assert "observed_qini_area" in metrics
    assert "random_qini_area" in metrics
    assert "qini_score_gain_vs_random" in metrics
    assert "observed_auuc" in metrics
    assert "random_auuc" in metrics
    assert "auuc_score_gain_vs_random" in metrics
    assert "normalized_qini_coefficient" in metrics
    assert "beats_random_targeting" in metrics
    assert np.isfinite(metrics["observed_qini_area"])
    assert np.isfinite(metrics["observed_auuc"])


def test_compute_decile_analysis():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "treatment": np.random.binomial(1, 0.5, size=n),
        "purchase": np.random.binomial(1, 0.3, size=n),
        "true_effect": np.random.normal(0.05, 0.02, size=n),
    })
    scores = np.linspace(0.15, -0.05, n)

    df_deciles = compute_decile_analysis(df, scores, ground_truth_col="true_effect")
    assert len(df_deciles) == 10
    assert "decile" in df_deciles.columns
    assert "treated_response_rate" in df_deciles.columns
    assert "control_response_rate" in df_deciles.columns
    assert "decile_uplift" in df_deciles.columns
    assert "cumulative_uplift" in df_deciles.columns
    assert "mean_true_effect" in df_deciles.columns
    assert df_deciles["decile"].tolist() == list(range(1, 11))


def test_classify_customer_segments():
    pred_uplift = np.array([0.08, 0.01, 0.02, -0.05])
    p_baseline = np.array([0.25, 0.65, 0.15, 0.30])

    segments, recommendations = classify_customer_segments(
        pred_uplift, p_baseline, persuadable_threshold=0.05, sleeping_dog_threshold=-0.01
    )

    assert len(segments) == 4
    assert len(recommendations) == 4

    # 0: uplift (0.08) >= 0.05 -> Persuadable, Target
    assert segments[0] == "Persuadables"
    assert recommendations[0] == "Target"

    # 1: uplift (0.01) < 0.05, p0 (0.65) >= median (0.275) -> Sure Things, Do Not Target
    assert segments[1] == "Sure Things"
    assert recommendations[1] == "Do Not Target"

    # 2: uplift (0.02) < 0.05, p0 (0.15) < median (0.275) -> Lost Causes, Do Not Target
    assert segments[2] == "Lost Causes"
    assert recommendations[2] == "Do Not Target"

    # 3: uplift (-0.05) <= -0.01 -> Sleeping Dogs, Do Not Target
    assert segments[3] == "Sleeping Dogs"
    assert recommendations[3] == "Do Not Target"


def test_meta_learners_fit_predict():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "baseline_price": np.random.uniform(10, 100, n),
        "customer_history": np.random.randint(1, 10, n),
        "seasonality": np.random.randint(0, 4, n),
        "category": np.random.choice(["Electronics", "Fashion", "Beauty", "Home"], n),
    })
    t = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.3, n)

    # T-Learner
    t_learner = TLearner(random_state=42)
    t_learner.fit(X, y, t)
    tau_t = t_learner.predict(X)
    assert len(tau_t) == n
    assert np.isfinite(tau_t).all()

    # S-Learner
    s_learner = SLearner(random_state=42)
    s_learner.fit(X, y, t)
    tau_s = s_learner.predict(X)
    assert len(tau_s) == n
    assert np.isfinite(tau_s).all()

    # X-Learner
    x_learner = XLearner(random_state=42)
    x_learner.fit(X, y, t)
    tau_x = x_learner.predict(X)
    assert len(tau_x) == n
    assert np.isfinite(tau_x).all()


def test_dml_uplift_fit_predict_with_groups():
    np.random.seed(42)
    n = 60
    groups = np.repeat(np.arange(15), 4)  # 15 customers, 4 rows each
    X = pd.DataFrame({
        "baseline_price": np.random.uniform(10, 100, n),
        "customer_history": np.random.randint(1, 10, n),
        "seasonality": np.random.randint(0, 4, n),
        "category": np.random.choice(["Electronics", "Fashion", "Beauty", "Home"], n),
    })
    t = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.3, n)

    dml = DMLUplift(n_splits=2, random_state=42)
    dml.fit(X, y, t, groups=groups)
    tau_dml = dml.predict(X)

    assert len(tau_dml) == n
    assert np.isfinite(tau_dml).all()


def test_uplift_artifacts_and_metrics_integrity():
    assert UPLIFT_MODEL_ARTIFACT.exists(), f"Model artifact missing: {UPLIFT_MODEL_ARTIFACT}"
    assert UPLIFT_EVALUATION_PATH.exists(), f"Evaluation report missing: {UPLIFT_EVALUATION_PATH}"
    assert UPLIFT_SCORED_CUSTOMERS_PATH.exists(), f"Customer scores missing: {UPLIFT_SCORED_CUSTOMERS_PATH}"
    assert UPLIFT_DECILE_REPORT_PATH.exists(), f"Deciles report missing: {UPLIFT_DECILE_REPORT_PATH}"

    # Verify evaluation JSON contents
    with open(UPLIFT_EVALUATION_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    assert eval_data["beats_random_targeting"] is True
    assert eval_data["qini_area_difference_vs_random"] > 0
    assert "validation_model_comparison" in eval_data
    assert "DML-Uplift" in eval_data["validation_model_comparison"]
    assert eval_data["validation_model_comparison"]["DML-Uplift"]["beats_random"] is True

    # Verify customer scores CSV
    df_scores = pd.read_csv(UPLIFT_SCORED_CUSTOMERS_PATH)
    assert len(df_scores) == 2000
    expected_cols = {"customer_id", "predicted_uplift", "segment", "targeting_recommendation"}
    assert expected_cols.issubset(set(df_scores.columns))
    assert df_scores["predicted_uplift"].notnull().all()
    assert set(df_scores["segment"].unique()).issubset({
        "Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"
    })
    assert set(df_scores["targeting_recommendation"].unique()).issubset({"Target", "Do Not Target"})

    # Verify deciles CSV
    df_deciles = pd.read_csv(UPLIFT_DECILE_REPORT_PATH)
    assert len(df_deciles) == 10
    assert (df_deciles["decile"] == list(range(1, 11))).all()

    # Verify model artifact loads and predicts
    model = joblib.load(UPLIFT_MODEL_ARTIFACT)
    assert hasattr(model, "predict")
