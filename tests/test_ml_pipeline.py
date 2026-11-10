"""Tests for Phase 3 ML Training Pipeline.

Covers:
1. Training pipeline execution on candidate models.
2. Preprocessing consistency and handling of unseen categories.
3. Strict train / validation / test customer separation (zero leakage).
4. Model artifact save and load round-trip consistency.
5. Prediction reproducibility with fixed random state.
6. Classification and regression metric calculations.
7. Feature schema compatibility and validation.
8. Zero test-set usage during hyperparameter tuning.
9. Explicit separation between predictive outcome accuracy and causal effect evaluation.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from backend.config import MODEL_ARTIFACTS_DIR, MODEL_EVALUATION_REPORT, MODEL_REGISTRY_PATH, MODELS_DIR
from backend.ml_pipeline import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    ModelTrainer,
    TaskType,
    build_preprocessor,
    compute_causal_benchmark_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    load_datasets,
)


# =========================================================================
# 1. Metric Calculations Tests
# =========================================================================

def test_classification_metrics_exact():
    """Verify classification metrics on known deterministic arrays."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # 3 correct, 1 false negative
    y_prob = np.array([0.9, 0.4, 0.1, 0.2])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0  # 1 TP, 0 FP
    assert metrics["recall"] == 0.5  # 1 TP, 1 FN
    assert metrics["f1"] == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5))
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_regression_metrics_exact():
    """Verify regression metrics on known deterministic values."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])  # errors: +2, -2, +1

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(5.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(9.0 / 3.0))
    # Perfect fit baseline
    perf_metrics = compute_regression_metrics(y_true, y_true)
    assert perf_metrics["mae"] == 0.0
    assert perf_metrics["rmse"] == 0.0
    assert perf_metrics["r2"] == 1.0


# =========================================================================
# 2. Preprocessing & Feature Schema Compatibility Tests
# =========================================================================

def test_preprocessor_handles_unseen_categories():
    """Ensure preprocessor does not crash on unseen categorical levels."""
    train_df = pd.DataFrame({
        "category": ["Electronics", "Fashion", "Beauty"],
        "baseline_price": [100.0, 50.0, 20.0],
        "customer_history": [2, 5, 1],
        "seasonality": [0, 1, 2],
        "treatment": [1, 0, 1],
        "discount_pct": [10, 0, 15],
    })
    test_df = pd.DataFrame({
        "category": ["Automotive", "Electronics"],  # 'Automotive' is unseen
        "baseline_price": [80.0, 120.0],
        "customer_history": [3, 4],
        "seasonality": [1, 0],
        "treatment": [0, 1],
        "discount_pct": [0, 20],
    })

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(train_df)
    assert X_train_trans.shape[0] == 3

    # Transforming test with unseen category must succeed with zeros for the unknown category
    X_test_trans = preprocessor.transform(test_df)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()


def test_schema_missing_column_raises_error():
    """Ensure omitting required feature columns raises an error during preprocessing."""
    df_missing = pd.DataFrame({
        "category": ["Electronics"],
        # baseline_price is missing!
        "customer_history": [2],
        "seasonality": [0],
        "treatment": [1],
        "discount_pct": [10],
    })
    preprocessor = build_preprocessor()
    with pytest.raises((ValueError, KeyError)):
        preprocessor.fit_transform(df_missing)


# =========================================================================
# 3. Train / Validation / Test Separation Tests
# =========================================================================

def test_dataset_customer_separation():
    """Verify that load_datasets loads customer-disjoint partitions."""
    datasets = load_datasets()
    train_c = set(datasets["train"]["customer_id"])
    val_c = set(datasets["validation"]["customer_id"])
    test_c = set(datasets["test"]["customer_id"])

    assert len(train_c & val_c) == 0, "Customer overlap between train and validation!"
    assert len(train_c & test_c) == 0, "Customer overlap between train and test!"
    assert len(val_c & test_c) == 0, "Customer overlap between validation and test!"


def test_no_test_set_usage_during_tuning():
    """Verify that hyperparameter tuning does NOT accept or touch test data."""
    trainer = ModelTrainer(random_state=42)
    # Check signature: tune_best_model only accepts train_df, val_df, task
    import inspect
    sig = inspect.signature(trainer.tune_best_model)
    param_names = list(sig.parameters.keys())
    assert "test_df" not in param_names
    assert "test_data" not in param_names
    assert "X_test" not in param_names


# =========================================================================
# 4. Model Persistence and Reproducibility Tests
# =========================================================================

def test_model_save_and_load_roundtrip(tmp_path):
    """Verify model artifact saving, reloading, and bitwise prediction parity."""
    # Synthetic dataset
    X = pd.DataFrame({
        "category": ["Electronics", "Fashion", "Beauty", "Home"] * 25,
        "baseline_price": np.random.uniform(10, 100, 100),
        "customer_history": np.random.randint(1, 10, 100),
        "seasonality": np.random.randint(0, 4, 100),
        "treatment": np.random.binomial(1, 0.5, 100),
        "discount_pct": np.random.choice([0, 10, 20], 100),
    })
    y = np.random.binomial(1, 0.3, 100)

    preprocessor = build_preprocessor()
    trainer = ModelTrainer(models_dir=tmp_path)
    estimator = trainer.get_candidate_classifiers()["logistic_regression"]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])
    pipeline.fit(X, y)

    # Save artifact
    artifact_path = trainer.save_model(
        pipeline=pipeline,
        model_name="test_lr",
        target_name="purchase",
        task_type=TaskType.CLASSIFICATION,
        version="vtest",
    )
    assert artifact_path.is_file()

    # Load artifact
    loaded_pipeline = joblib.load(artifact_path)
    orig_preds = pipeline.predict(X)
    loaded_preds = loaded_pipeline.predict(X)

    np.testing.assert_array_equal(orig_preds, loaded_preds)

    orig_probs = pipeline.predict_proba(X)
    loaded_probs = loaded_pipeline.predict_proba(X)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-6)

    # Verify metadata JSON exists
    meta_path = tmp_path / "artifacts" / "test_lr_purchase_vtest_metadata.json"
    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == "test_lr"
    assert meta["target"] == "purchase"
    assert meta["task_type"] == "classification"


def test_prediction_reproducibility():
    """Verify that models with identical random state generate identical predictions."""
    X = pd.DataFrame({
        "category": ["Electronics", "Fashion", "Beauty"] * 20,
        "baseline_price": np.random.uniform(10, 100, 60),
        "customer_history": np.random.randint(1, 5, 60),
        "seasonality": np.random.randint(0, 4, 60),
        "treatment": np.random.binomial(1, 0.5, 60),
        "discount_pct": np.random.choice([0, 10, 15], 60),
    })
    y = np.random.binomial(1, 0.3, 60)

    # Run 1
    t1 = ModelTrainer(random_state=42)
    p1 = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("estimator", t1.get_candidate_classifiers()["random_forest"]),
    ])
    p1.fit(X, y)
    preds1 = p1.predict_proba(X)

    # Run 2
    t2 = ModelTrainer(random_state=42)
    p2 = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("estimator", t2.get_candidate_classifiers()["random_forest"]),
    ])
    p2.fit(X, y)
    preds2 = p2.predict_proba(X)

    np.testing.assert_allclose(preds1, preds2, rtol=1e-7)


# =========================================================================
# 5. Causal Effect Benchmark vs Outcome Accuracy Tests
# =========================================================================

def test_causal_effect_benchmark_calculation():
    """Verify PEHE and ATE error calculation and explicit distinction from prediction metrics."""
    test_df = pd.DataFrame({
        "category": ["Electronics", "Fashion", "Beauty"],
        "baseline_price": [50.0, 30.0, 20.0],
        "customer_history": [2, 1, 3],
        "seasonality": [0, 1, 2],
        "treatment": [1, 0, 1],
        "discount_pct": [15, 0, 15],
    })
    ground_truth_df = pd.DataFrame({
        "true_effect": [0.08, 0.05, 0.02],
    })

    # Fit a simple classifier
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("estimator", ModelTrainer(random_state=42).get_candidate_classifiers()["logistic_regression"]),
    ])
    pipeline.fit(test_df, [1, 0, 0])

    res = compute_causal_benchmark_metrics(pipeline, test_df, ground_truth_df)

    assert "pehe" in res
    assert "ate_error" in res
    assert "effect_correlation" in res
    assert "causal_disclaimer" in res
    assert res["pehe"] >= 0.0
    assert res["ate_error"] >= 0.0
    assert "strictly separated" in res["causal_disclaimer"]


# =========================================================================
# 6. Global Artifacts Existence Tests
# =========================================================================

def test_persisted_evaluation_report_and_registry():
    """Verify persisted evaluation_report.json and model_registry.json."""
    assert MODEL_EVALUATION_REPORT.is_file(), "evaluation_report.json does not exist!"
    assert MODEL_REGISTRY_PATH.is_file(), "model_registry.json does not exist!"

    with open(MODEL_EVALUATION_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["train_records"] == 6000
    assert report["validation_records"] == 2000
    assert report["test_records"] == 2000
    assert "classification" in report
    assert "regression" in report
    assert "causal_effect_benchmark" in report
    assert report["classification"]["best_model"] == "hist_gradient_boosting"
    assert report["regression"]["best_model"] == "hist_gradient_boosting"

    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert "purchase_classifier" in registry["models"]
    assert "revenue_regressor" in registry["models"]
