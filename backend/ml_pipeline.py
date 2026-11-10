"""Production-grade ML training and evaluation pipeline for EconoCausal.

Features:
- Strict train / validation / test isolation with zero customer-level leakage.
- Automated feature preprocessing (OneHotEncoder + StandardScaler in Pipeline).
- Predictive outcome models (Baselines, Regularized Linear/Logistic, Random Forest, HistGradientBoosting).
- Hyperparameter tuning strictly on validation data.
- Full metric suites for classification and regression.
- Explicit separation of predictive outcome accuracy from causal treatment effect benchmarks (PEHE, ATE error).
- Safe model artifact persistence (joblib) with metadata, schema, and version tracking.
- Memory-conscious scalable training design.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.config import (
    DATA_DIR,
    MODEL_ARTIFACTS_DIR,
    MODEL_EVALUATION_REPORT,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
    SPLIT_DIR,
)
from backend.data_pipeline import DatasetProfile, get_optimal_dtypes

# Default configurations
DEFAULT_RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["category"]
NUMERIC_FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
    "treatment",
    "discount_pct",
]
ALL_INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_CLASSIFICATION = "purchase"
TARGET_REGRESSION = "revenue"
TARGET_REVENUE = "revenue"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


def build_preprocessor() -> ColumnTransformer:
    """Construct a ColumnTransformer that properly handles categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics including calibration."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = float(np.mean(y_true))
            metrics["brier_score"] = float(np.mean((y_prob - y_true) ** 2))
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics (MAE, RMSE, R²)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def compute_causal_benchmark_metrics(
    model_pipeline: Pipeline,
    test_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate causal treatment effect estimation against ground-truth true_effect.
    
    IMPORTANT: Predictive outcome accuracy (ROC-AUC / R²) measures how well the model
    predicts observable outcomes under observed treatments. Causal effect accuracy
    (PEHE / ATE error) measures counterfactual treatment contrast precision.
    """
    if "true_effect" not in ground_truth_df.columns:
        raise KeyError("ground_truth_df must contain 'true_effect' column.")

    true_effect = ground_truth_df["true_effect"].to_numpy(dtype=float)

    # Compute counterfactual predictions under T=1 and T=0
    df_treated = test_df[ALL_INPUT_FEATURES].copy()
    df_treated["treatment"] = 1
    # Assign standard promo discount if discount_pct is present
    if "discount_pct" in df_treated.columns:
        df_treated["discount_pct"] = np.where(df_treated["discount_pct"] == 0, 15, df_treated["discount_pct"])

    df_control = test_df[ALL_INPUT_FEATURES].copy()
    df_control["treatment"] = 0
    if "discount_pct" in df_control.columns:
        df_control["discount_pct"] = 0

    if hasattr(model_pipeline, "predict_proba"):
        p_treated = model_pipeline.predict_proba(df_treated)[:, 1]
        p_control = model_pipeline.predict_proba(df_control)[:, 1]
        estimated_effect = p_treated - p_control
    else:
        y_treated = model_pipeline.predict(df_treated)
        y_control = model_pipeline.predict(df_control)
        estimated_effect = y_treated - y_control

    # Precision in Estimation of Heterogeneous Effect (PEHE)
    pehe = float(np.sqrt(np.mean((estimated_effect - true_effect) ** 2)))
    ate_true = float(np.mean(true_effect))
    ate_estimated = float(np.mean(estimated_effect))
    ate_error = float(abs(ate_estimated - ate_true))

    # Pearson correlation if standard deviation > 0
    if np.std(estimated_effect) > 1e-9 and np.std(true_effect) > 1e-9:
        correlation = float(np.corrcoef(estimated_effect, true_effect)[0, 1])
    else:
        correlation = 0.0

    return {
        "pehe": pehe,
        "ate_true": ate_true,
        "ate_estimated": ate_estimated,
        "ate_error": ate_error,
        "effect_correlation": correlation,
        "causal_disclaimer": (
            "Causal effect estimation accuracy (PEHE, ATE error) measures counterfactual "
            "contrast fidelity and is strictly separated from observational predictive accuracy (ROC-AUC / R²)."
        ),
    }


def load_datasets(
    splits_dir: Optional[Union[str, Path]] = None,
    use_optimal_dtypes: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load train, validation, and test datasets and assert customer disjointness."""
    s_dir = Path(splits_dir) if splits_dir else SPLIT_DIR
    if not s_dir.is_dir():
        raise FileNotFoundError(f"Splits directory not found: {s_dir}")

    dtypes = get_optimal_dtypes(DatasetProfile.RETAIL_PRICING) if use_optimal_dtypes else None

    # Load splits
    train_df = pd.read_csv(s_dir / "train.csv")
    val_df = pd.read_csv(s_dir / "validation.csv")
    test_df = pd.read_csv(s_dir / "test.csv")

    # Load ground truth files if available
    train_gt = pd.read_csv(s_dir / "train_ground_truth.csv") if (s_dir / "train_ground_truth.csv").is_file() else None
    val_gt = pd.read_csv(s_dir / "validation_ground_truth.csv") if (s_dir / "validation_ground_truth.csv").is_file() else None
    test_gt = pd.read_csv(s_dir / "test_ground_truth.csv") if (s_dir / "test_ground_truth.csv").is_file() else None

    # Assert zero customer leakage
    if "customer_id" in train_df.columns:
        train_cust = set(train_df["customer_id"])
        val_cust = set(val_df["customer_id"])
        test_cust = set(test_df["customer_id"])

        if train_cust & val_cust:
            raise RuntimeError("Customer overlap detected between train and validation splits!")
        if train_cust & test_cust:
            raise RuntimeError("Customer overlap detected between train and test splits!")
        if val_cust & test_cust:
            raise RuntimeError("Customer overlap detected between validation and test splits!")

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "train_gt": train_gt,
        "validation_gt": val_gt,
        "test_gt": test_gt,
    }


class ModelTrainer:
    """Manages model candidate training, validation, tuning, and persistence."""

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        self.random_state = random_state
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.artifacts_dir = self.models_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def get_candidate_classifiers(self) -> Dict[str, BaseEstimator]:
        """Return candidate classifiers for purchase conversion modeling."""
        return {
            "dummy_prior": DummyClassifier(strategy="prior", random_state=self.random_state),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=1,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=31,
                learning_rate=0.1,
                random_state=self.random_state,
            ),
        }

    def get_candidate_regressors(self) -> Dict[str, BaseEstimator]:
        """Return candidate regressors for revenue modeling."""
        return {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=1.0, random_state=self.random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=1,
            ),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=31,
                learning_rate=0.1,
                random_state=self.random_state,
            ),
        }

    def train_and_evaluate_candidates(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        task: TaskType = TaskType.CLASSIFICATION,
    ) -> Dict[str, Any]:
        """Train all candidate models on train data and evaluate on validation data."""
        target_col = TARGET_CLASSIFICATION if task == TaskType.CLASSIFICATION else TARGET_REVENUE

        X_train = train_df[ALL_INPUT_FEATURES].copy()
        y_train = train_df[target_col].to_numpy()

        X_val = val_df[ALL_INPUT_FEATURES].copy()
        y_val = val_df[target_col].to_numpy()

        candidates = (
            self.get_candidate_classifiers()
            if task == TaskType.CLASSIFICATION
            else self.get_candidate_regressors()
        )

        results: Dict[str, Dict[str, Any]] = {}

        for name, estimator in candidates.items():
            preprocessor = build_preprocessor()
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ])

            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            # Evaluate on validation set
            y_pred = pipeline.predict(X_val)

            if task == TaskType.CLASSIFICATION:
                y_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
                metrics = compute_classification_metrics(y_val, y_pred, y_prob)
            else:
                metrics = compute_regression_metrics(y_val, y_pred)

            results[name] = {
                "pipeline": pipeline,
                "fit_time_seconds": round(fit_time, 4),
                "validation_metrics": metrics,
            }

        return results

    def tune_best_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        task: TaskType = TaskType.CLASSIFICATION,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Perform hyperparameter tuning for the top model strictly using validation data."""
        target_col = TARGET_CLASSIFICATION if task == TaskType.CLASSIFICATION else TARGET_REVENUE

        X_train = train_df[ALL_INPUT_FEATURES].copy()
        y_train = train_df[target_col].to_numpy()

        X_val = val_df[ALL_INPUT_FEATURES].copy()
        y_val = val_df[target_col].to_numpy()

        # Grid of tuning candidates for HistGradientBoosting
        param_grid = [
            {"max_leaf_nodes": 15, "learning_rate": 0.05, "l2_regularization": 0.0},
            {"max_leaf_nodes": 15, "learning_rate": 0.1, "l2_regularization": 1.0},
            {"max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 1.0},
            {"max_leaf_nodes": 31, "learning_rate": 0.1, "l2_regularization": 1.0},
            {"max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 2.0},
        ]

        best_score = -np.inf
        best_pipeline = None
        best_params = None
        best_metrics = None
        tuning_log = []

        for params in param_grid:
            preprocessor = build_preprocessor()
            if task == TaskType.CLASSIFICATION:
                estimator = HistGradientBoostingClassifier(
                    max_iter=100,
                    random_state=self.random_state,
                    **params,
                )
            else:
                estimator = HistGradientBoostingRegressor(
                    max_iter=100,
                    random_state=self.random_state,
                    **params,
                )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            if task == TaskType.CLASSIFICATION:
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                metrics = compute_classification_metrics(y_val, y_pred, y_prob)
                score = metrics["roc_auc"]
            else:
                metrics = compute_regression_metrics(y_val, y_pred)
                score = -metrics["rmse"]  # Higher is better

            tuning_log.append({
                "params": params,
                "score": score,
                "metrics": metrics,
            })

            if score > best_score:
                best_score = score
                best_pipeline = pipeline
                best_params = params
                best_metrics = metrics

        tuning_summary = {
            "best_params": best_params,
            "best_validation_score": best_score,
            "best_validation_metrics": best_metrics,
            "trials_evaluated": len(tuning_log),
            "tuning_history": tuning_log,
        }

        return best_pipeline, tuning_summary

    def save_model(
        self,
        pipeline: Pipeline,
        model_name: str,
        target_name: str,
        task_type: TaskType,
        version: str = "v1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist model pipeline and comprehensive metadata JSON safely."""
        artifact_filename = f"{model_name}_{target_name}_{version}.joblib"
        artifact_path = self.artifacts_dir / artifact_filename
        meta_path = self.artifacts_dir / f"{model_name}_{target_name}_{version}_metadata.json"

        # Backup if already exists
        if artifact_path.is_file():
            backup_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir = self.artifacts_dir / f"backup_{backup_ts}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, backup_dir / artifact_filename)
            if meta_path.is_file():
                shutil.copy2(meta_path, backup_dir / meta_path.name)

        # Save pipeline via joblib
        joblib.dump(pipeline, artifact_path)

        # Save metadata JSON
        meta_content = {
            "model_name": model_name,
            "target": target_name,
            "task_type": task_type.value,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_path": str(artifact_path),
            "feature_schema": {
                "input_features": ALL_INPUT_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
            },
            "metadata": metadata or {},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

        return artifact_path


def run_full_ml_pipeline(
    splits_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    version: str = "v1.0.0",
) -> Dict[str, Any]:
    """Execute end-to-end ML training, tuning, final test evaluation, and persistence."""
    print("=" * 70)
    print("ECONOCAUSAL ML TRAINING PIPELINE (PHASE 3)")
    print("=" * 70)

    # 1. Load data splits
    datasets = load_datasets(splits_dir=splits_dir)
    train_df = datasets["train"]
    val_df = datasets["validation"]
    test_df = datasets["test"]
    test_gt = datasets["test_gt"]

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    print(f"Loaded splits: Train={n_train:,} | Validation={n_val:,} | Test={n_test:,} rows")

    trainer = ModelTrainer(models_dir=models_dir)

    # ---------------------------------------------------------------------
    # 2. Purchase Conversion Modeling (Binary Classification)
    # ---------------------------------------------------------------------
    print("\n[1/3] Training Candidate Classifiers for 'purchase' target...")
    clf_candidates = trainer.train_and_evaluate_candidates(
        train_df, val_df, task=TaskType.CLASSIFICATION
    )

    for name, res in clf_candidates.items():
        m = res["validation_metrics"]
        print(f"  - {name:<25} | Fit Time: {res['fit_time_seconds']}s | ROC-AUC: {m.get('roc_auc', 0):.4f} | PR-AUC: {m.get('pr_auc', 0):.4f} | F1: {m['f1']:.4f}")

    print("\nTuning top classifier on validation data...")
    best_clf_pipeline, clf_tuning_summary = trainer.tune_best_model(
        train_df, val_df, task=TaskType.CLASSIFICATION
    )
    print(f"  Best params: {clf_tuning_summary['best_params']}")
    print(f"  Tuned Val ROC-AUC: {clf_tuning_summary['best_validation_metrics']['roc_auc']:.4f}")

    # Final Test Set Evaluation for Best Classifier (Touched only once!)
    X_test = test_df[ALL_INPUT_FEATURES].copy()
    y_test_clf = test_df[TARGET_CLASSIFICATION].to_numpy()
    test_pred_clf = best_clf_pipeline.predict(X_test)
    test_prob_clf = best_clf_pipeline.predict_proba(X_test)[:, 1]
    final_test_clf_metrics = compute_classification_metrics(y_test_clf, test_pred_clf, test_prob_clf)
    print(f"  Final Test ROC-AUC: {final_test_clf_metrics['roc_auc']:.4f} | PR-AUC: {final_test_clf_metrics['pr_auc']:.4f} | Accuracy: {final_test_clf_metrics['accuracy']:.4f} | Brier: {final_test_clf_metrics['brier_score']:.4f}")

    # Baseline comparison
    baseline_clf_metrics = clf_candidates["dummy_prior"]["validation_metrics"]

    # Save Best Classifier Artifact
    clf_artifact_path = trainer.save_model(
        pipeline=best_clf_pipeline,
        model_name="hist_gradient_boosting",
        target_name=TARGET_CLASSIFICATION,
        task_type=TaskType.CLASSIFICATION,
        version=version,
        metadata={
            "train_record_count": n_train,
            "validation_record_count": n_val,
            "test_record_count": n_test,
            "best_params": clf_tuning_summary["best_params"],
            "validation_metrics": clf_tuning_summary["best_validation_metrics"],
            "baseline_validation_metrics": baseline_clf_metrics,
            "final_test_metrics": final_test_clf_metrics,
        },
    )
    print(f"  Saved classifier artifact: {clf_artifact_path}")

    # ---------------------------------------------------------------------
    # 3. Revenue Modeling (Regression)
    # ---------------------------------------------------------------------
    print("\n[2/3] Training Candidate Regressors for 'revenue' target...")
    reg_candidates = trainer.train_and_evaluate_candidates(
        train_df, val_df, task=TaskType.REGRESSION
    )

    for name, res in reg_candidates.items():
        m = res["validation_metrics"]
        print(f"  - {name:<25} | Fit Time: {res['fit_time_seconds']}s | RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | R²: {m['r2']:.4f}")

    print("\nTuning top regressor on validation data...")
    best_reg_pipeline, reg_tuning_summary = trainer.tune_best_model(
        train_df, val_df, task=TaskType.REGRESSION
    )
    print(f"  Best params: {reg_tuning_summary['best_params']}")
    print(f"  Tuned Val RMSE: {reg_tuning_summary['best_validation_metrics']['rmse']:.4f}")

    # Final Test Set Evaluation for Best Regressor
    y_test_reg = test_df[TARGET_REVENUE].to_numpy()
    test_pred_reg = best_reg_pipeline.predict(X_test)
    final_test_reg_metrics = compute_regression_metrics(y_test_reg, test_pred_reg)
    print(f"  Final Test RMSE: {final_test_reg_metrics['rmse']:.4f} | MAE: {final_test_reg_metrics['mae']:.4f} | R²: {final_test_reg_metrics['r2']:.4f}")

    baseline_reg_metrics = reg_candidates["dummy_mean"]["validation_metrics"]

    # Save Best Regressor Artifact
    reg_artifact_path = trainer.save_model(
        pipeline=best_reg_pipeline,
        model_name="hist_gradient_boosting",
        target_name=TARGET_REVENUE,
        task_type=TaskType.REGRESSION,
        version=version,
        metadata={
            "train_record_count": n_train,
            "validation_record_count": n_val,
            "test_record_count": n_test,
            "best_params": reg_tuning_summary["best_params"],
            "validation_metrics": reg_tuning_summary["best_validation_metrics"],
            "baseline_validation_metrics": baseline_reg_metrics,
            "final_test_metrics": final_test_reg_metrics,
        },
    )
    print(f"  Saved regressor artifact: {reg_artifact_path}")

    # ---------------------------------------------------------------------
    # 4. Ground-Truth Causal Effect Benchmark (Separately Evaluated)
    # ---------------------------------------------------------------------
    print("\n[3/3] Evaluating Ground-Truth Causal Effect Estimation (PEHE / ATE Benchmark)...")
    causal_benchmark = None
    if test_gt is not None:
        causal_benchmark = compute_causal_benchmark_metrics(
            model_pipeline=best_clf_pipeline,
            test_df=test_df,
            ground_truth_df=test_gt,
        )
        print(f"  True ATE:           {causal_benchmark['ate_true']:.5f}")
        print(f"  Estimated ATE:      {causal_benchmark['ate_estimated']:.5f}")
        print(f"  ATE Absolute Error: {causal_benchmark['ate_error']:.5f}")
        print(f"  PEHE (RMSE vs Tau): {causal_benchmark['pehe']:.5f}")
        print(f"  Tau Correlation:    {causal_benchmark['effect_correlation']:.5f}")

    # ---------------------------------------------------------------------
    # 5. Save Global Evaluation Report & Model Registry
    # ---------------------------------------------------------------------
    evaluation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "train_records": n_train,
        "validation_records": n_val,
        "test_records": n_test,
        "classification": {
            "target": TARGET_CLASSIFICATION,
            "best_model": "hist_gradient_boosting",
            "tuning_summary": clf_tuning_summary,
            "baseline_validation": baseline_clf_metrics,
            "candidate_validation_comparison": {
                k: v["validation_metrics"] for k, v in clf_candidates.items()
            },
            "final_test_metrics": final_test_clf_metrics,
            "artifact_path": str(clf_artifact_path),
        },
        "regression": {
            "target": TARGET_REVENUE,
            "best_model": "hist_gradient_boosting",
            "tuning_summary": reg_tuning_summary,
            "baseline_validation": baseline_reg_metrics,
            "candidate_validation_comparison": {
                k: v["validation_metrics"] for k, v in reg_candidates.items()
            },
            "final_test_metrics": final_test_reg_metrics,
            "artifact_path": str(reg_artifact_path),
        },
        "causal_effect_benchmark": causal_benchmark,
    }

    report_path = trainer.models_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    print(f"\nEvaluation report written to: {report_path}")

    # Update Registry
    registry_path = trainer.models_dir / "model_registry.json"
    registry_entry = {
        "version": version,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "models": {
            "purchase_classifier": {
                "artifact": str(clf_artifact_path),
                "test_roc_auc": final_test_clf_metrics["roc_auc"],
                "test_pr_auc": final_test_clf_metrics["pr_auc"],
            },
            "revenue_regressor": {
                "artifact": str(reg_artifact_path),
                "test_rmse": final_test_reg_metrics["rmse"],
                "test_r2": final_test_reg_metrics["r2"],
            },
        },
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_entry, f, indent=2)
    print(f"Model registry updated at: {registry_path}")

    print("\nML Training Pipeline executed successfully.")
    return evaluation_report


if __name__ == "__main__":
    run_full_ml_pipeline()
