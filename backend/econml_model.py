
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "synthetic" / "splits"

NUMERIC_FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_training_data():
    train = pd.read_csv(SPLIT_DIR / "train.csv")

    required = FEATURES + ["treatment", "purchase"]
    missing = sorted(set(required) - set(train.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if train[required].isnull().any().any():
        raise ValueError("Training data contains missing required values.")

    raw_X = train[FEATURES].copy()

    # Fit preprocessing on training covariates only.
    preprocessor = make_preprocessor()
    X = preprocessor.fit_transform(raw_X)
    X = np.asarray(X, dtype=float)

    T = train["treatment"].to_numpy(dtype=int)
    Y = train["purchase"].to_numpy(dtype=float)

    if set(np.unique(T)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")

    if not set(np.unique(Y)).issubset({0.0, 1.0}):
        raise ValueError("Purchase outcome must be binary.")

    if not np.isfinite(X).all():
        raise ValueError("Encoded features contain non-finite values.")

    return train, X, T, Y, preprocessor


def make_model():
    return LinearDML(
        model_y=HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
        model_t=HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
        discrete_treatment=True,
        cv=3,
        random_state=42,
    )



def main():
    train, X_train, T_train, Y_train, preprocessor = load_training_data()

    model = make_model()
    model.fit(Y_train, T_train, X=X_train)

    # Load held-out test data. Do not use it for model fitting.
    split_dir = SPLIT_DIR
    test = pd.read_csv(split_dir / "test.csv")
    ground_truth = pd.read_csv(split_dir / "test_ground_truth.csv")

    required = FEATURES + ["treatment", "purchase"]
    if test[required].isnull().any().any():
        raise ValueError("Test data contains missing required values.")

    if len(test) != len(ground_truth):
        raise ValueError("Test and ground-truth row counts do not match.")

    if ground_truth["true_effect"].isnull().any():
        raise ValueError("Ground-truth effects contain missing values.")

    X_test_raw = test[FEATURES].copy()
    X_test = np.asarray(
        preprocessor.transform(X_test_raw),
        dtype=float,
    )

    T_test = test["treatment"].to_numpy(dtype=int)
    Y_test = test["purchase"].to_numpy(dtype=float)
    true_effects = ground_truth["true_effect"].to_numpy(dtype=float)

    if not np.isfinite(X_test).all():
        raise ValueError("Encoded test features contain non-finite values.")

    # Predict treatment effects for unseen rows.
    predicted_effects = np.asarray(
        model.effect(X_test, T0=0, T1=1)
    ).reshape(-1)

    if len(predicted_effects) != len(true_effects):
        raise ValueError("Prediction and ground-truth lengths do not match.")

    predicted_ate = float(np.mean(predicted_effects))
    true_ate = float(np.mean(true_effects))
    ate_absolute_error = abs(predicted_ate - true_ate)

    ite_mae = float(np.mean(np.abs(predicted_effects - true_effects)))
    ite_rmse = float(
        np.sqrt(np.mean((predicted_effects - true_effects) ** 2))
    )
    report = {
        "model": "EconML LinearDML",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "encoded_features": int(X_train.shape[1]),
        "predicted_test_ate": predicted_ate,
        "ground_truth_test_ate": true_ate,
        "ate_absolute_error": ate_absolute_error,
        "ite_mae": ite_mae,
        "ite_rmse": ite_rmse,
    }

    report_path = ROOT / "data" / "synthetic" / "econml_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        __import__("json").dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluation report saved: {report_path}")

    print("ECONOCAUSAL — ECONML LINEAR DML")
    print("-" * 46)
    print(f"Training rows: {len(train):,}")
    print(f"Encoded feature count: {X_train.shape[1]}")
    print(f"Test rows: {len(test):,}")
    print(f"Test control (0): {int(np.sum(T_test == 0)):,}")
    print(f"Test treatment (1): {int(np.sum(T_test == 1)):,}")
    print()
    print("HELD-OUT TEST EVALUATION")
    print(f"Predicted test ATE: {predicted_ate:.4f}")
    print(f"Ground-truth test ATE: {true_ate:.4f}")
    print(f"ATE absolute error: {ate_absolute_error:.4f}")
    print(f"ITE MAE: {ite_mae:.4f}")
    print(f"ITE RMSE: {ite_rmse:.4f}")
    print()
    print(
        "Interpretation: ITE metrics compare estimated effects with "
        "known synthetic effects. They evaluate this synthetic benchmark, "
        "not real-world causal validity."
    )


if __name__ == "__main__":
    main()
