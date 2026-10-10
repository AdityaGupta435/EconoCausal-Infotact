
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic" / "splits"

FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
    "category",
]

CATEGORICAL_FEATURES = ["category"]
NUMERIC_FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def make_outcome_model():
    return make_pipeline(
        make_preprocessor(),
        HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def make_treatment_model():
    return make_pipeline(
        make_preprocessor(),
        HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def estimate_ate(data, n_splits=5):
    x = data[FEATURES].reset_index(drop=True)
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["purchase"].to_numpy(dtype=float)

    y_residual = np.zeros(len(data))
    t_residual = np.zeros(len(data))

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    for train_idx, holdout_idx in splitter.split(x):
        x_train = x.iloc[train_idx]
        x_holdout = x.iloc[holdout_idx]

        outcome_model = make_outcome_model()
        treatment_model = make_treatment_model()

        outcome_model.fit(x_train, outcome[train_idx])
        treatment_model.fit(x_train, treatment[train_idx])

        y_residual[holdout_idx] = (
            outcome[holdout_idx]
            - outcome_model.predict(x_holdout)
        )

        propensity = treatment_model.predict_proba(
            x_holdout
        )[:, 1]

        # Avoid numerical instability from extreme propensity scores.
        propensity = np.clip(propensity, 0.01, 0.99)

        t_residual[holdout_idx] = (
            treatment[holdout_idx] - propensity
        )

    denominator = np.sum(t_residual**2)

    if denominator < 1e-10:
        raise ValueError("Insufficient residual treatment variation.")

    ate = np.sum(t_residual * y_residual) / denominator

    return float(ate)


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_truth = pd.read_csv(
        DATA_DIR / "test_ground_truth.csv"
    )

    if not (len(test_df) == len(test_truth)):
        raise ValueError("Test data and ground truth do not align.")

    estimate = estimate_ate(train_df)

    # Evaluation-only column: never used as a model feature.
    true_ate = float(test_truth["true_effect"].mean())

    absolute_error = abs(estimate - true_ate)

    print("ECONOCAUSAL — DML BASELINE")
    print("-" * 38)
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Estimated ATE: {estimate:.4f}")
    print(f"Estimated effect: {estimate * 100:.2f} percentage points")
    print(f"Test ground-truth ATE: {true_ate:.4f}")
    print(f"Absolute ATE error: {absolute_error:.4f}")
    print(f"Absolute error in percentage points: {absolute_error * 100:.2f}")

    print(
        "\nNote: The test ground truth is used only for evaluation. "
        "The DML estimate is trained on the training split."
    )


if __name__ == "__main__":
    main()
