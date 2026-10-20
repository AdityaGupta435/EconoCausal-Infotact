import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.econml_model import (
    FEATURES,
    SPLIT_DIR,
    load_training_data,
    make_model,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "data" / "synthetic" / "uplift_evaluation.json"


def qini_curve(outcomes, treatment, scores):
    """Observed cumulative Qini gain, sorted by predicted uplift."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(outcomes, dtype=float)[order]
    t = np.asarray(treatment, dtype=int)[order]

    cumulative_treated = np.cumsum(t == 1)
    cumulative_control = np.cumsum(t == 0)
    cumulative_treated_outcomes = np.cumsum(y * (t == 1))
    cumulative_control_outcomes = np.cumsum(y * (t == 0))

    valid = (cumulative_treated > 0) & (cumulative_control > 0)
    if valid.sum() < 2:
        raise ValueError("Insufficient treated/control observations for Qini.")

    gains = (
        cumulative_treated_outcomes[valid]
        - cumulative_control_outcomes[valid]
        * cumulative_treated[valid]
        / cumulative_control[valid]
    )
    coverage = np.flatnonzero(valid) / max(len(y) - 1, 1)
    return coverage, gains


def qini_area(outcomes, treatment, scores):
    coverage, gains = qini_curve(outcomes, treatment, scores)
    return float(np.trapezoid(gains, coverage))


def main():
    train, X_train, T_train, Y_train, preprocessor = load_training_data()

    test = pd.read_csv(SPLIT_DIR / "test.csv")
    truth = pd.read_csv(SPLIT_DIR / "test_ground_truth.csv")

    if len(test) != len(truth):
        raise ValueError("Test and ground-truth row counts do not match.")
    if not test[FEATURES + ["treatment", "purchase"]].notna().all().all():
        raise ValueError("Test data contains missing required values.")
    if not truth["true_effect"].notna().all():
        raise ValueError("Ground-truth effects contain missing values.")

    X_test = np.asarray(preprocessor.transform(test[FEATURES]), dtype=float)
    if not np.isfinite(X_test).all():
        raise ValueError("Encoded test features contain non-finite values.")

    model = make_model()
    model.fit(Y_train, T_train, X=X_train)
    predicted_effect = np.asarray(model.effect(X_test), dtype=float).reshape(-1)
    true_effect = truth["true_effect"].to_numpy(dtype=float)

    if len(predicted_effect) != len(test):
        raise ValueError("Predicted effects do not match test row count.")

    y = test["purchase"].to_numpy(dtype=float)
    t = test["treatment"].to_numpy(dtype=int)

    model_qini = qini_area(y, t, predicted_effect)

    rng = np.random.default_rng(42)
    random_qini = [
        qini_area(y, t, rng.permutation(predicted_effect))
        for _ in range(30)
    ]

    model_order = np.argsort(-predicted_effect)
    oracle_order = np.argsort(-true_effect)
    top_k = max(1, int(np.ceil(0.20 * len(test))))

    model_top_effect = float(np.mean(true_effect[model_order[:top_k]]))
    random_top_effect = float(np.mean(true_effect))
    oracle_top_effect = float(np.mean(true_effect[oracle_order[:top_k]]))

    report = {
        "evaluation": "synthetic held-out test set",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "treatment_rows": int(np.sum(t == 1)),
        "control_rows": int(np.sum(t == 0)),
        "predicted_ite_mae": float(np.mean(np.abs(predicted_effect - true_effect))),
        "predicted_ite_rmse": float(np.sqrt(np.mean((predicted_effect - true_effect) ** 2))),
        "observed_qini_area": model_qini,
        "random_qini_area_mean_30_runs": float(np.mean(random_qini)),
        "qini_area_difference_vs_random": float(model_qini - np.mean(random_qini)),
        "top_20_percent": {
            "model_ranked_mean_true_effect": model_top_effect,
            "random_targeting_mean_true_effect": random_top_effect,
            "oracle_ranked_mean_true_effect": oracle_top_effect,
        },
        "interpretation": (
            "Observed Qini is a noisy ranking diagnostic on this synthetic test set, "
            "not proof of real-world causal validity. True effects are used only "
            "for evaluation, never as model features."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
