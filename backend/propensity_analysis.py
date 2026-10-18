
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "synthetic" / "splits"
OUTPUT_DIR = ROOT / "data" / "synthetic"

NUMERIC_FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    required = FEATURES + ["treatment", "purchase"]

    for name, frame in [("training", train), ("test", test)]:
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing columns: {missing}")
        if frame[required].isnull().any().any():
            raise ValueError(f"{name} data contains missing values.")

        if set(frame["treatment"].unique()) != {0, 1}:
            raise ValueError(f"{name} treatment must contain 0 and 1.")

    return train, test


def make_pipeline():
    preprocessing = ColumnTransformer(
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
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def calculate_standardized_mean_differences(data, propensity_scores):
    """Compare baseline covariate balance before and after weighting."""
    treatment = data["treatment"].to_numpy(dtype=int)
    results = []

    # Stabilized inverse-propensity weights for a descriptive balance check.
    epsilon = 0.01
    scores = np.clip(propensity_scores, epsilon, 1 - epsilon)

    weights = np.where(
        treatment == 1,
        1.0 / scores,
        1.0 / (1.0 - scores),
    )

    numeric_features = [
        "baseline_price",
        "customer_history",
        "seasonality",
    ]

    for feature in numeric_features:
        values = data[feature].to_numpy(dtype=float)
        treated = values[treatment == 1]
        control = values[treatment == 0]

        treated_mean = float(np.mean(treated))
        control_mean = float(np.mean(control))

        pooled_sd = float(
            np.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2)
        )

        before_smd = (
            (treated_mean - control_mean) / pooled_sd
            if pooled_sd > 0
            else 0.0
        )

        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_weights = weights[treated_mask]
        control_weights = weights[control_mask]

        weighted_treated_mean = float(
            np.average(values[treated_mask], weights=treated_weights)
        )
        weighted_control_mean = float(
            np.average(values[control_mask], weights=control_weights)
        )

        treated_weighted_var = float(
            np.average(
                (values[treated_mask] - weighted_treated_mean) ** 2,
                weights=treated_weights,
            )
        )
        control_weighted_var = float(
            np.average(
                (values[control_mask] - weighted_control_mean) ** 2,
                weights=control_weights,
            )
        )

        weighted_pooled_sd = float(
            np.sqrt((treated_weighted_var + control_weighted_var) / 2)
        )

        after_smd = (
            (weighted_treated_mean - weighted_control_mean)
            / weighted_pooled_sd
            if weighted_pooled_sd > 0
            else 0.0
        )

        results.append(
            {
                "feature": feature,
                "treated_mean_before": treated_mean,
                "control_mean_before": control_mean,
                "smd_before": float(before_smd),
                "smd_after_ipw": float(after_smd),
                "absolute_smd_before": abs(float(before_smd)),
                "absolute_smd_after_ipw": abs(float(after_smd)),
            }
        )

    # Also compare category proportions before and after weighting.
    for category in sorted(data["category"].dropna().unique()):
        indicator = (data["category"] == category).to_numpy(dtype=float)
        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_mean = float(np.mean(indicator[treated_mask]))
        control_mean = float(np.mean(indicator[control_mask]))

        pooled_sd = float(
            np.sqrt(
                (
                    np.var(indicator[treated_mask], ddof=1)
                    + np.var(indicator[control_mask], ddof=1)
                ) / 2
            )
        )

        before_smd = (
            (treated_mean - control_mean) / pooled_sd
            if pooled_sd > 0
            else 0.0
        )

        treated_weights = weights[treated_mask]
        control_weights = weights[control_mask]

        weighted_treated = float(
            np.average(indicator[treated_mask], weights=treated_weights)
        )
        weighted_control = float(
            np.average(indicator[control_mask], weights=control_weights)
        )

        treated_var = float(
            np.average(
                (indicator[treated_mask] - weighted_treated) ** 2,
                weights=treated_weights,
            )
        )
        control_var = float(
            np.average(
                (indicator[control_mask] - weighted_control) ** 2,
                weights=control_weights,
            )
        )
        weighted_pooled_sd = float(np.sqrt((treated_var + control_var) / 2))

        after_smd = (
            (weighted_treated - weighted_control) / weighted_pooled_sd
            if weighted_pooled_sd > 0
            else 0.0
        )

        results.append(
            {
                "feature": f"category={category}",
                "treated_mean_before": treated_mean,
                "control_mean_before": control_mean,
                "smd_before": float(before_smd),
                "smd_after_ipw": float(after_smd),
                "absolute_smd_before": abs(float(before_smd)),
                "absolute_smd_after_ipw": abs(float(after_smd)),
            }
        )

    return pd.DataFrame(results)




def overlap_summary(scores, treatment):
    rows = []

    for group, label in [(0, "Control"), (1, "Treatment")]:
        values = scores[treatment == group]

        rows.append(
            {
                "group": label,
                "sample_count": int(len(values)),
                "score_min": float(values.min()),
                "score_q01": float(np.quantile(values, 0.01)),
                "score_q25": float(np.quantile(values, 0.25)),
                "score_median": float(np.median(values)),
                "score_q75": float(np.quantile(values, 0.75)),
                "score_q99": float(np.quantile(values, 0.99)),
                "score_max": float(values.max()),
            }
        )

    return rows


def main():
    train, test = load_data()

    # Fit propensity model on training data only.
    propensity_model = make_pipeline()
    propensity_model.fit(train[FEATURES], train["treatment"])

    # Evaluate treatment assignment prediction on held-out test data.
    test_treatment = test["treatment"].to_numpy(dtype=int)
    test_scores = propensity_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    if not np.isfinite(test_scores).all():
        raise ValueError("Propensity scores contain non-finite values.")

    auc = float(roc_auc_score(test_treatment, test_scores))

    # Report score distributions separately for treatment and control.
    overlap_rows = overlap_summary(test_scores, test_treatment)

    balance_report = calculate_standardized_mean_differences(
        test,
        test_scores,
    )

    balance_path = OUTPUT_DIR / "propensity_balance_report.csv"
    balance_report.to_csv(balance_path, index=False)

    print("\nCOVARIATE BALANCE (STANDARDIZED MEAN DIFFERENCES)")
    print(balance_report.to_string(index=False))
    print(f"Balance report saved: {balance_path}")

    # Descriptive common-range diagnostic.
    control_scores = test_scores[test_treatment == 0]
    treated_scores = test_scores[test_treatment == 1]

    common_lower = max(control_scores.min(), treated_scores.min())
    common_upper = min(control_scores.max(), treated_scores.max())
    has_common_range = bool(common_lower <= common_upper)

    if has_common_range:
        control_in_range = (
            (control_scores >= common_lower)
            & (control_scores <= common_upper)
        )
        treated_in_range = (
            (treated_scores >= common_lower)
            & (treated_scores <= common_upper)
        )
        control_common_rate = float(control_in_range.mean())
        treated_common_rate = float(treated_in_range.mean())
    else:
        control_common_rate = 0.0
        treated_common_rate = 0.0

    report = {
        "model": "HistGradientBoostingClassifier",
        "purpose": "Propensity score and overlap diagnostics",
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_propensity_auc": auc,
        "common_score_range": (
            [float(common_lower), float(common_upper)]
            if has_common_range
            else None
        ),
        "control_fraction_in_common_range": control_common_rate,
        "treatment_fraction_in_common_range": treated_common_rate,
        "score_distributions": overlap_rows,
        "interpretation_note": (
            "This is a descriptive overlap diagnostic. "
            "A common score range alone does not establish adequate "
            "overlap or prove causal identification."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / "propensity_overlap_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("ECONOCAUSAL — PROPENSITY & OVERLAP ANALYSIS")
    print("-" * 48)
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test propensity model ROC-AUC: {auc:.4f}")
    print()
    print("PROPENSITY SCORE DISTRIBUTIONS")
    print(pd.DataFrame(overlap_rows).to_string(index=False))
    print()
    print(f"Common score range: {report['common_score_range']}")
    print(
        "Control fraction in common range: "
        f"{control_common_rate:.2%}"
    )
    print(
        "Treatment fraction in common range: "
        f"{treated_common_rate:.2%}"
    )
    print(f"\nReport saved: {report_path}")
    print()
    print(
        "Note: These diagnostics do not, by themselves, "
        "establish causal validity."
    )


if __name__ == "__main__":
    main()
