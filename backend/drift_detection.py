
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


NUMERIC_FEATURES = [
    "baseline_price",
    "customer_history",
    "seasonality",
]

CATEGORICAL_FEATURES = [
    "product_id",
    "category",
]

DRIFT_THRESHOLD = 0.05



def detect_numeric_drift(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Compare numerical feature distributions using the KS test."""

    reference_raw = pd.to_numeric(reference, errors="coerce")
    current_raw = pd.to_numeric(current, errors="coerce")

    reference_missing = int(reference_raw.isna().sum())
    current_missing = int(current_raw.isna().sum())

    reference_clean = reference_raw.dropna()
    current_clean = current_raw.dropna()

    if reference_clean.empty or current_clean.empty:
        raise ValueError("Numerical feature contains no valid observations.")

    result = ks_2samp(reference_clean, current_clean)

    return {
        "method": "Kolmogorov-Smirnov test",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drift_detected": bool(result.pvalue < DRIFT_THRESHOLD),
        "reference_mean": float(reference_clean.mean()),
        "current_mean": float(current_clean.mean()),
        "reference_count": int(reference_clean.size),
        "current_count": int(current_clean.size),
        "reference_missing": reference_missing,
        "current_missing": current_missing,
    }


def detect_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Compare categorical distributions using an appropriate statistical test."""

    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)

    if reference.empty or current.empty:
        raise ValueError("Categorical feature contains no valid observations.")

    categories = sorted(set(reference) | set(current))

    if len(categories) < 2:
        return {
            "method": "Insufficient categories",
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reference_categories": int(reference.nunique()),
            "current_categories": int(current.nunique()),
        }

    reference_codes = pd.Categorical(
        reference, categories=categories
    ).codes

    current_codes = pd.Categorical(
        current, categories=categories
    ).codes

    reference_counts = np.bincount(
        reference_codes, minlength=len(categories)
    )
    current_counts = np.bincount(
        current_codes, minlength=len(categories)
    )

    observed = np.array([
        reference_counts,
        current_counts,
    ])

    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    grand_total = observed.sum()

    expected = np.outer(row_totals, column_totals) / grand_total

    statistic = float(
        np.sum((observed - expected) ** 2 / expected)
    )

    if np.all(expected >= 5):
        _, p_value, _, _ = chi2_contingency(
            observed,
            correction=False,
        )
        method = "Chi-square test"

    else:
        # Permutation test for sparse categorical distributions.
        combined_codes = np.concatenate([
            reference_codes,
            current_codes,
        ])

        reference_size = len(reference_codes)
        rng = np.random.default_rng(42)
        permutations = 999
        extreme_count = 0

        for _ in range(permutations):
            shuffled = rng.permutation(combined_codes)

            perm_reference = np.bincount(
                shuffled[:reference_size],
                minlength=len(categories),
            )

            perm_current = np.bincount(
                shuffled[reference_size:],
                minlength=len(categories),
            )

            perm_observed = np.array([
                perm_reference,
                perm_current,
            ])

            perm_expected = np.outer(
                perm_observed.sum(axis=1),
                perm_observed.sum(axis=0),
            ) / grand_total

            perm_statistic = np.sum(
                (perm_observed - perm_expected) ** 2
                / perm_expected
            )

            if perm_statistic >= statistic:
                extreme_count += 1

        p_value = (extreme_count + 1) / (permutations + 1)
        method = "Permutation test (999 resamples)"

    return {
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < DRIFT_THRESHOLD),
        "reference_categories": int(reference.nunique()),
        "current_categories": int(current.nunique()),
    }



def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a feature-level distribution drift report."""

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing_reference = [
        feature for feature in features
        if feature not in reference.columns
    ]
    missing_current = [
        feature for feature in features
        if feature not in current.columns
    ]

    if missing_reference or missing_current:
        raise ValueError(
            f"Missing required features. "
            f"Reference: {missing_reference}; "
            f"Current: {missing_current}"
        )

    feature_results = {}

    for feature in NUMERIC_FEATURES:
        feature_results[feature] = detect_numeric_drift(
            reference[feature],
            current[feature],
        )

    for feature in CATEGORICAL_FEATURES:
        feature_results[feature] = detect_categorical_drift(
            reference[feature],
            current[feature],
        )

    drifted_features = [
        feature
        for feature, result in feature_results.items()
        if result["drift_detected"]
    ]

    drift_count = len(drifted_features)

    if drift_count == 0:
        status = "stable"
    else:
        status = "drift_detected"

    return {
        "status": status,
        "drift_detected": drift_count > 0,
        "total_features": len(features),
        "drifted_feature_count": drift_count,
        "drifted_features": drifted_features,
        "threshold": DRIFT_THRESHOLD,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": feature_results,
        "interpretation": (
            "Distribution changes were detected. Investigate the affected "
            "features and assess model performance before taking action."
            if drift_count > 0
            else "No statistically significant feature drift was detected "
                 "at the configured threshold."
        ),
        "limitation": (
            "Statistical drift is a monitoring signal, not proof of model "
            "degradation or invalid causal estimates."
        ),
    }
