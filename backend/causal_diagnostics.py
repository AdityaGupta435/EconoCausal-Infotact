
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from causal_model import (
    FEATURES,
    make_outcome_model,
    make_treatment_model,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic" / "splits"


def run_diagnostics(data, n_splits=5, n_bootstrap=500):
    data = data.reset_index(drop=True)
    x = data[FEATURES]
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["purchase"].to_numpy(dtype=float)

    y_residual = np.zeros(len(data))
    t_residual = np.zeros(len(data))
    propensity_oof = np.zeros(len(data))

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

        propensity_oof[holdout_idx] = propensity
        propensity = np.clip(propensity, 0.01, 0.99)

        t_residual[holdout_idx] = (
            treatment[holdout_idx] - propensity
        )

    denominator = np.sum(t_residual**2)

    if denominator <= 1e-10:
        raise ValueError("Insufficient residual treatment variation.")

    contributions = t_residual * y_residual / denominator
    ate = float(contributions.sum())

    rng = np.random.default_rng(42)
    bootstrap_ates = np.empty(n_bootstrap)

    # Diagnostic row bootstrap; not a cluster-robust interval.
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(data), size=len(data))
        t_star = t_residual[idx]
        y_star = y_residual[idx]
        denom_star = np.sum(t_star**2)

        bootstrap_ates[i] = (
            np.sum(t_star * y_star) / denom_star
            if denom_star > 1e-10
            else np.nan
        )

    bootstrap_ates = bootstrap_ates[np.isfinite(bootstrap_ates)]
    lower, upper = np.quantile(bootstrap_ates, [0.025, 0.975])

    print("PROPENSITY OVERLAP")
    print("-" * 35)
    print(
        "Control propensity quantiles:",
        np.round(
            np.quantile(propensity_oof[treatment == 0], [0, 0.1, 0.5, 0.9, 1]),
            3,
        ),
    )
    print(
        "Treated propensity quantiles:",
        np.round(
            np.quantile(propensity_oof[treatment == 1], [0, 0.1, 0.5, 0.9, 1]),
            3,
        ),
    )

    overlap_fraction = np.mean(
        (propensity_oof >= 0.05) & (propensity_oof <= 0.95)
    )
    print(f"Rows with propensity in [0.05, 0.95]: {overlap_fraction:.1%}")

    print("\nDML ESTIMATE")
    print("-" * 35)
    print(f"Estimated ATE: {ate:.4f} ({ate * 100:.2f} pp)")
    print(
        "Bootstrap 95% interval: "
        f"[{lower:.4f}, {upper:.4f}] "
        f"([{lower * 100:.2f}, {upper * 100:.2f}] pp)"
    )

    return {
        "ate": ate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "overlap_fraction": float(overlap_fraction),
    }


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    results = run_diagnostics(train_df)

    print("\nDiagnostics complete.")
    print(
        "Note: This bootstrap interval is approximate. "
        "Repeated customer observations mean a future customer-clustered "
        "analysis is also needed."
    )


if __name__ == "__main__":
    main()
