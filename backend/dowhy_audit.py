
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_estimator import CausalEstimate
from sklearn.linear_model import LinearRegression


def _pandas3_compatible_estimate_effect(
    self, data_df=None, need_conditional_estimates=None
):
    """Compatibility fix for DoWhy 0.8 with pandas 3.x."""
    if data_df is None:
        data_df = self._data

    if need_conditional_estimates is None:
        need_conditional_estimates = self.need_conditional_estimates

    if not self.model:
        _, self.model = self._build_model()
        coefficients = self.model.params.iloc[1:]
        self.logger.debug(
            "Coefficients of the fitted model: "
            + ",".join(map(str, coefficients))
        )
        self.logger.debug(self.model.summary())

    effect_estimate = (
        self._do(self._treatment_value, data_df)
        - self._do(self._control_value, data_df)
    )

    conditional_effect_estimates = None
    if need_conditional_estimates:
        conditional_effect_estimates = self._estimate_conditional_effects(
            self._estimate_effect_fn,
            effect_modifier_names=self._effect_modifier_names,
        )

    return CausalEstimate(
        estimate=effect_estimate,
        control_value=self._control_value,
        treatment_value=self._treatment_value,
        conditional_estimates=conditional_effect_estimates,
        target_estimand=self._target_estimand,
        realized_estimand_expr=self.symbolic_estimator,
        intercept=self.model.params.iloc[0],
    )


# DoWhy 0.8 indexes pandas Series by integer label; pandas 3 requires
# positional access here. Patch the inherited estimator method at runtime.
RegressionEstimator._estimate_effect = _pandas3_compatible_estimate_effect


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "synthetic" / "splits"
OUTPUT_PATH = ROOT / "data" / "synthetic" / "dowhy_audit_report.json"

TREATMENT = "treatment"
OUTCOME = "purchase"
COVARIATES = [
    "baseline_price",
    "customer_history",
    "seasonality",
    "category",
]


def load_data():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    ground_truth = pd.read_csv(SPLIT_DIR / "train_ground_truth.csv")

    required = COVARIATES + [TREATMENT, OUTCOME]
    missing = sorted(set(required) - set(train.columns))

    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    if len(train) != len(ground_truth):
        raise ValueError("Training and ground-truth rows do not align.")

    if train[required].isnull().any().any():
        raise ValueError("Required training columns contain missing values.")

    if set(train[TREATMENT].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")

    if not set(train[OUTCOME].unique()).issubset({0, 1}):
        raise ValueError("Outcome must be binary.")

    if "true_effect" not in ground_truth.columns:
        raise ValueError("Ground truth is missing true_effect.")

    return train, ground_truth


def build_model(data):
    return CausalModel(
        data=data,
        treatment=TREATMENT,
        outcome=OUTCOME,
        common_causes=COVARIATES,
        proceed_when_unidentifiable=False,
    )


def _number_or_none(value):
    """Convert numeric scalars to JSON-safe Python floats."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _refutation_summary(result):
    """Extract stable, serializable fields from a DoWhy refutation result."""
    summary = {"text": str(result)}

    for attribute in (
        "estimated_effect",
        "new_effect",
        "refutation_type",
        "refutation_result",
    ):
        if hasattr(result, attribute):
            value = getattr(result, attribute)
            numeric = _number_or_none(value)
            if numeric is not None:
                summary[attribute] = numeric
            elif isinstance(value, (str, bool, int, float)) or value is None:
                summary[attribute] = value

    return summary


def run_audit():
    train, ground_truth = load_data()

    model_data = train[COVARIATES + [TREATMENT, OUTCOME]].copy()
    model = build_model(model_data)
    estimand = model.identify_effect()

    # Independent sklearn regression-adjustment estimate for comparison.
    X = pd.get_dummies(
        model_data[COVARIATES + [TREATMENT]],
        columns=["category"],
        drop_first=True,
        dtype=float,
    )
    y = model_data[OUTCOME].astype(float)

    regressor = LinearRegression()
    regressor.fit(X, y)

    treated_data = X.copy()
    control_data = X.copy()
    treated_data[TREATMENT] = 1
    control_data[TREATMENT] = 0

    sklearn_ate = float(
        np.mean(
            regressor.predict(treated_data)
            - regressor.predict(control_data)
        )
    )
    true_ate = float(ground_truth["true_effect"].mean())

    # Actual DoWhy estimate; required by DoWhy's refutation API.
    dowhy_estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    dowhy_ate = float(dowhy_estimate.value)

    refutations = {}
    refutation_warnings = {}

    tests = [
        ("random_common_cause", "random_common_cause"),
        ("placebo_treatment", "placebo_treatment_refuter"),
    ]

    for report_name, method_name in tests:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.refute_estimate(
                    estimand,
                    dowhy_estimate,
                    method_name=method_name,
                )
                refutations[report_name] = {
                    "status": "completed",
                    **_refutation_summary(result),
                }
            except Exception as exc:
                refutations[report_name] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

            if caught:
                refutation_warnings[report_name] = [
                    f"{item.category.__name__}: {item.message}"
                    for item in caught
                ]

    results = {
        "project": "EconoCausal",
        "audit": "DoWhy identification, regression adjustment, and refutation",
        "dataset": "Synthetic retail promotion benchmark",
        "rows": int(len(model_data)),
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "adjustment_covariates": COVARIATES,
        "identified_estimand": str(estimand),
        "estimates": {
            "dowhy_linear_regression_ate": dowhy_ate,
            "sklearn_regression_adjustment_ate": sklearn_ate,
            "known_synthetic_ate": true_ate,
            "dowhy_absolute_error_vs_truth": abs(dowhy_ate - true_ate),
            "sklearn_absolute_error_vs_truth": abs(sklearn_ate - true_ate),
        },
        "refutations": refutations,
        "refutation_warnings": refutation_warnings,
        "limitations": [
            "Synthetic benchmark results do not establish real-world effects.",
            "Known effects are used only for evaluation, not model fitting.",
            "Refutation tests assess robustness; they do not prove causality.",
            "Rows repeat customers, so row-level uncertainty may understate dependence.",
            "This binary treatment estimates promotion versus no promotion, "
            "not separate effects for each discount percentage.",
            "Random common cause and placebo refutations are sensitivity checks, "
            "not proof that confounding has been eliminated.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Rows audited: {results['rows']}")
    print(f"DoWhy ATE: {dowhy_ate:.6f}")
    print(f"sklearn ATE: {sklearn_ate:.6f}")
    print(f"Known synthetic ATE: {true_ate:.6f}")
    print(
        "DoWhy absolute ATE error: "
        f"{results['estimates']['dowhy_absolute_error_vs_truth']:.6f}"
    )

    for name, details in refutations.items():
        print(f"Refutation [{name}]: {details['status']}")
        if "error" in details:
            print(f"  Error: {details['error']}")
        elif "text" in details:
            print(f"  {details['text']}")

    for name, messages in refutation_warnings.items():
        for message in messages:
            print(f"Warning [{name}]: {message}")

    print(f"Audit JSON saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_audit()
