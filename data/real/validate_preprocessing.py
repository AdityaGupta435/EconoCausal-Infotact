
from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "criteo_sample.csv"
OUTPUT_DIR = BASE_DIR / "processed" / "eda_output"

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
REQUIRED_COLUMNS = FEATURES + BINARY_COLUMNS


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_columns = FEATURES + BINARY_COLUMNS
    non_numeric = [
        column for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    non_finite_counts = {}
    for column in numeric_columns:
        values = df[column].to_numpy(dtype=float)
        non_finite_counts[column] = int((~np.isfinite(values)).sum())

    invalid_binary_values = {}
    for column in BINARY_COLUMNS:
        values = df[column].dropna().unique()
        invalid_binary_values[column] = sorted(
            float(value) for value in values if value not in (0, 1)
        )

    treatment_counts = df["treatment"].value_counts(dropna=False)
    outcome_rates = (
        df.groupby("treatment")[["conversion", "visit", "exposure"]]
        .mean()
    )

    report = {
        "dataset_path": str(DATA_PATH),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "required_columns_present": True,
        "missing_required_columns": missing_columns,
        "non_numeric_columns": non_numeric,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "non_finite_values_by_column": non_finite_counts,
        "invalid_binary_values": invalid_binary_values,
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "treatment_counts": {
            str(key): int(value)
            for key, value in treatment_counts.items()
        },
        "outcome_rates_by_treatment": {
            str(group): {
                column: float(value)
                for column, value in rates.items()
            }
            for group, rates in outcome_rates.iterrows()
        },
    }

    # These checks report problems; they do not silently clean the data.
    report["checks_passed"] = (
        len(df) > 0
        and not missing_columns
        and not non_numeric
        and report["missing_cells"] == 0
        and all(count == 0 for count in non_finite_counts.values())
        and all(not values for values in invalid_binary_values.values())
        and set(df["treatment"].dropna().unique()).issubset({0, 1})
    )

    report_path = OUTPUT_DIR / "preprocessing_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Preprocessing validation completed.")
    print(f"Rows: {report['rows']:,}")
    print(f"Missing cells: {report['missing_cells']:,}")
    print(f"Exact duplicate rows: {report['exact_duplicate_rows']:,}")
    print(f"Non-finite values: {sum(non_finite_counts.values()):,}")
    print(f"Invalid binary values: {sum(map(len, invalid_binary_values.values()))}")
    print(f"Checks passed: {report['checks_passed']}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
