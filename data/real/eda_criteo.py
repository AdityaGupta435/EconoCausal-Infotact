
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "criteo_sample.csv"
OUTPUT_DIR = BASE_DIR / "processed" / "eda_output"

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]


def save_bar_chart(series, title, ylabel, filename):
    ax = series.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sample not found: {DATA_PATH}\n"
            "Run data/real/sample_criteo.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + BINARY_COLUMNS
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset sample is empty.")

    # 1. Dataset overview and data quality
    quality = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },
        "dtypes": {
            key: str(value) for key, value in df.dtypes.items()
        },
    }

    for column in BINARY_COLUMNS:
        observed = set(df[column].dropna().unique())
        quality[f"{column}_invalid_values"] = sorted(
            int(value) if isinstance(value, (int, np.integer)) else str(value)
            for value in observed - {0, 1}
        )

    with (OUTPUT_DIR / "data_quality.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(quality, file, indent=2)

    df.describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    # 2. Treatment group size
    treatment_counts = (
        df["treatment"]
        .value_counts(dropna=False)
        .sort_index()
        .rename(index={0: "Control", 1: "Treatment"})
    )
    treatment_counts.to_csv(OUTPUT_DIR / "treatment_counts.csv")
    save_bar_chart(
        treatment_counts,
        "Treatment Group Counts",
        "Rows",
        "treatment_counts.png",
    )

    # 3. Outcomes by treatment assignment
    outcome_summary = (
        df.groupby("treatment")[["conversion", "visit", "exposure"]]
        .agg(["count", "mean", "sum"])
    )
    outcome_summary.to_csv(OUTPUT_DIR / "outcomes_by_treatment.csv")

    conversion_rates = (
        df.groupby("treatment")["conversion"].mean()
        .rename(index={0: "Control", 1: "Treatment"})
    )
    save_bar_chart(
        conversion_rates,
        "Observed Conversion Rate by Assignment",
        "Conversion probability",
        "conversion_by_treatment.png",
    )

    outcome_rates = (
        df[["conversion", "visit", "exposure"]].mean()
        .rename("rate")
        .to_frame()
    )
    outcome_rates.to_csv(OUTPUT_DIR / "overall_outcome_rates.csv")

    # 4. Feature distributions
    df[FEATURES].hist(
        bins=35,
        figsize=(15, 12),
        layout=(4, 3),
        grid=False,
    )
    plt.suptitle("Criteo Feature Distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=160)
    plt.close()

    # 5. Feature summary and outlier indicators
    feature_summary = df[FEATURES].describe().T
    feature_summary["missing"] = df[FEATURES].isna().sum()
    feature_summary["unique_values"] = df[FEATURES].nunique()

    q1 = df[FEATURES].quantile(0.25)
    q3 = df[FEATURES].quantile(0.75)
    iqr = q3 - q1

    outlier_counts = (
        (df[FEATURES] < (q1 - 1.5 * iqr))
        | (df[FEATURES] > (q3 + 1.5 * iqr))
    ).sum()

    feature_summary["iqr_outlier_count"] = outlier_counts
    feature_summary.to_csv(OUTPUT_DIR / "feature_summary.csv")

    # 6. Feature correlation heatmap
    corr = df[FEATURES].corr()
    corr.to_csv(OUTPUT_DIR / "feature_correlation.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=160)
    plt.close()

    # 7. Feature means by treatment assignment
    means = df.groupby("treatment")[FEATURES].mean().T
    means.columns = ["control_mean", "treatment_mean"]
    means["difference"] = (
        means["treatment_mean"] - means["control_mean"]
    )
    means.to_csv(OUTPUT_DIR / "feature_means_by_treatment.csv")

    # 8. Markdown report
    report = [
        "# Criteo Sample — Exploratory Data Analysis",
        "",
        "## Dataset overview",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
        f"- Duplicate rows: {df.duplicated().sum():,}",
        f"- Missing cells: {df.isna().sum().sum():,}",
        "",
        "## Treatment and outcome rates",
        "",
        outcome_rates.to_string(),
        "",
        "## Outcome summary by treatment assignment",
        "",
        outcome_summary.to_string(),
        "",
        "## Data-quality notes",
        "- Treatment-group comparisons are descriptive, not proof of causality.",
        "- Conversion is a rare outcome; uncertainty must be considered.",
        "- IQR outlier counts are diagnostic flags, not automatic deletion rules.",
        "- Anonymized features cannot be interpreted as named business variables.",
        "- Exposure is reported separately and is not included in the feature matrix.",
        "",
        "## Generated outputs",
    ]

    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != "eda_report.md":
            report.append(f"- `{path.name}`")

    (OUTPUT_DIR / "eda_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("Criteo EDA completed.")
    print(f"Rows: {len(df):,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Missing cells: {df.isna().sum().sum():,}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nConversion rate by treatment:")
    print(conversion_rates)


if __name__ == "__main__":
    main()
