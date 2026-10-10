
from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "retail_data.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "customer_id",
        "product_id",
        "baseline_price",
        "treatment",
        "discount_pct",
        "purchase",
        "true_effect",
        "propensity_score",
    ]

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    assert len(df) == 10_000, "Unexpected row count"
    assert df[required_columns].notna().all().all(), "Missing values found"
    assert df["treatment"].isin([0, 1]).all()
    assert df["purchase"].isin([0, 1]).all()
    assert df["discount_pct"].between(0, 100).all()
    assert (df["baseline_price"] >= 0).all()
    assert df["propensity_score"].between(0, 1).all()

    print("DATA QUALITY REPORT")
    print("-" * 35)
    print(f"Rows: {len(df):,}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing cells: {df.isna().sum().sum()}")

    print("\nTreatment balance:")
    print(df["treatment"].value_counts(normalize=True).sort_index())

    print("\nPurchase rate by treatment:")
    print(df.groupby("treatment")["purchase"].mean())

    print("\nKnown treatment effect:")
    print(f"{df['true_effect'].mean():.2%}")

    print("\nAll data quality assertions passed.")


if __name__ == "__main__":
    main()
