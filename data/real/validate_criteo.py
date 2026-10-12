
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "real" / "processed" / "criteo_sample.csv"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head().to_string(index=False))

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().to_string())

    for column in ["treatment", "conversion", "visit", "exposure"]:
        if column in df.columns:
            print(f"\n--- {column.upper()} DISTRIBUTION ---")
            print(df[column].value_counts(dropna=False).to_string())

    if {"treatment", "conversion"}.issubset(df.columns):
        print("\n--- CONVERSION RATE BY TREATMENT ---")
        print(
            df.groupby("treatment")["conversion"]
            .agg(["count", "mean"])
            .to_string()
        )

    print("\nValidation inspection complete.")

if __name__ == "__main__":
    main()
