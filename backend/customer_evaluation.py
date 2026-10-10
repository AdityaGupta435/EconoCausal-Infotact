
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from causal_model import estimate_ate


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
SPLIT_DIR = DATA_DIR / "splits"
RAW_PATH = DATA_DIR / "retail_data.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    # Split by customer so one customer cannot appear in both sets.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["customer_id"])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    estimate = estimate_ate(train_df)

    # Evaluation-only ground truth: never passed to the model.
    true_ate = float(test_df["true_effect"].mean())
    error = abs(estimate - true_ate)

    print("CUSTOMER-GROUPED DML EVALUATION")
    print("-" * 40)
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Train customers: {train_df['customer_id'].nunique():,}")
    print(f"Test customers: {test_df['customer_id'].nunique():,}")
    print(
        "Customer overlap:",
        len(
            set(train_df["customer_id"])
            & set(test_df["customer_id"])
        ),
    )
    print(f"Estimated ATE: {estimate:.4f}")
    print(f"Test ground-truth ATE: {true_ate:.4f}")
    print(f"Absolute ATE error: {error:.4f}")
    print(f"Absolute error (percentage points): {error * 100:.2f}")


if __name__ == "__main__":
    main()
