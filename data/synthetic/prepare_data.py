from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from backend.data_pipeline import (
    create_splits_backup,
    split_customer_data,
    save_splits,
    verify_splits,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "retail_data.csv"
OUTPUT_DIR = BASE_DIR / "splits"
RANDOM_STATE = 42


def main():
    print(f"Reading input data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # 1. Back up existing splits if they exist
    if OUTPUT_DIR.is_dir() and any(OUTPUT_DIR.glob("*.csv")):
        backup_dir = create_splits_backup(OUTPUT_DIR)
        print(f"Existing splits backed up to: {backup_dir}")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Perform customer-grouped split with zero customer leakage
    print("Performing customer-grouped split (train=0.60, val=0.20, test=0.20)...")
    splits = split_customer_data(
        df,
        group_col="customer_id",
        train_size=0.60,
        val_size=0.20,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # 3. Save new splits and aligned ground truth
    save_splits(splits, OUTPUT_DIR)
    print(f"New splits saved to: {OUTPUT_DIR}")

    # 4. Verify customer leakage and ground-truth alignment
    verification = verify_splits(OUTPUT_DIR, group_col="customer_id")
    print("\n--- Split Verification Report ---")
    print(f"Customer leakage-free: {verification['is_leakage_free']}")
    print(f"Ground-truth aligned:  {verification['alignment_valid']}")
    print(f"Train rows:            {verification['train_rows']:,} (customers: {verification['train_customers']:,}, treatment: {verification['train_treatment_rate']:.2%})")
    print(f"Validation rows:       {verification['validation_rows']:,} (customers: {verification['validation_customers']:,}, treatment: {verification['val_treatment_rate']:.2%})")
    print(f"Test rows:             {verification['test_rows']:,} (customers: {verification['test_customers']:,}, treatment: {verification['test_treatment_rate']:.2%})")
    print(f"Customer overlap T-V:  {verification['overlap_train_val']}")
    print(f"Customer overlap T-Te: {verification['overlap_train_test']}")
    print(f"Customer overlap V-Te: {verification['overlap_val_test']}")

    assert verification["is_leakage_free"], "Customer overlap detected!"
    assert verification["alignment_valid"], "Ground truth alignment mismatch!"
    print("\nCustomer-grouped data preparation completed successfully.")


if __name__ == "__main__":
    main()
