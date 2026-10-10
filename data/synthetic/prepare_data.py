
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "retail_data.csv"
OUTPUT_DIR = BASE_DIR / "splits"

RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep evaluation-only columns separate from model inputs.
    evaluation_columns = [
        "true_effect",
        "baseline_probability",
        "propensity_score",
    ]

    evaluation_data = df[evaluation_columns].copy()
    model_data = df.drop(columns=evaluation_columns).copy()

    train_df, temporary_df = train_test_split(
        model_data,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=model_data["treatment"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_df["treatment"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Store ground truth separately for later causal evaluation.
    evaluation_data.loc[train_df.index].to_csv(
        OUTPUT_DIR / "train_ground_truth.csv", index=False
    )
    evaluation_data.loc[validation_df.index].to_csv(
        OUTPUT_DIR / "validation_ground_truth.csv", index=False
    )
    evaluation_data.loc[test_df.index].to_csv(
        OUTPUT_DIR / "test_ground_truth.csv", index=False
    )

    print("Dataset split completed.")
    print(f"Train:      {len(train_df):,} rows")
    print(f"Validation: {len(validation_df):,} rows")
    print(f"Test:       {len(test_df):,} rows")
    print("Ground-truth columns saved separately.")


if __name__ == "__main__":
    main()
