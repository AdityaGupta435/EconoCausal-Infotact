"""Regression and unit tests for the EconoCausal data pipeline.

Verifies:
1. Zero customer overlap across train/validation/test splits.
2. Ground-truth 1-to-1 row alignment.
3. Split proportions and balanced treatment representation.
4. Dataset profile validation, schema enforcement, and semantic constraints.
5. Distinct handling of Retail Pricing vs. Criteo Advertising Uplift profiles.
6. Chunked streaming execution for bounded-memory processing.
7. Reproducibility and backup utilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.data_pipeline import (
    DATASET_PROFILES,
    DatasetProfile,
    create_splits_backup,
    get_optimal_dtypes,
    load_dataset,
    process_large_dataset,
    save_splits,
    split_customer_data,
    stream_csv_chunks,
    validate_dataset,
    verify_splits,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "synthetic" / "splits"
RETAIL_DATA_PATH = BASE_DIR / "data" / "synthetic" / "retail_data.csv"


# =========================================================================
# 1. Customer Leakage Regression Tests
# =========================================================================

def test_zero_customer_overlap_in_persisted_splits():
    """Verify that persisted splits have zero customer ID overlap across all partitions."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    train_customers = set(train_df["customer_id"])
    val_customers = set(val_df["customer_id"])
    test_customers = set(test_df["customer_id"])

    # Explicit pairwise customer disjointness
    overlap_train_val = train_customers.intersection(val_customers)
    overlap_train_test = train_customers.intersection(test_customers)
    overlap_val_test = val_customers.intersection(test_customers)

    assert len(overlap_train_val) == 0, f"Customer overlap between train and validation: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Customer overlap between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Customer overlap between validation and test: {overlap_val_test}"


def test_ground_truth_row_alignment_in_persisted_splits():
    """Verify that each split matches its ground-truth file exactly in row count."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    train_gt = pd.read_csv(SPLITS_DIR / "train_ground_truth.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    val_gt = pd.read_csv(SPLITS_DIR / "validation_ground_truth.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_gt = pd.read_csv(SPLITS_DIR / "test_ground_truth.csv")

    assert len(train_df) == len(train_gt), f"Train mismatch: {len(train_df)} vs {len(train_gt)}"
    assert len(val_df) == len(val_gt), f"Val mismatch: {len(val_df)} vs {len(val_gt)}"
    assert len(test_df) == len(test_gt), f"Test mismatch: {len(test_df)} vs {len(test_gt)}"

    # Verify ground truth columns exist and have non-null values
    expected_gt_cols = {"true_effect", "baseline_probability", "propensity_score"}
    for name, gt_df in [("train", train_gt), ("val", val_gt), ("test", test_gt)]:
        assert expected_gt_cols.issubset(set(gt_df.columns)), f"{name} ground truth missing columns"
        assert gt_df["true_effect"].isnull().sum() == 0, f"{name} ground truth has nulls"


def test_persisted_split_sizes_and_treatment_balance():
    """Verify standard split sizes (6000/2000/2000) and balanced treatment rates."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    assert len(train_df) == 6000
    assert len(val_df) == 2000
    assert len(test_df) == 2000

    # Total row conservation
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == 10000

    # Treatment balance across splits
    train_tr = train_df["treatment"].mean()
    val_tr = val_df["treatment"].mean()
    test_tr = test_df["treatment"].mean()

    # Treatment should be roughly 35-45% in all splits
    assert 0.35 <= train_tr <= 0.45, f"Unusual train treatment rate: {train_tr:.4f}"
    assert 0.35 <= val_tr <= 0.45, f"Unusual val treatment rate: {val_tr:.4f}"
    assert 0.35 <= test_tr <= 0.45, f"Unusual test treatment rate: {test_tr:.4f}"


def test_split_customer_data_isolated():
    """Test split_customer_data logic with synthetic grouping and edge cases."""
    # Create synthetic data with 100 customers, 10 records per customer = 1000 records
    n_customers = 100
    records_per_cust = 10
    customer_ids = np.repeat(np.arange(n_customers), records_per_cust)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "feature_1": np.random.randn(len(customer_ids)),
        "treatment": np.random.binomial(1, 0.4, len(customer_ids)),
        "purchase": np.random.binomial(1, 0.2, len(customer_ids)),
        "true_effect": np.random.uniform(0.01, 0.1, len(customer_ids)),
        "baseline_probability": np.random.uniform(0.1, 0.3, len(customer_ids)),
        "propensity_score": np.random.uniform(0.3, 0.5, len(customer_ids)),
    })

    splits = split_customer_data(
        df,
        group_col="customer_id",
        train_size=0.60,
        val_size=0.20,
        test_size=0.20,
        random_state=123,
    )

    train_c = set(splits["train"]["customer_id"])
    val_c = set(splits["validation"]["customer_id"])
    test_c = set(splits["test"]["customer_id"])

    assert len(train_c & val_c) == 0
    assert len(train_c & test_c) == 0
    assert len(val_c & test_c) == 0

    assert len(splits["train"]) == 600
    assert len(splits["validation"]) == 200
    assert len(splits["test"]) == 200

    # Ground truth separated and aligned
    assert len(splits["train_ground_truth"]) == 600
    assert len(splits["validation_ground_truth"]) == 200
    assert len(splits["test_ground_truth"]) == 200
    assert "true_effect" not in splits["train"].columns
    assert "true_effect" in splits["train_ground_truth"].columns


def test_split_customer_data_invalid_inputs():
    """Test error handling for invalid split ratios or missing columns."""
    df = pd.DataFrame({"customer_id": [1, 2, 3], "value": [10, 20, 30]})

    # Ratios that do not sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_customer_data(df, train_size=0.5, val_size=0.2, test_size=0.2)

    # Missing group column
    with pytest.raises(KeyError, match="not found in DataFrame"):
        split_customer_data(df, group_col="non_existent_column")


# =========================================================================
# 2. Dataset Profiles and Metadata Tests
# =========================================================================

def test_dataset_profile_definitions():
    """Verify schema definitions, pricing compatibility, and usage guidance."""
    retail = DATASET_PROFILES[DatasetProfile.RETAIL_PRICING]
    criteo = DATASET_PROFILES[DatasetProfile.CRITEO_UPLIFT]

    assert retail.is_pricing_compatible is True
    assert criteo.is_pricing_compatible is False

    assert "customer_id" in retail.required_columns
    assert "baseline_price" in retail.required_columns
    assert "treatment" in retail.required_columns
    assert "purchase" in retail.required_columns

    assert "conversion" in criteo.required_columns
    assert "visit" in criteo.required_columns
    assert "f0" in criteo.required_columns
    assert "baseline_price" not in criteo.required_columns

    assert "Advertising uplift benchmark only" in criteo.usage_guidance


def test_optimal_dtypes_specification():
    """Verify memory downcasting specifications."""
    retail_dtypes = get_optimal_dtypes(DatasetProfile.RETAIL_PRICING)
    assert retail_dtypes["customer_id"] == "int32"
    assert retail_dtypes["baseline_price"] == "float32"
    assert retail_dtypes["treatment"] == "int8"
    assert retail_dtypes["purchase"] == "int8"
    assert retail_dtypes["category"] == "category"

    criteo_dtypes = get_optimal_dtypes(DatasetProfile.CRITEO_UPLIFT)
    assert criteo_dtypes["f0"] == "float32"
    assert criteo_dtypes["treatment"] == "int8"
    assert criteo_dtypes["conversion"] == "int8"


# =========================================================================
# 3. Data Validation and Quality Auditing Tests
# =========================================================================

def test_validate_retail_dataset_valid():
    """Test validation on valid retail data."""
    valid_df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "product_id": ["P1", "P2", "P3"],
        "category": ["A", "B", "C"],
        "baseline_price": [10.5, 20.0, 99.9],
        "customer_history": [1, 5, 10],
        "seasonality": [0, 1, 2],
        "treatment": [0, 1, 0],
        "purchase": [0, 1, 0],
    })

    result = validate_dataset(valid_df, profile=DatasetProfile.RETAIL_PRICING)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert result["is_pricing_compatible"] is True
    assert result["total_rows"] == 3


def test_validate_retail_dataset_missing_columns():
    """Test validation failure when required columns are absent."""
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "treatment": [0, 1],
    })

    result = validate_dataset(df, profile=DatasetProfile.RETAIL_PRICING)
    assert result["is_valid"] is False
    assert any("Missing required columns" in err for err in result["errors"])


def test_validate_retail_dataset_invalid_price_and_binary():
    """Test semantic bounds: non-positive price and non-binary treatment."""
    invalid_df = pd.DataFrame({
        "customer_id": [1, 2],
        "product_id": ["P1", "P2"],
        "category": ["A", "B"],
        "baseline_price": [-5.0, 0.0],  # Invalid prices
        "customer_history": [1, 2],
        "seasonality": [0, 1],
        "treatment": [3, 0],  # Invalid treatment (not in {0, 1})
        "purchase": [0, 1],
    })

    result = validate_dataset(invalid_df, profile=DatasetProfile.RETAIL_PRICING)
    assert result["is_valid"] is False
    assert any("non-positive baseline_price" in err for err in result["errors"])
    assert any("outside binary set" in err for err in result["errors"])


def test_validate_criteo_profile():
    """Test validation with Criteo advertising uplift profile."""
    # Valid Criteo row
    criteo_data = {f"f{i}": [0.1, 0.2] for i in range(12)}
    criteo_data.update({
        "treatment": [1, 0],
        "conversion": [0, 1],
        "visit": [1, 1],
        "exposure": [0, 0],
    })
    criteo_df = pd.DataFrame(criteo_data)

    result = validate_dataset(criteo_df, profile=DatasetProfile.CRITEO_UPLIFT)
    assert result["is_valid"] is True
    assert result["is_pricing_compatible"] is False

    # Invalid binary column
    criteo_df.loc[0, "conversion"] = 99
    invalid_result = validate_dataset(criteo_df, profile=DatasetProfile.CRITEO_UPLIFT)
    assert invalid_result["is_valid"] is False
    assert any("outside binary set" in err for err in invalid_result["errors"])


# =========================================================================
# 4. Large CSV Streaming and Bounded Memory Processing Tests
# =========================================================================

def test_stream_csv_chunks_bounded_memory(tmp_path):
    """Test that stream_csv_chunks streams records in expected chunk sizes."""
    csv_file = tmp_path / "stream_test.csv"
    
    # Generate 500 rows with 50-row chunks
    n_rows = 500
    df = pd.DataFrame({
        "customer_id": np.arange(n_rows, dtype=np.int32),
        "baseline_price": np.random.uniform(10, 100, n_rows).astype(np.float32),
        "treatment": np.random.binomial(1, 0.5, n_rows).astype(np.int8),
        "purchase": np.random.binomial(1, 0.2, n_rows).astype(np.int8),
    })
    df.to_csv(csv_file, index=False)

    chunk_count = 0
    total_streamed_rows = 0
    for chunk in stream_csv_chunks(csv_file, chunksize=100):
        chunk_count += 1
        total_streamed_rows += len(chunk)
        assert len(chunk) <= 100

    assert chunk_count == 5
    assert total_streamed_rows == 500


def test_process_large_dataset_filtering_and_clean_write(tmp_path):
    """Test process_large_dataset cleaning, invalid dropping, and chunked output."""
    in_file = tmp_path / "raw_data.csv"
    out_file = tmp_path / "cleaned_data.csv"

    # Create dataset with 200 valid rows and 50 invalid rows (negative baseline_price)
    valid_rows = pd.DataFrame({
        "customer_id": np.arange(200),
        "product_id": [f"P{i%10}" for i in range(200)],
        "category": ["Electronics"] * 200,
        "baseline_price": [50.0] * 200,
        "customer_history": [5] * 200,
        "seasonality": [1] * 200,
        "treatment": [1] * 200,
        "purchase": [0] * 200,
    })
    invalid_rows = pd.DataFrame({
        "customer_id": np.arange(200, 250),
        "product_id": [f"P{i%10}" for i in range(50)],
        "category": ["Electronics"] * 50,
        "baseline_price": [-10.0] * 50,  # Negative price -> should be dropped
        "customer_history": [5] * 50,
        "seasonality": [1] * 50,
        "treatment": [1] * 50,
        "purchase": [0] * 50,
    })
    full_df = pd.concat([valid_rows, invalid_rows], ignore_index=True)
    full_df.to_csv(in_file, index=False)

    summary = process_large_dataset(
        in_file,
        out_file,
        profile=DatasetProfile.RETAIL_PRICING,
        chunksize=75,
        drop_invalid_rows=True,
    )

    assert summary["total_input_rows"] == 250
    assert summary["total_output_rows"] == 200
    assert summary["dropped_rows"] == 50
    assert summary["chunks_processed"] == 4

    cleaned_df = pd.read_csv(out_file)
    assert len(cleaned_df) == 200
    assert (cleaned_df["baseline_price"] > 0).all()


# =========================================================================
# 5. Backup & Split Verification Utility Tests
# =========================================================================

def test_create_splits_backup(tmp_path):
    """Test that create_splits_backup safely copies all split files."""
    dummy_splits_dir = tmp_path / "splits"
    dummy_splits_dir.mkdir()

    for fname in ["train.csv", "train_ground_truth.csv", "validation.csv"]:
        (dummy_splits_dir / fname).write_text("dummy,data\n1,2", encoding="utf-8")

    backup_dir = create_splits_backup(dummy_splits_dir, backup_name="test_backup")
    assert backup_dir.is_dir()
    assert (backup_dir / "train.csv").is_file()
    assert (backup_dir / "train_ground_truth.csv").is_file()
    assert (backup_dir / "validation.csv").is_file()


def test_verify_splits_utility():
    """Test verify_splits utility on persisted splits directory."""
    verification = verify_splits(SPLITS_DIR)
    assert verification["is_leakage_free"] is True
    assert verification["alignment_valid"] is True
    assert verification["overlap_train_val"] == 0
    assert verification["overlap_train_test"] == 0
    assert verification["overlap_val_test"] == 0
    assert verification["train_rows"] == 6000
    assert verification["validation_rows"] == 2000
    assert verification["test_rows"] == 2000

