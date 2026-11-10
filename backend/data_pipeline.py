"""Data engineering pipeline for EconoCausal.

Provides scalable large-dataset ingestion, memory-conscious processing,
dataset profiling, data quality auditing, and customer-disjoint splitting.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class DatasetProfile(str, Enum):
    """Supported dataset profile configurations."""
    RETAIL_PRICING = "retail_pricing"
    CRITEO_UPLIFT = "criteo_uplift"


# Schema specifications
RETAIL_REQUIRED_COLUMNS = [
    "customer_id",
    "product_id",
    "category",
    "baseline_price",
    "customer_history",
    "seasonality",
    "treatment",
    "purchase",
]

RETAIL_OPTIONAL_COLUMNS = [
    "discount_pct",
    "revenue",
    "promotion_cost",
]

RETAIL_GROUND_TRUTH_COLUMNS = [
    "true_effect",
    "baseline_probability",
    "propensity_score",
]

CRITEO_REQUIRED_COLUMNS = [f"f{i}" for i in range(12)] + [
    "treatment",
    "conversion",
    "visit",
    "exposure",
]


@dataclass(frozen=True)
class ProfileMetadata:
    """Metadata describing a dataset profile and its operational scope."""
    profile: DatasetProfile
    description: str
    required_columns: list[str]
    optional_columns: list[str] = field(default_factory=list)
    ground_truth_columns: list[str] = field(default_factory=list)
    is_pricing_compatible: bool = True
    usage_guidance: str = ""


DATASET_PROFILES: dict[DatasetProfile, ProfileMetadata] = {
    DatasetProfile.RETAIL_PRICING: ProfileMetadata(
        profile=DatasetProfile.RETAIL_PRICING,
        description="Retail transaction & promotional discount benchmark.",
        required_columns=RETAIL_REQUIRED_COLUMNS,
        optional_columns=RETAIL_OPTIONAL_COLUMNS,
        ground_truth_columns=RETAIL_GROUND_TRUTH_COLUMNS,
        is_pricing_compatible=True,
        usage_guidance="Suitable for dynamic pricing, price elasticity, and discount optimization.",
    ),
    DatasetProfile.CRITEO_UPLIFT: ProfileMetadata(
        profile=DatasetProfile.CRITEO_UPLIFT,
        description="Criteo AI Lab online display advertising uplift benchmark.",
        required_columns=CRITEO_REQUIRED_COLUMNS,
        optional_columns=[],
        ground_truth_columns=[],
        is_pricing_compatible=False,
        usage_guidance=(
            "Advertising uplift benchmark only. Does not contain pricing, "
            "discounts, product categories, or margins. Must not be used for "
            "price elasticity, revenue, or profit estimation."
        ),
    ),
}


def get_optimal_dtypes(profile: DatasetProfile) -> dict[str, str]:
    """Return memory-efficient pandas dtype specifications for a profile."""
    if profile == DatasetProfile.RETAIL_PRICING:
        return {
            "customer_id": "int32",
            "product_id": "category",
            "category": "category",
            "baseline_price": "float32",
            "customer_history": "int16",
            "seasonality": "int8",
            "treatment": "int8",
            "discount_pct": "int8",
            "purchase": "int8",
            "revenue": "float32",
            "promotion_cost": "float32",
            "true_effect": "float32",
            "baseline_probability": "float32",
            "propensity_score": "float32",
        }
    elif profile == DatasetProfile.CRITEO_UPLIFT:
        dtypes: dict[str, str] = {f"f{i}": "float32" for i in range(12)}
        dtypes.update({
            "treatment": "int8",
            "conversion": "int8",
            "visit": "int8",
            "exposure": "int8",
        })
        return dtypes
    return {}


def stream_csv_chunks(
    filepath: str | Path,
    chunksize: int = 100_000,
    profile: DatasetProfile | None = None,
    use_optimal_dtypes: bool = True,
) -> Generator[pd.DataFrame, None, None]:
    """Stream large CSV files in chunks with bounded memory consumption.
    
    Designed to process datasets with 1M+ rows efficiently.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dtype_map = get_optimal_dtypes(profile) if (profile and use_optimal_dtypes) else None

    # Filter dtype map to columns present in file header
    header_cols = pd.read_csv(path, nrows=0).columns
    filtered_dtypes = (
        {k: v for k, v in dtype_map.items() if k in header_cols}
        if dtype_map else None
    )

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=filtered_dtypes):
        yield chunk


def load_dataset(
    filepath: str | Path,
    profile: DatasetProfile = DatasetProfile.RETAIL_PRICING,
    memory_optimized: bool = True,
) -> pd.DataFrame:
    """Load a dataset with optional memory optimization."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dtypes = get_optimal_dtypes(profile) if memory_optimized else None
    header_cols = pd.read_csv(path, nrows=0).columns
    filtered_dtypes = {k: v for k, v in dtypes.items() if k in header_cols} if dtypes else None

    return pd.read_csv(path, dtype=filtered_dtypes)


def validate_dataset(
    data: pd.DataFrame | str | Path,
    profile: DatasetProfile = DatasetProfile.RETAIL_PRICING,
) -> dict[str, Any]:
    """Audit data quality, missingness, duplicates, and profile conformance."""
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data, nrows=100_000)
    else:
        df = data

    meta = DATASET_PROFILES[profile]
    errors: list[str] = []
    warnings: list[str] = []

    # Check required columns
    missing_required = sorted(set(meta.required_columns) - set(df.columns))
    if missing_required:
        errors.append(f"Missing required columns for profile {profile.value}: {missing_required}")

    # Check missing values
    missing_counts = {str(col): int(val) for col, val in df.isnull().sum().items() if val > 0}
    total_missing_cells = sum(missing_counts.values())
    if total_missing_cells > 0:
        warnings.append(f"Dataset contains {total_missing_cells} missing values across {len(missing_counts)} columns.")

    # Check exact duplicates
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} exact duplicate rows.")

    # Semantic domain checks
    if profile == DatasetProfile.RETAIL_PRICING:
        if "baseline_price" in df.columns:
            invalid_prices = int((df["baseline_price"] <= 0).sum())
            if invalid_prices > 0:
                errors.append(f"Found {invalid_prices} records with non-positive baseline_price.")

        for binary_col in ["treatment", "purchase"]:
            if binary_col in df.columns:
                valid_binary = set(df[binary_col].dropna().unique()).issubset({0, 1})
                if not valid_binary:
                    errors.append(f"Column '{binary_col}' contains values outside binary set {{0, 1}}.")

    elif profile == DatasetProfile.CRITEO_UPLIFT:
        for binary_col in ["treatment", "conversion", "visit", "exposure"]:
            if binary_col in df.columns:
                valid_binary = set(df[binary_col].dropna().unique()).issubset({0, 1})
                if not valid_binary:
                    errors.append(f"Column '{binary_col}' contains values outside binary set {{0, 1}}.")

    # Calculate numeric summary stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats: dict[str, dict[str, float]] = {}
    for col in numeric_cols[:10]:
        series = df[col].dropna()
        if not series.empty:
            stats[str(col)] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    return {
        "profile": profile.value,
        "is_valid": len(errors) == 0,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_counts": missing_counts,
        "total_missing_cells": total_missing_cells,
        "duplicate_rows": duplicate_rows,
        "errors": errors,
        "warnings": warnings,
        "numeric_stats": stats,
        "is_pricing_compatible": meta.is_pricing_compatible,
        "usage_guidance": meta.usage_guidance,
    }


def process_large_dataset(
    input_path: str | Path,
    output_path: str | Path,
    profile: DatasetProfile = DatasetProfile.RETAIL_PRICING,
    chunksize: int = 100_000,
    drop_invalid_rows: bool = True,
) -> dict[str, Any]:
    """Stream, clean, validate, and write large CSV datasets in chunks.
    
    Operates in bounded memory O(chunksize) to process arbitrary volume.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_valid_rows = 0
    chunks_processed = 0

    for i, chunk in enumerate(stream_csv_chunks(in_path, chunksize=chunksize, profile=profile)):
        chunks_processed += 1
        initial_chunk_rows = len(chunk)
        total_rows += initial_chunk_rows

        if drop_invalid_rows:
            # Drop nulls in required columns
            required = [c for c in DATASET_PROFILES[profile].required_columns if c in chunk.columns]
            chunk = chunk.dropna(subset=required)

            if profile == DatasetProfile.RETAIL_PRICING and "baseline_price" in chunk.columns:
                chunk = chunk[chunk["baseline_price"] > 0]

        valid_chunk_rows = len(chunk)
        total_valid_rows += valid_chunk_rows

        mode = "w" if i == 0 else "a"
        header = (i == 0)
        chunk.to_csv(out_path, mode=mode, header=header, index=False)

    return {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "chunks_processed": chunks_processed,
        "total_input_rows": total_rows,
        "total_output_rows": total_valid_rows,
        "dropped_rows": total_rows - total_valid_rows,
    }


def split_customer_data(
    df: pd.DataFrame,
    group_col: str = "customer_id",
    train_size: float = 0.60,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split dataset with guaranteed customer-group mutual exclusivity.
    
    Ensures zero customer ID leakage across train, validation, and test splits.
    Separates ground-truth evaluation columns while maintaining 1-to-1 row alignment.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0.")

    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame.")

    # Separate ground-truth columns if present
    gt_cols = [c for c in RETAIL_GROUND_TRUTH_COLUMNS if c in df.columns]
    has_ground_truth = len(gt_cols) > 0

    model_cols = [c for c in df.columns if c not in gt_cols]
    model_df = df[model_cols].copy()
    gt_df = df[gt_cols].copy() if has_ground_truth else None

    # Step 1: Split into Train and Temp (Val + Test)
    temp_size = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(model_df, groups=model_df[group_col]))

    train_split = model_df.iloc[train_idx].copy().reset_index(drop=True)
    temp_split = model_df.iloc[temp_idx].copy()

    # Step 2: Split Temp into Validation and Test
    test_rel_size = test_size / temp_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_rel_size, random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_split, groups=temp_split[group_col]))

    val_split = temp_split.iloc[val_rel_idx].copy().reset_index(drop=True)
    test_split = temp_split.iloc[test_rel_idx].copy().reset_index(drop=True)

    # Verify zero customer overlap
    train_cust = set(train_split[group_col])
    val_cust = set(val_split[group_col])
    test_cust = set(test_split[group_col])

    if train_cust & val_cust:
        raise RuntimeError("Customer overlap detected between train and validation splits!")
    if train_cust & test_cust:
        raise RuntimeError("Customer overlap detected between train and test splits!")
    if val_cust & test_cust:
        raise RuntimeError("Customer overlap detected between validation and test splits!")

    result = {
        "train": train_split,
        "validation": val_split,
        "test": test_split,
    }

    # Slice matching ground-truth rows with identical row ordering
    if has_ground_truth and gt_df is not None:
        result["train_ground_truth"] = gt_df.iloc[train_idx].copy().reset_index(drop=True)
        result["validation_ground_truth"] = gt_df.iloc[temp_idx].iloc[val_rel_idx].copy().reset_index(drop=True)
        result["test_ground_truth"] = gt_df.iloc[temp_idx].iloc[test_rel_idx].copy().reset_index(drop=True)

        # Assert exact length alignment
        assert len(result["train"]) == len(result["train_ground_truth"])
        assert len(result["validation"]) == len(result["validation_ground_truth"])
        assert len(result["test"]) == len(result["test_ground_truth"])

    return result


def create_splits_backup(splits_dir: str | Path, backup_name: str | None = None) -> Path:
    """Create a timestamped backup of all existing split files."""
    source_dir = Path(splits_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Splits directory not found: {source_dir}")

    if backup_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"

    backup_dir = source_dir / backup_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_backed_up = 0
    for filename in [
        "train.csv",
        "train_ground_truth.csv",
        "validation.csv",
        "validation_ground_truth.csv",
        "test.csv",
        "test_ground_truth.csv",
    ]:
        src_file = source_dir / filename
        if src_file.is_file():
            shutil.copy2(src_file, backup_dir / filename)
            files_backed_up += 1

    return backup_dir


def save_splits(split_data: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist generated splits and ground-truth files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_mapping = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
        "train_ground_truth": "train_ground_truth.csv",
        "validation_ground_truth": "validation_ground_truth.csv",
        "test_ground_truth": "test_ground_truth.csv",
    }

    for key, filename in name_mapping.items():
        if key in split_data:
            split_data[key].to_csv(out_dir / filename, index=False)


def verify_splits(splits_dir: str | Path, group_col: str = "customer_id") -> dict[str, Any]:
    """Verify customer group mutual exclusivity and ground-truth row alignment."""
    s_dir = Path(splits_dir)
    train_df = pd.read_csv(s_dir / "train.csv")
    val_df = pd.read_csv(s_dir / "validation.csv")
    test_df = pd.read_csv(s_dir / "test.csv")

    train_gt = pd.read_csv(s_dir / "train_ground_truth.csv")
    val_gt = pd.read_csv(s_dir / "validation_ground_truth.csv")
    test_gt = pd.read_csv(s_dir / "test_ground_truth.csv")

    train_cust = set(train_df[group_col])
    val_cust = set(val_df[group_col])
    test_cust = set(test_df[group_col])

    overlap_train_val = len(train_cust & val_cust)
    overlap_train_test = len(train_cust & test_cust)
    overlap_val_test = len(val_cust & test_cust)

    is_leakage_free = (
        overlap_train_val == 0 and
        overlap_train_test == 0 and
        overlap_val_test == 0
    )

    alignment_valid = (
        len(train_df) == len(train_gt) and
        len(val_df) == len(val_gt) and
        len(test_df) == len(test_gt)
    )

    return {
        "is_leakage_free": is_leakage_free,
        "alignment_valid": alignment_valid,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_customers": len(train_cust),
        "validation_customers": len(val_cust),
        "test_customers": len(test_cust),
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
        "train_treatment_rate": float(train_df["treatment"].mean()) if "treatment" in train_df else None,
        "val_treatment_rate": float(val_df["treatment"].mean()) if "treatment" in val_df else None,
        "test_treatment_rate": float(test_df["treatment"].mean()) if "treatment" in test_df else None,
    }

