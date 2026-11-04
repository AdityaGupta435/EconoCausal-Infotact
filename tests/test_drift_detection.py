
import pandas as pd
import pytest

from backend.drift_detection import (
    detect_categorical_drift,
    detect_numeric_drift,
    generate_drift_report,
)


@pytest.fixture
def reference_data():
    return pd.DataFrame({
        "baseline_price": [100, 110, 120, 130, 140, 150],
        "customer_history": [1, 2, 3, 4, 5, 6],
        "seasonality": [0, 1, 2, 3, 0, 1],
        "product_id": ["P1", "P2", "P3", "P1", "P2", "P3"],
        "category": [
            "Fashion", "Electronics", "Grocery",
            "Fashion", "Electronics", "Grocery",
        ],
    })


def test_numeric_drift_detected():
    reference = pd.Series([10, 12, 14, 16, 18, 20])
    current = pd.Series([100, 120, 140, 160, 180, 200])

    result = detect_numeric_drift(reference, current)

    assert result["drift_detected"] is True
    assert result["reference_missing"] == 0
    assert result["current_missing"] == 0


def test_categorical_drift_detected():
    reference = pd.Series(["A", "A", "B", "B", "A", "B"])
    current = pd.Series(["C", "C", "C", "C", "C", "C"])

    result = detect_categorical_drift(reference, current)

    assert result["drift_detected"] is True


def test_stable_dataset_has_no_drift(reference_data):
    current = reference_data.copy()

    report = generate_drift_report(reference_data, current)

    assert report["status"] == "stable"
    assert report["drift_detected"] is False
    assert report["drifted_feature_count"] == 0
    assert report["total_features"] == 5



def test_controlled_drift_is_detected(reference_data):
    current = reference_data.copy()

    current["baseline_price"] *= 5

    # Use a larger sample to reliably test categorical drift.
    reference_data = pd.concat(
        [reference_data] * 10,
        ignore_index=True,
    )

    current = pd.concat(
        [current] * 10,
        ignore_index=True,
    )

    reference_data["category"] = (
        ["Fashion", "Electronics", "Grocery"] * 20
    )

    current["category"] = "Electronics"

    report = generate_drift_report(reference_data, current)

    assert report["status"] == "drift_detected"
    assert "baseline_price" in report["drifted_features"]
    assert "category" in report["drifted_features"]



def test_missing_values_are_reported():
    reference = pd.Series([10, 20, None, 40])
    current = pd.Series([10, None, 30, 40])

    result = detect_numeric_drift(reference, current)

    assert result["reference_missing"] == 1
    assert result["current_missing"] == 1
    assert result["reference_count"] == 3
    assert result["current_count"] == 3


def test_missing_required_feature_raises_error(reference_data):
    current = reference_data.drop(columns=["category"])

    with pytest.raises(ValueError, match="Missing required features"):
        generate_drift_report(reference_data, current)


def test_empty_numeric_feature_raises_error():
    reference = pd.Series([None, None])
    current = pd.Series([10, 20])

    with pytest.raises(ValueError, match="no valid observations"):
        detect_numeric_drift(reference, current)
