import numpy as np
import pytest

from backend.uplift_analysis import qini_area, qini_curve


def test_qini_curve_returns_finite_values():
    outcomes = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    coverage, gains = qini_curve(outcomes, treatment, scores)

    assert len(coverage) == len(gains)
    assert len(coverage) >= 2
    assert np.isfinite(coverage).all()
    assert np.isfinite(gains).all()


def test_qini_area_is_finite():
    outcomes = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    assert np.isfinite(qini_area(outcomes, treatment, scores))


def test_qini_rejects_data_without_both_groups():
    outcomes = np.array([1, 0, 1, 0])
    treatment = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    with pytest.raises(ValueError, match="Insufficient"):
        qini_curve(outcomes, treatment, scores)
