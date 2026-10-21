import numpy as np
import pandas as pd
import pytest

from backend.optimization import build_optimization_problem


def test_builds_problem_with_fixed_ten_percent_discount():
    candidates = pd.DataFrame({
        "baseline_price": [100.0, 250.0],
        "predicted_ite": [0.08, 0.12],
    })

    problem = build_optimization_problem(candidates, budget=20.0)

    np.testing.assert_allclose(
        problem.expected_incremental_purchases, [0.08, 0.12]
    )
    np.testing.assert_allclose(problem.promotion_costs, [10.0, 25.0])
    assert problem.budget == 20.0
    assert problem.discount_pct == 10.0
    assert problem.decision_variable == "x_i in {0, 1}"


def test_accepts_custom_fixed_discount():
    candidates = pd.DataFrame({
        "baseline_price": [200.0],
        "predicted_ite": [0.05],
    })

    problem = build_optimization_problem(
        candidates, budget=50.0, discount_pct=15.0
    )

    np.testing.assert_allclose(problem.promotion_costs, [30.0])


def test_rejects_missing_columns():
    candidates = pd.DataFrame({"baseline_price": [100.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        build_optimization_problem(candidates, budget=20.0)


def test_rejects_negative_budget():
    candidates = pd.DataFrame({
        "baseline_price": [100.0],
        "predicted_ite": [0.05],
    })

    with pytest.raises(ValueError, match="budget"):
        build_optimization_problem(candidates, budget=-1.0)


def test_rejects_non_finite_effects():
    candidates = pd.DataFrame({
        "baseline_price": [100.0],
        "predicted_ite": [np.nan],
    })

    with pytest.raises(ValueError, match="predicted_ite"):
        build_optimization_problem(candidates, budget=20.0)


def test_rejects_empty_candidates():
    candidates = pd.DataFrame({
        "baseline_price": pd.Series(dtype=float),
        "predicted_ite": pd.Series(dtype=float),
    })

    with pytest.raises(ValueError, match="at least one row"):
        build_optimization_problem(candidates, budget=20.0)
