import numpy as np
import pandas as pd
import pytest

from backend.optimization import (
    build_optimization_problem,
    solve_optimization_problem,
)


def test_solver_selects_best_combination_within_budget():
    candidates = pd.DataFrame({
        "baseline_price": [100.0, 200.0, 50.0],
        "predicted_ite": [0.10, 0.18, 0.04],
    })
    problem = build_optimization_problem(candidates, budget=25.0)

    result = solve_optimization_problem(problem)

    # Costs are [10, 20, 5]. Rows 1 and 2 give the highest effect within budget.
    assert result.selected_indices == (1, 2)
    assert result.selected_count == 2
    assert result.total_cost == pytest.approx(25.0)
    assert result.expected_incremental_purchases == pytest.approx(0.22)
    assert result.unused_budget == pytest.approx(0.0)
    assert result.solver_status == "optimal"


def test_solver_does_not_select_negative_or_zero_effects():
    candidates = pd.DataFrame({
        "baseline_price": [50.0, 100.0, 80.0],
        "predicted_ite": [-0.10, 0.0, 0.08],
    })
    problem = build_optimization_problem(candidates, budget=20.0)

    result = solve_optimization_problem(problem)

    assert result.selected_indices == (2,)
    assert result.expected_incremental_purchases == pytest.approx(0.08)
    assert result.total_cost <= result.budget


def test_solver_handles_zero_budget():
    candidates = pd.DataFrame({
        "baseline_price": [100.0, 200.0],
        "predicted_ite": [0.10, 0.20],
    })
    problem = build_optimization_problem(candidates, budget=0.0)

    result = solve_optimization_problem(problem)

    assert result.selected_indices == ()
    assert result.total_cost == pytest.approx(0.0)
    assert result.expected_incremental_purchases == pytest.approx(0.0)
    assert result.unused_budget == pytest.approx(0.0)


def test_solver_rejects_mismatched_arrays():
    problem = build_optimization_problem(
        pd.DataFrame({
            "baseline_price": [100.0, 200.0],
            "predicted_ite": [0.1, 0.2],
        }),
        budget=30.0,
    )
    problem = type(problem)(
        expected_incremental_purchases=np.array([0.1]),
        promotion_costs=problem.promotion_costs,
        budget=problem.budget,
        discount_pct=problem.discount_pct,
    )

    with pytest.raises(ValueError, match="equal non-zero lengths"):
        solve_optimization_problem(problem)


def test_solver_rejects_invalid_costs():
    problem = build_optimization_problem(
        pd.DataFrame({
            "baseline_price": [100.0],
            "predicted_ite": [0.1],
        }),
        budget=30.0,
    )
    problem = type(problem)(
        expected_incremental_purchases=problem.expected_incremental_purchases,
        promotion_costs=np.array([np.nan]),
        budget=problem.budget,
        discount_pct=problem.discount_pct,
    )

    with pytest.raises(ValueError, match="costs must be finite and positive"):
        solve_optimization_problem(problem)
