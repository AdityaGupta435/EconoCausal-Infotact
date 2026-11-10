"""Comprehensive test suite for Phase 7: Advanced Promotion Optimization Engine."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.config import (
    PROMOTION_ALLOCATION_REPORT,
    PROMOTION_CUSTOMER_ALLOCATIONS_PATH,
    PROMOTION_SCENARIO_COMPARISON_PATH,
)
from backend.optimization import (
    AdvancedCampaignResult,
    AdvancedPromotionOptimizer,
    CampaignObjective,
    CostModel,
    CustomerCandidate,
    ScenarioAnalyzer,
    TargetingPolicy,
    load_integrated_candidates,
)


@pytest.fixture
def sample_candidates() -> list[CustomerCandidate]:
    """Generates controlled synthetic candidates for testing optimization logic."""
    np.random.seed(42)
    candidates = []
    segments = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]

    for i in range(1, 41):  # 40 customers
        seg = segments[(i - 1) % 4]
        if seg == "Persuadables":
            uplift = 0.08
            base_prob = 0.20
            rec = "Target"
        elif seg == "Sure Things":
            uplift = 0.01
            base_prob = 0.65
            rec = "Do Not Target"
        elif seg == "Lost Causes":
            uplift = 0.005
            base_prob = 0.10
            rec = "Do Not Target"
        else:  # Sleeping Dogs
            uplift = -0.05
            base_prob = 0.30
            rec = "Do Not Target"

        candidates.append(CustomerCandidate(
            customer_id=i,
            baseline_price=50.0 + (i % 5) * 10.0,
            baseline_probability=base_prob,
            predicted_uplift=uplift,
            uplift_segment=seg,
            targeting_recommendation=rec,
            unit_cost=None,
            category="Fashion",
            elasticity=-0.05,
        ))
    return candidates


def test_advanced_optimizer_budget_feasibility(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    budget = 100.0  # Tight budget

    result = optimizer.solve(
        candidates=sample_candidates,
        budget=budget,
        allowed_discounts=(5.0, 10.0, 15.0, 20.0),
        objective=CampaignObjective.INCREMENTAL_REVENUE,
    )

    assert result.solver_status == "optimal"
    assert result.total_spend <= budget + 1e-6
    assert result.unused_budget >= 0.0
    assert result.budget_utilization_pct <= 1.0


def test_customer_exclusivity(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    budget = 500.0

    result = optimizer.solve(
        candidates=sample_candidates,
        budget=budget,
        allowed_discounts=(5.0, 10.0, 15.0, 20.0),
    )

    # Verify exactly one allocation decision per customer
    assert len(result.allocations) == len(sample_candidates)
    allocated_ids = [a.customer_id for a in result.allocations]
    assert len(allocated_ids) == len(set(allocated_ids))

    # Each customer has at most one non-zero discount
    for alloc in result.allocations:
        assert alloc.selected_discount_pct in {0.0, 5.0, 10.0, 15.0, 20.0}
        if alloc.is_promoted:
            assert alloc.selected_discount_pct > 0.0
        else:
            assert alloc.selected_discount_pct == 0.0


def test_sleeping_dogs_are_never_targeted(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    budget = 10000.0  # Infinite/huge budget

    result = optimizer.solve(
        candidates=sample_candidates,
        budget=budget,
        allowed_discounts=(5.0, 10.0, 15.0, 20.0),
    )

    # Sleeping Dogs must NEVER be promoted regardless of budget
    for alloc in result.allocations:
        if alloc.uplift_segment == "Sleeping Dogs":
            assert not alloc.is_promoted
            assert alloc.selected_discount_pct == 0.0
            assert "Sleeping Dog" in alloc.selection_explanation


def test_persuadables_only_policy(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    budget = 10000.0

    result = optimizer.solve(
        candidates=sample_candidates,
        budget=budget,
        allowed_discounts=(5.0, 10.0, 15.0, 20.0),
        policy=TargetingPolicy.PERSUADABLES_ONLY,
    )

    for alloc in result.allocations:
        if alloc.is_promoted:
            assert alloc.uplift_segment == "Persuadables"
        elif alloc.uplift_segment != "Persuadables":
            assert not alloc.is_promoted
            assert alloc.selected_discount_pct == 0.0


def test_missing_cost_margin_behavior(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    # All candidates have unit_cost=None
    result = optimizer.solve(
        candidates=sample_candidates,
        budget=200.0,
        objective=CampaignObjective.INCREMENTAL_PROFIT,
    )

    # Should fall back cleanly without error and leave profit as None
    assert result.solver_status == "optimal"
    assert result.total_expected_incremental_profit is None
    for alloc in result.allocations:
        assert alloc.expected_incremental_profit is None


def test_valid_cost_margin_computes_profit(sample_candidates):
    # Inject unit cost
    candidates_with_cost = []
    for c in sample_candidates:
        candidates_with_cost.append(CustomerCandidate(
            customer_id=c.customer_id,
            baseline_price=c.baseline_price,
            baseline_probability=c.baseline_probability,
            predicted_uplift=c.predicted_uplift,
            uplift_segment=c.uplift_segment,
            targeting_recommendation=c.targeting_recommendation,
            unit_cost=c.baseline_price * 0.40,  # 40% COGS
            category=c.category,
            elasticity=c.elasticity,
        ))

    optimizer = AdvancedPromotionOptimizer()
    result = optimizer.solve(
        candidates=candidates_with_cost,
        budget=300.0,
        objective=CampaignObjective.INCREMENTAL_PROFIT,
    )

    assert result.solver_status == "optimal"
    assert result.total_expected_incremental_profit is not None
    assert np.isfinite(result.total_expected_incremental_profit)
    for alloc in result.allocations:
        if alloc.is_promoted:
            assert alloc.expected_incremental_profit is not None


def test_zero_budget_handling(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    result = optimizer.solve(candidates=sample_candidates, budget=0.0)

    assert result.solver_status == "optimal"
    assert result.total_spend == 0.0
    assert result.targeted_customers == 0
    assert result.targeting_rate == 0.0
    assert result.total_expected_incremental_revenue == 0.0
    assert all(not a.is_promoted for a in result.allocations)


def test_infeasible_or_negative_budget_raises(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    with pytest.raises(ValueError, match="Budget must be a finite non-negative number"):
        optimizer.solve(candidates=sample_candidates, budget=-50.0)


def test_scenario_analyzer_evaluates_all_scenarios(sample_candidates):
    optimizer = AdvancedPromotionOptimizer()
    analyzer = ScenarioAnalyzer(optimizer)

    results = analyzer.run_scenarios(candidates=sample_candidates, base_budget=200.0)
    assert "comparison_summary" in results
    assert "scenario_details" in results
    assert len(results["comparison_summary"]) == 7

    for scen in results["comparison_summary"]:
        assert "scenario_name" in scen
        assert "spend" in scen
        assert "expected_incremental_revenue" in scen
        assert "campaign_roi" in scen
        assert scen["spend"] <= scen["budget"] + 1e-4


def test_phase7_artifacts_integrity():
    assert PROMOTION_ALLOCATION_REPORT.exists(), f"Missing: {PROMOTION_ALLOCATION_REPORT}"
    assert PROMOTION_CUSTOMER_ALLOCATIONS_PATH.exists(), f"Missing: {PROMOTION_CUSTOMER_ALLOCATIONS_PATH}"
    assert PROMOTION_SCENARIO_COMPARISON_PATH.exists(), f"Missing: {PROMOTION_SCENARIO_COMPARISON_PATH}"

    # Verify allocation report JSON
    with open(PROMOTION_ALLOCATION_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "campaign_summary" in report
    assert report["campaign_summary"]["solver_status"] == "optimal"
    assert report["campaign_summary"]["total_customers"] == 400
    assert report["campaign_summary"]["targeted_customers"] > 0
    assert "scenario_comparison" in report
    assert len(report["scenario_comparison"]) == 7

    # Verify customer allocations CSV
    df_alloc = pd.read_csv(PROMOTION_CUSTOMER_ALLOCATIONS_PATH)
    assert len(df_alloc) == 400
    expected_cols = {
        "customer_id", "uplift_segment", "selected_discount_pct", "is_promoted",
        "expected_incremental_demand", "expected_incremental_revenue",
        "expected_promotion_cost", "selection_explanation",
    }
    assert expected_cols.issubset(set(df_alloc.columns))
    assert df_alloc["customer_id"].nunique() == 400

