"""Budget-constrained promotion allocation and SciPy MILP solver (Phase 7).

Integrates:
- Phase 4 Causal Inference (ITE / CATE effects)
- Phase 5 Uplift Modeling (Uplift segments, baseline probabilities, targeting recommendation)
- Phase 6 Dynamic Pricing (Baseline price, price elasticity, revenue models)

Implements:
1. Customer selection under marketing budget constraints.
2. Multi-tier discount depth selection (e.g. 5%, 10%, 15%, 20%).
3. Multiple campaign objectives: Incremental Revenue, Incremental Profit, Incremental Conversions.
4. Customer-level exclusivity constraints (no duplicate allocations).
5. Explainable decisions for every customer.
6. Multi-scenario comparative analysis across budgets, discount depths, and policies.
7. SciPy MILP solver status reporting and centralized artifact persistence.
"""

from __future__ import annotations

import sys
if __name__ == "__main__":
    sys.modules["backend.optimization"] = sys.modules["__main__"]

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import dok_matrix, csr_matrix

from backend.config import (
    CUSTOMER_PRICING_RECOMMENDATIONS_PATH,
    DEFAULT_ALLOWED_DISCOUNTS,
    DEFAULT_DISCOUNT_PCT,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_PRICE,
    DEFAULT_PROMOTION_BUDGET,
    PROMOTION_ALLOCATION_REPORT,
    PROMOTION_CUSTOMER_ALLOCATIONS_PATH,
    PROMOTION_SCENARIO_COMPARISON_PATH,
    SPLIT_DIR,
    SYNTHETIC_DATA_DIR,
    UPLIFT_SCORED_CUSTOMERS_PATH,
)

REQUIRED_COLUMNS = {"baseline_price", "predicted_ite"}


# =========================================================================
# 1. Backward-Compatible Phase 1 Baseline Problem & Solver
# =========================================================================

@dataclass(frozen=True)
class OptimizationProblem:
    """Inputs and mathematical definition for promotion allocation."""

    expected_incremental_purchases: np.ndarray
    promotion_costs: np.ndarray
    budget: float
    discount_pct: float
    decision_variable: str = "x_i in {0, 1}"

    @property
    def objective_description(self) -> str:
        return "Maximize sum(predicted_ite_i * x_i)"

    @property
    def constraint_description(self) -> str:
        return "sum(promotion_cost_i * x_i) <= budget"


@dataclass(frozen=True)
class OptimizationResult:
    """Solution and business metrics from the allocation solver."""

    selected_indices: tuple[int, ...]
    selected_count: int
    total_cost: float
    expected_incremental_purchases: float
    budget: float
    unused_budget: float
    discount_pct: float
    solver_status: str


def build_optimization_problem(
    candidates: pd.DataFrame,
    budget: float,
    discount_pct: float = DEFAULT_DISCOUNT_PCT,
) -> OptimizationProblem:
    """Build the binary promotion-allocation problem without solving it."""
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame.")

    missing = REQUIRED_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if candidates.empty:
        raise ValueError("candidates must contain at least one row.")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite non-negative number.")

    if (
        not np.isfinite(discount_pct)
        or discount_pct <= 0
        or discount_pct > 100
    ):
        raise ValueError("discount_pct must be greater than 0 and at most 100.")

    prices = candidates["baseline_price"].to_numpy(dtype=float)
    effects = candidates["predicted_ite"].to_numpy(dtype=float)

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("baseline_price must contain finite positive values.")

    if not np.isfinite(effects).all():
        raise ValueError("predicted_ite must contain only finite values.")

    costs = prices * (discount_pct / 100.0)

    return OptimizationProblem(
        expected_incremental_purchases=effects.copy(),
        promotion_costs=costs,
        budget=float(budget),
        discount_pct=float(discount_pct),
    )


def solve_optimization_problem(
    problem: OptimizationProblem,
) -> OptimizationResult:
    """Maximize predicted incremental purchases under the promotion budget.

    Each decision variable is binary: promote this opportunity or do not.
    Non-positive predicted effects are excluded from promotion.
    """
    effects = np.asarray(
        problem.expected_incremental_purchases, dtype=float
    ).reshape(-1)
    costs = np.asarray(problem.promotion_costs, dtype=float).reshape(-1)

    if effects.size == 0 or effects.size != costs.size:
        raise ValueError("Effects and costs must have equal non-zero lengths.")

    if not np.isfinite(effects).all():
        raise ValueError("Expected incremental purchases must be finite.")

    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Promotion costs must be finite and positive.")

    if not np.isfinite(problem.budget) or problem.budget < 0:
        raise ValueError("Budget must be finite and non-negative.")

    # milp minimizes, so negate effects to maximize their sum.
    upper_bounds = (effects > 0).astype(float)

    result = milp(
        c=-effects,
        integrality=np.ones(effects.size, dtype=int),
        bounds=Bounds(
            lb=np.zeros(effects.size),
            ub=upper_bounds,
        ),
        constraints=LinearConstraint(
            csr_matrix(costs.reshape(1, -1)),
            lb=-np.inf,
            ub=problem.budget,
        ),
        options={"disp": False},
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            f"Promotion optimization failed: {result.message}"
        )

    decisions = np.rint(result.x).astype(int)
    selected = tuple(np.flatnonzero(decisions == 1).tolist())
    total_cost = float(np.dot(decisions, costs))
    total_effect = float(np.dot(decisions, effects))

    # Guard against numerical tolerance at the budget boundary.
    if total_cost > problem.budget + 1e-7:
        raise RuntimeError("Solver result exceeds the promotion budget.")

    return OptimizationResult(
        selected_indices=selected,
        selected_count=len(selected),
        total_cost=total_cost,
        expected_incremental_purchases=total_effect,
        budget=float(problem.budget),
        unused_budget=max(0.0, float(problem.budget) - total_cost),
        discount_pct=float(problem.discount_pct),
        solver_status="optimal",
    )


# =========================================================================
# 2. Advanced Promotion Engine (Phase 7) Data Structures & Enums
# =========================================================================

class CampaignObjective(str, Enum):
    INCREMENTAL_REVENUE = "incremental_revenue"
    INCREMENTAL_PROFIT = "incremental_profit"
    INCREMENTAL_CONVERSIONS = "incremental_conversions"


class CostModel(str, Enum):
    REDEMPTION = "redemption"  # cost = (P * d) * P(Purchase=1 | d)
    LIABILITY = "liability"    # cost = (P * d) face-value liability


class TargetingPolicy(str, Enum):
    UNCONSTRAINED = "unconstrained"
    PERSUADABLES_ONLY = "persuadables_only"


@dataclass
class CustomerCandidate:
    """Consolidated customer representation integrating Phases 4, 5, and 6."""
    customer_id: int
    baseline_price: float
    baseline_probability: float
    predicted_uplift: float
    uplift_segment: str
    targeting_recommendation: str
    unit_cost: Optional[float] = None
    category: Optional[str] = None
    elasticity: Optional[float] = None


@dataclass
class CustomerAllocationDecision:
    """Individual customer allocation result with economic explainability."""
    customer_id: int
    uplift_segment: str
    baseline_probability: float
    predicted_uplift: float
    selected_discount_pct: float
    is_promoted: bool
    expected_incremental_demand: float
    expected_incremental_revenue: float
    expected_incremental_profit: Optional[float]
    expected_promotion_cost: float
    budget_impact: float
    roi: float
    selection_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdvancedCampaignResult:
    """Aggregate campaign outcome from the multi-choice allocation optimizer."""
    objective: str
    budget: float
    total_spend: float
    unused_budget: float
    budget_utilization_pct: float
    total_customers: int
    targeted_customers: int
    targeting_rate: float
    total_expected_incremental_demand: float
    total_expected_incremental_revenue: float
    total_expected_incremental_profit: Optional[float]
    campaign_roi: float
    solver_status: str
    cost_model: str
    discount_distribution: Dict[str, int]
    segment_breakdown: Dict[str, Dict[str, Any]]
    allocations: List[CustomerAllocationDecision] = field(default_factory=list)

    def to_dict(self, include_allocations: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_allocations:
            d.pop("allocations", None)
        return d


# =========================================================================
# 3. Multi-Choice Knapsack Promotion Allocation Optimizer
# =========================================================================

class AdvancedPromotionOptimizer:
    """Production promotion allocation engine using SciPy MILP.
    
    Solves:
        Maximize sum_{i, d} Value_{i, d} * x_{i, d}
        Subject to:
            sum_{d in D} x_{i, d} <= 1   for all i (Customer exclusivity)
            sum_{i, d} Cost_{i, d} * x_{i, d} <= Budget (Budget feasibility)
            x_{i, d} in {0, 1}
    """

    def __init__(
        self,
        min_price: float = DEFAULT_MIN_PRICE,
        max_price: float = DEFAULT_MAX_PRICE,
        max_discount_pct: float = 20.0,
    ):
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.max_discount_pct = float(max_discount_pct)

    def solve(
        self,
        candidates: List[CustomerCandidate],
        budget: float,
        allowed_discounts: Tuple[float, ...] = DEFAULT_ALLOWED_DISCOUNTS,
        objective: CampaignObjective = CampaignObjective.INCREMENTAL_REVENUE,
        policy: TargetingPolicy = TargetingPolicy.UNCONSTRAINED,
        cost_model: CostModel = CostModel.REDEMPTION,
        time_limit_seconds: float = 30.0,
    ) -> AdvancedCampaignResult:
        """Execute the multi-tier promotion optimization under all constraints."""
        n_cust = len(candidates)
        if n_cust == 0:
            raise ValueError("Candidates list cannot be empty.")

        if not np.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be a finite non-negative number.")

        discounts = sorted([d for d in allowed_discounts if 0 < d <= self.max_discount_pct])
        if not discounts:
            raise ValueError("At least one valid positive discount depth must be provided.")

        n_disc = len(discounts)
        n_vars = n_cust * n_disc

        # Objective vector c (minimize -Value)
        c = np.zeros(n_vars, dtype=float)
        ub_vars = np.ones(n_vars, dtype=float)

        # Store candidate option metrics for reporting: indexed by k = i * n_disc + j
        option_metadata = []

        has_cost_data = any(c.unit_cost is not None for c in candidates)
        if objective == CampaignObjective.INCREMENTAL_PROFIT and not has_cost_data:
            print("Warning: Unit cost/margin data is unavailable. Falling back to INCREMENTAL_REVENUE objective.")
            objective = CampaignObjective.INCREMENTAL_REVENUE

        for i, cust in enumerate(candidates):
            p0 = cust.baseline_price
            q0 = cust.baseline_probability
            tau_base = cust.predicted_uplift
            segment = cust.uplift_segment
            unit_cost = cust.unit_cost

            # Exclude invalid price ranges
            is_price_valid = (p0 >= self.min_price) and (p0 <= self.max_price)

            for j, d in enumerate(discounts):
                k = i * n_disc + j

                # Uplift scaling proportional to discount depth (normalized at 10% discount)
                tau_d = tau_base * (d / 10.0)
                q1_d = float(np.clip(q0 + tau_d, 0.0, 1.0))
                p1_d = p0 * (1.0 - d / 100.0)

                # Economic metrics
                rev0 = p0 * q0
                rev1_d = p1_d * q1_d
                inc_rev = rev1_d - rev0
                inc_demand = tau_d

                inc_profit = None
                if unit_cost is not None:
                    prof0 = (p0 - unit_cost) * q0
                    prof1_d = (p1_d - unit_cost) * q1_d
                    inc_profit = prof1_d - prof0

                # Promotion Cost
                if cost_model == CostModel.REDEMPTION:
                    # Incurred upon redemption / purchase
                    cost_d = (p0 * d / 100.0) * q1_d
                else:
                    # Face-value liability
                    cost_d = p0 * d / 100.0

                # Determine objective value
                if objective == CampaignObjective.INCREMENTAL_REVENUE:
                    val = inc_rev
                elif objective == CampaignObjective.INCREMENTAL_PROFIT:
                    val = inc_profit if inc_profit is not None else inc_rev
                else:  # INCREMENTAL_CONVERSIONS
                    val = inc_demand

                roi = (inc_rev / cost_d) if cost_d > 1e-6 else 0.0

                option_metadata.append({
                    "customer_id": cust.customer_id,
                    "discount_pct": d,
                    "inc_demand": inc_demand,
                    "inc_rev": inc_rev,
                    "inc_profit": inc_profit,
                    "cost": cost_d,
                    "roi": roi,
                    "val": val,
                })

                # Eligibility filters:
                # 1. Non-positive value filter
                if val <= 0:
                    ub_vars[k] = 0.0

                # 2. Price bounds
                if not is_price_valid:
                    ub_vars[k] = 0.0

                # 3. Sleeping Dogs (promotion fatigue / negative uplift)
                if segment == "Sleeping Dogs":
                    ub_vars[k] = 0.0

                # 4. Policy filter
                if policy == TargetingPolicy.PERSUADABLES_ONLY and segment != "Persuadables":
                    ub_vars[k] = 0.0

                # SciPy milp minimizes c^T x
                c[k] = -val

        # Construct constraints matrix A
        # Row 0: Budget constraint
        # Rows 1..N: Customer exclusivity constraints
        n_constraints = 1 + n_cust
        A = dok_matrix((n_constraints, n_vars), dtype=float)

        # Budget constraint
        for k in range(n_vars):
            A[0, k] = option_metadata[k]["cost"]

        # Customer exclusivity constraints: sum_j x_{i, j} <= 1
        for i in range(n_cust):
            row_idx = 1 + i
            for j in range(n_disc):
                A[row_idx, i * n_disc + j] = 1.0

        A_csr = A.tocsr()

        lb_con = -np.inf * np.ones(n_constraints)
        ub_con = np.zeros(n_constraints)
        ub_con[0] = float(budget)
        ub_con[1:] = 1.0
        lb_con[1:] = 0.0

        # Zero-budget early exit
        if budget <= 1e-6:
            return self._build_empty_result(
                candidates, budget, objective, policy, cost_model, status="optimal"
            )

        # Solve MILP
        res = milp(
            c=c,
            integrality=np.ones(n_vars, dtype=int),
            bounds=Bounds(lb=np.zeros(n_vars), ub=ub_vars),
            constraints=LinearConstraint(A_csr, lb_con, ub_con),
            options={"time_limit": time_limit_seconds, "disp": False},
        )

        status_map = {
            0: "optimal",
            1: "iteration_or_time_limit",
            2: "infeasible",
            3: "unbounded",
        }
        solver_status = status_map.get(res.status, f"solver_error_{res.status}")

        if not res.success and res.status != 0:
            if res.status == 2:
                return self._build_empty_result(
                    candidates, budget, objective, policy, cost_model, status="infeasible"
                )
            raise RuntimeError(f"MILP Solver failed: status {res.status} ({res.message})")

        decisions_binary = np.rint(res.x).astype(int)

        # Extract customer allocations
        allocations = []
        discount_dist = {f"{d}%": 0 for d in discounts}
        discount_dist["0% (No Promo)"] = 0

        segment_stats: Dict[str, Dict[str, Any]] = {}
        for s_name in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs", "Unclassified"]:
            segment_stats[s_name] = {
                "total_customers": 0,
                "targeted_customers": 0,
                "spend": 0.0,
                "incremental_revenue": 0.0,
                "incremental_demand": 0.0,
            }

        tot_spend = 0.0
        tot_inc_rev = 0.0
        tot_inc_demand = 0.0
        tot_inc_profit = 0.0 if has_cost_data else None
        targeted_count = 0

        for i, cust in enumerate(candidates):
            seg = cust.uplift_segment if cust.uplift_segment in segment_stats else "Unclassified"
            segment_stats[seg]["total_customers"] += 1

            # Check if any discount was chosen
            chosen_j = None
            for j in range(n_disc):
                k = i * n_disc + j
                if decisions_binary[k] == 1:
                    chosen_j = j
                    break

            if chosen_j is not None:
                k = i * n_disc + chosen_j
                meta = option_metadata[k]
                d_val = discounts[chosen_j]

                is_promoted = True
                targeted_count += 1
                discount_dist[f"{d_val}%"] += 1

                c_spend = meta["cost"]
                c_inc_rev = meta["inc_rev"]
                c_inc_demand = meta["inc_demand"]
                c_inc_profit = meta["inc_profit"]

                tot_spend += c_spend
                tot_inc_rev += c_inc_rev
                tot_inc_demand += c_inc_demand
                if tot_inc_profit is not None and c_inc_profit is not None:
                    tot_inc_profit += c_inc_profit

                segment_stats[seg]["targeted_customers"] += 1
                segment_stats[seg]["spend"] += c_spend
                segment_stats[seg]["incremental_revenue"] += c_inc_rev
                segment_stats[seg]["incremental_demand"] += c_inc_demand

                exp_text = (
                    f"Selected {d_val}% discount promotion. Customer is in '{seg}' with predicted uplift "
                    f"+{cust.predicted_uplift:.2%}. Expected incremental revenue of ${c_inc_rev:.2f} "
                    f"for ${c_spend:.2f} promotional cost (ROI = {meta['roi']:.2f})."
                )

                allocations.append(CustomerAllocationDecision(
                    customer_id=cust.customer_id,
                    uplift_segment=seg,
                    baseline_probability=round(cust.baseline_probability, 4),
                    predicted_uplift=round(cust.predicted_uplift, 4),
                    selected_discount_pct=float(d_val),
                    is_promoted=True,
                    expected_incremental_demand=round(c_inc_demand, 4),
                    expected_incremental_revenue=round(c_inc_rev, 4),
                    expected_incremental_profit=round(c_inc_profit, 4) if c_inc_profit is not None else None,
                    expected_promotion_cost=round(c_spend, 4),
                    budget_impact=round(c_spend, 4),
                    roi=round(meta["roi"], 4),
                    selection_explanation=exp_text,
                ))
            else:
                discount_dist["0% (No Promo)"] += 1

                # Formulate non-selection rationale
                if seg == "Sleeping Dogs":
                    reason = "Excluded: Customer is a 'Sleeping Dog' with risk of promotion fatigue."
                elif seg == "Sure Things":
                    reason = "Not targeted: High organic purchase probability (Sure Thing); discounting causes cannibalization."
                elif seg == "Lost Causes":
                    reason = "Not targeted: Extremely low conversion probability regardless of promotion."
                elif cust.predicted_uplift <= 0:
                    reason = "Not targeted: Zero or negative predicted treatment effect."
                elif policy == TargetingPolicy.PERSUADABLES_ONLY and seg != "Persuadables":
                    reason = f"Excluded by campaign policy: Only Persuadables eligible (customer is {seg})."
                else:
                    reason = "Not targeted: Budget limit allocated to higher-ROI opportunities."

                allocations.append(CustomerAllocationDecision(
                    customer_id=cust.customer_id,
                    uplift_segment=seg,
                    baseline_probability=round(cust.baseline_probability, 4),
                    predicted_uplift=round(cust.predicted_uplift, 4),
                    selected_discount_pct=0.0,
                    is_promoted=False,
                    expected_incremental_demand=0.0,
                    expected_incremental_revenue=0.0,
                    expected_incremental_profit=0.0 if has_cost_data else None,
                    expected_promotion_cost=0.0,
                    budget_impact=0.0,
                    roi=0.0,
                    selection_explanation=reason,
                ))

        unused = max(0.0, float(budget) - tot_spend)
        util_pct = (tot_spend / float(budget)) if budget > 0 else 0.0
        roi_total = (tot_inc_rev / tot_spend) if tot_spend > 0 else 0.0

        return AdvancedCampaignResult(
            objective=objective.value,
            budget=round(float(budget), 2),
            total_spend=round(tot_spend, 2),
            unused_budget=round(unused, 2),
            budget_utilization_pct=round(util_pct, 4),
            total_customers=n_cust,
            targeted_customers=targeted_count,
            targeting_rate=round(targeted_count / n_cust, 4),
            total_expected_incremental_demand=round(tot_inc_demand, 4),
            total_expected_incremental_revenue=round(tot_inc_rev, 2),
            total_expected_incremental_profit=round(tot_inc_profit, 2) if tot_inc_profit is not None else None,
            campaign_roi=round(roi_total, 4),
            solver_status=solver_status,
            cost_model=cost_model.value,
            discount_distribution=discount_dist,
            segment_breakdown=segment_stats,
            allocations=allocations,
        )

    def _build_empty_result(
        self,
        candidates: List[CustomerCandidate],
        budget: float,
        objective: CampaignObjective,
        policy: TargetingPolicy,
        cost_model: CostModel,
        status: str,
    ) -> AdvancedCampaignResult:
        allocs = [
            CustomerAllocationDecision(
                customer_id=c.customer_id,
                uplift_segment=c.uplift_segment,
                baseline_probability=round(c.baseline_probability, 4),
                predicted_uplift=round(c.predicted_uplift, 4),
                selected_discount_pct=0.0,
                is_promoted=False,
                expected_incremental_demand=0.0,
                expected_incremental_revenue=0.0,
                expected_incremental_profit=0.0 if c.unit_cost is not None else None,
                expected_promotion_cost=0.0,
                budget_impact=0.0,
                roi=0.0,
                selection_explanation=f"Zero allocation due to budget constraints (solver status: {status}).",
            )
            for c in candidates
        ]
        return AdvancedCampaignResult(
            objective=objective.value,
            budget=float(budget),
            total_spend=0.0,
            unused_budget=float(budget),
            budget_utilization_pct=0.0,
            total_customers=len(candidates),
            targeted_customers=0,
            targeting_rate=0.0,
            total_expected_incremental_demand=0.0,
            total_expected_incremental_revenue=0.0,
            total_expected_incremental_profit=0.0 if any(c.unit_cost is not None for c in candidates) else None,
            campaign_roi=0.0,
            solver_status=status,
            cost_model=cost_model.value,
            discount_distribution={"0% (No Promo)": len(candidates)},
            segment_breakdown={},
            allocations=allocs,
        )


# =========================================================================
# 4. Multi-Scenario Analysis Engine
# =========================================================================

class ScenarioAnalyzer:
    """Evaluates and benchmarks promotion policies across budgets, objectives, and discount depths."""

    def __init__(self, optimizer: AdvancedPromotionOptimizer):
        self.optimizer = optimizer

    def run_scenarios(
        self,
        candidates: List[CustomerCandidate],
        base_budget: float = DEFAULT_PROMOTION_BUDGET,
    ) -> Dict[str, Any]:
        """Run standard set of comparative marketing scenarios."""
        scenarios = {
            "1. Low Budget ($1,000)": {
                "budget": 1000.0,
                "discounts": (5.0, 10.0, 15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
            "2. Baseline Budget ($2,500)": {
                "budget": base_budget,
                "discounts": (5.0, 10.0, 15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
            "3. High Budget ($5,000)": {
                "budget": 5000.0,
                "discounts": (5.0, 10.0, 15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
            "4. Shallow Discounts (5%, 10%)": {
                "budget": base_budget,
                "discounts": (5.0, 10.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
            "5. Deep Discounts (15%, 20%)": {
                "budget": base_budget,
                "discounts": (15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
            "6. Persuadables-Only Policy": {
                "budget": base_budget,
                "discounts": (5.0, 10.0, 15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_REVENUE,
                "policy": TargetingPolicy.PERSUADABLES_ONLY,
            },
            "7. Uplift Conversions Objective": {
                "budget": base_budget,
                "discounts": (5.0, 10.0, 15.0, 20.0),
                "objective": CampaignObjective.INCREMENTAL_CONVERSIONS,
                "policy": TargetingPolicy.UNCONSTRAINED,
            },
        }

        comparison_records = []
        detailed_results = {}

        for s_name, cfg in scenarios.items():
            res = self.optimizer.solve(
                candidates=candidates,
                budget=cfg["budget"],
                allowed_discounts=cfg["discounts"],
                objective=cfg["objective"],
                policy=cfg["policy"],
            )

            p_share = 0.0
            if "Persuadables" in res.segment_breakdown:
                p_tar = res.segment_breakdown["Persuadables"]["targeted_customers"]
                p_share = p_tar / res.targeted_customers if res.targeted_customers > 0 else 0.0

            comparison_records.append({
                "scenario_name": s_name,
                "budget": res.budget,
                "spend": res.total_spend,
                "budget_utilization_pct": res.budget_utilization_pct,
                "targeted_customers": res.targeted_customers,
                "targeting_rate": res.targeting_rate,
                "expected_incremental_conversions": res.total_expected_incremental_demand,
                "expected_incremental_revenue": res.total_expected_incremental_revenue,
                "campaign_roi": res.campaign_roi,
                "persuadables_share": round(p_share, 4),
                "solver_status": res.solver_status,
            })

            detailed_results[s_name] = res.to_dict(include_allocations=False)

        return {
            "comparison_summary": comparison_records,
            "scenario_details": detailed_results,
        }


# =========================================================================
# 5. Pipeline Integration & Artifact Generation (Phases 4, 5, 6 -> 7)
# =========================================================================

def load_integrated_candidates(
    test_path: Optional[Path] = None,
    uplift_scores_path: Optional[Path] = None,
    pricing_recs_path: Optional[Path] = None,
) -> List[CustomerCandidate]:
    """Integrate Phase 4 test split, Phase 5 uplift scores, and Phase 6 pricing recommendations."""
    te_path = test_path or (SPLIT_DIR / "test.csv")
    u_path = uplift_scores_path or UPLIFT_SCORED_CUSTOMERS_PATH
    p_path = pricing_recs_path or CUSTOMER_PRICING_RECOMMENDATIONS_PATH

    df_test = pd.read_csv(te_path)
    df_uplift = pd.read_csv(u_path) if u_path.exists() else None
    df_pricing = pd.read_csv(p_path) if p_path.exists() else None

    # Customer-level grouping
    candidates = []

    # Map uplift info
    uplift_map = {}
    if df_uplift is not None:
        for cid, group in df_uplift.groupby("customer_id"):
            uplift_map[cid] = {
                "uplift": float(group["predicted_uplift"].mean()),
                "baseline_prob": float(group["baseline_probability"].mean()) if "baseline_probability" in group else 0.25,
                "segment": group["segment"].iloc[0] if "segment" in group else "Persuadables",
                "recommendation": group["targeting_recommendation"].iloc[0] if "targeting_recommendation" in group else "Target",
            }

    # Map pricing info
    pricing_map = {}
    if df_pricing is not None:
        for _, row in df_pricing.iterrows():
            cid = int(row["customer_id"])
            pricing_map[cid] = {
                "mean_price": float(row["mean_current_price"]),
                "elasticity": float(row["mean_elasticity"]),
            }

    for cid, group in df_test.groupby("customer_id"):
        cid = int(cid)
        p_info = pricing_map.get(cid, {
            "mean_price": float(group["baseline_price"].mean()),
            "elasticity": -0.05,
        })
        u_info = uplift_map.get(cid, {
            "uplift": 0.05,
            "baseline_prob": 0.25,
            "segment": "Persuadables",
            "recommendation": "Target",
        })

        candidates.append(CustomerCandidate(
            customer_id=cid,
            baseline_price=p_info["mean_price"],
            baseline_probability=u_info["baseline_prob"],
            predicted_uplift=u_info["uplift"],
            uplift_segment=u_info["segment"],
            targeting_recommendation=u_info["recommendation"],
            unit_cost=None,  # Not present in synthetic retail data
            category=str(group["category"].iloc[0]),
            elasticity=p_info["elasticity"],
        ))

    return candidates


def run_promotion_optimization_pipeline(
    budget: float = DEFAULT_PROMOTION_BUDGET,
    allowed_discounts: Tuple[float, ...] = DEFAULT_ALLOWED_DISCOUNTS,
    objective: CampaignObjective = CampaignObjective.INCREMENTAL_REVENUE,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end Phase 7 advanced promotion optimization and scenario comparison."""
    out_dir = output_dir or SYNTHETIC_DATA_DIR

    print("=" * 70)
    print("ECONOCAUSAL ADVANCED PROMOTION OPTIMIZATION ENGINE (PHASE 7)")
    print(f"Configured Budget: ${budget:,.2f} | Objective: {objective.value}")
    print("=" * 70)

    # 1. Integrate Candidates
    t0 = time.perf_counter()
    print("\n[1/4] Integrating Candidate Customers from Phases 4, 5, and 6...")
    candidates = load_integrated_candidates()
    print(f"  Loaded {len(candidates)} unique customer candidates in {time.perf_counter() - t0:.3f}s")

    # 2. Advanced Optimization Solver
    t1 = time.perf_counter()
    print(f"\n[2/4] Solving Budget-Constrained Promotion Optimization (Budget: ${budget:,.2f})...")
    optimizer = AdvancedPromotionOptimizer()
    campaign_result = optimizer.solve(
        candidates=candidates,
        budget=budget,
        allowed_discounts=allowed_discounts,
        objective=objective,
    )
    opt_duration = time.perf_counter() - t1
    print(f"  Solved MILP in {opt_duration:.3f}s | Status: {campaign_result.solver_status}")
    print(f"  Targeted Customers: {campaign_result.targeted_customers}/{campaign_result.total_customers} ({campaign_result.targeting_rate:.1%})")
    print(f"  Total Promotional Spend: ${campaign_result.total_spend:,.2f} (Utilization: {campaign_result.budget_utilization_pct:.1%})")
    print(f"  Expected Incremental Conversions: {campaign_result.total_expected_incremental_demand:.1f}")
    print(f"  Expected Incremental Revenue: ${campaign_result.total_expected_incremental_revenue:,.2f} (Campaign ROI: {campaign_result.campaign_roi:.2f}x)")

    print("\n  Discount Depth Distribution:")
    for d_str, count in campaign_result.discount_distribution.items():
        print(f"    * {d_str:18s}: {count:3d} customers")

    print("\n  Uplift Segment Allocations:")
    for seg, data in campaign_result.segment_breakdown.items():
        if data["total_customers"] > 0:
            print(f"    * {seg:15s}: {data['targeted_customers']:3d}/{data['total_customers']:3d} targeted | Spend: ${data['spend']:,.2f} | Inc Rev: ${data['incremental_revenue']:,.2f}")

    # 3. Scenario Analysis
    t2 = time.perf_counter()
    print("\n[3/4] Running Comprehensive Multi-Scenario Analysis (7 Scenarios)...")
    analyzer = ScenarioAnalyzer(optimizer)
    scenarios = analyzer.run_scenarios(candidates, base_budget=budget)
    print(f"  Evaluated 7 scenarios in {time.perf_counter() - t2:.3f}s")

    print("\n  Scenario Comparison Summary:")
    df_scen = pd.DataFrame(scenarios["comparison_summary"])
    for _, sc in df_scen.iterrows():
        print(f"    - {sc['scenario_name']:32s} | Target: {sc['targeted_customers']:3d} ({sc['targeting_rate']:.1%}) | Spend: ${sc['spend']:7.2f} | Inc Rev: ${sc['expected_incremental_revenue']:7.2f} | ROI: {sc['campaign_roi']:.2f}x")

    # 4. Save Artifacts
    t3 = time.perf_counter()
    print("\n[4/4] Persisting Optimization Results and Scenario Reports...")
    alloc_csv_path = out_dir / "promotion_customer_allocations.csv"
    df_alloc = pd.DataFrame([a.to_dict() for a in campaign_result.allocations])
    df_alloc.to_csv(alloc_csv_path, index=False)
    print(f"  Saved customer allocations to: {alloc_csv_path}")

    scen_path = out_dir / "promotion_scenario_comparison.json"
    with open(scen_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved scenario comparisons to: {scen_path}")

    report_path = out_dir / "promotion_allocation_report.json"
    report_data = {
        "evaluation": "Phase 7 Advanced Promotion Optimization Engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_summary": campaign_result.to_dict(include_allocations=False),
        "scenario_comparison": scenarios["comparison_summary"],
        "artifact_paths": {
            "allocation_report": str(report_path),
            "customer_allocations": str(alloc_csv_path),
            "scenario_comparison": str(scen_path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved campaign allocation report to: {report_path}")

    print("\nPhase 7 Advanced Promotion Optimization Engine completed successfully.")
    return report_data


if __name__ == "__main__":
    run_promotion_optimization_pipeline()
