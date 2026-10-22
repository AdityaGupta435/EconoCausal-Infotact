"""Budget-constrained promotion allocation and SciPy MILP solver."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


DEFAULT_DISCOUNT_PCT = 10.0
REQUIRED_COLUMNS = {"baseline_price", "predicted_ite"}


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
    # A zero upper bound prevents promotions with non-positive effects.
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
