"""Budget-constrained promotion allocation formulation.

Day 12 defines the optimization problem only.
The numerical solver will be implemented separately on Day 13.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def build_optimization_problem(
    candidates: pd.DataFrame,
    budget: float,
    discount_pct: float = DEFAULT_DISCOUNT_PCT,
) -> OptimizationProblem:
    """Build the binary promotion-allocation problem without solving it.

    Each row represents one customer-product opportunity.
    ``predicted_ite`` must be estimated by a fitted causal model.
    The fixed discount is used for cost accounting, not as a causal dose model.
    """
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
