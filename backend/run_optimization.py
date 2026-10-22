import json

import numpy as np
import pandas as pd

from backend.econml_model import (
    FEATURES,
    SPLIT_DIR,
    load_training_data,
    make_model,
)
from backend.optimization import (
    DEFAULT_DISCOUNT_PCT,
    build_optimization_problem,
    solve_optimization_problem,
)


ROOT = SPLIT_DIR.parents[2]
OUTPUT_PATH = ROOT / "data" / "synthetic" / "optimization_report.json"


def main():
    train, X_train, T_train, Y_train, preprocessor = load_training_data()

    model = make_model()
    model.fit(Y_train, T_train, X=X_train)

    candidates = pd.read_csv(SPLIT_DIR / "test.csv")
    X_candidates = np.asarray(
        preprocessor.transform(candidates[FEATURES]),
        dtype=float,
    )
    predicted_ite = np.asarray(
        model.effect(X_candidates, T0=0, T1=1)
    ).reshape(-1)

    if not np.isfinite(predicted_ite).all():
        raise ValueError("Predicted treatment effects contain non-finite values.")

    optimization_candidates = candidates[
        ["customer_id", "product_id", "category", "baseline_price"]
    ].copy()
    optimization_candidates["predicted_ite"] = predicted_ite

    budget = float(
        optimization_candidates["baseline_price"].sum()
        * DEFAULT_DISCOUNT_PCT / 100.0
        * 0.10
    )

    problem = build_optimization_problem(
        optimization_candidates,
        budget=budget,
        discount_pct=DEFAULT_DISCOUNT_PCT,
    )
    result = solve_optimization_problem(problem)

    selected = optimization_candidates.iloc[
        list(result.selected_indices)
    ].copy()
    selected["promotion_cost"] = problem.promotion_costs[
        list(result.selected_indices)
    ]
    selected["expected_incremental_purchases"] = selected["predicted_ite"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "synthetic retail held-out test split",
        "candidate_count": int(len(candidates)),
        "training_rows": int(len(train)),
        "discount_pct": result.discount_pct,
        "budget_assumption": "10% of the test candidates' total 10%-discount cost",
        "budget": result.budget,
        "selected_count": result.selected_count,
        "total_cost": result.total_cost,
        "unused_budget": result.unused_budget,
        "expected_incremental_purchases": result.expected_incremental_purchases,
        "solver_status": result.solver_status,
        "causal_interpretation": (
            "Expected incremental purchases are summed predicted ITEs. "
            "These are model-based estimates on synthetic data, not observed "
            "incremental purchases or a real-world causal guarantee."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selected.to_csv(
        ROOT / "data" / "synthetic" / "optimization_selected_candidates.csv",
        index=False,
    )

    print("OPTIMIZATION EVALUATION")
    print("-" * 40)
    print(f"Training rows: {len(train):,}")
    print(f"Test candidates: {len(candidates):,}")
    print(f"Fixed discount: {result.discount_pct:.1f}%")
    print(f"Budget: {result.budget:.2f}")
    print(f"Selected promotions: {result.selected_count:,}")
    print(f"Total estimated cost: {result.total_cost:.2f}")
    print(f"Unused budget: {result.unused_budget:.2f}")
    print(
        "Expected incremental purchases (model-based): "
        f"{result.expected_incremental_purchases:.4f}"
    )
    print(f"Solver status: {result.solver_status}")
    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
