
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.causal_model import DATA_DIR, estimate_ate
from backend.optimization import (
    build_optimization_problem,
    solve_optimization_problem,
)


ROOT = Path(__file__).resolve().parents[1]
UPLIFT_REPORT = ROOT / "data" / "synthetic" / "uplift_evaluation.json"

app = FastAPI(
    title="EconoCausal API",
    description="Causal inference for dynamic pricing",
    version="0.1.0",
)


class OptimizationCandidate(BaseModel):
    baseline_price: float = Field(gt=0)
    predicted_ite: float


class OptimizationRequest(BaseModel):
    candidates: list[OptimizationCandidate] = Field(min_length=1)
    budget: float = Field(ge=0)
    discount_pct: float = Field(default=10.0, gt=0, le=100)


@app.get("/")
def home():
    return {
        "project": "EconoCausal",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/causal/ate")
def causal_ate():
    """Estimate ATE from the existing synthetic training split."""
    try:
        train_path = DATA_DIR / "train.csv"

        if not train_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="Training dataset is unavailable.",
            )

        train_df = pd.read_csv(train_path)
        ate = estimate_ate(train_df)

        return {
            "model": "DML baseline",
            "dataset": "synthetic training split",
            "training_rows": int(len(train_df)),
            "estimated_ate": ate,
            "estimated_effect_percentage_points": ate * 100,
            "interpretation": (
                "Model estimate on synthetic data; "
                "not evidence of real-world causal validity."
            ),
        }

    except HTTPException:
        raise
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ATE estimation failed: {exc}",
        ) from exc


@app.get("/api/uplift")
def uplift_evaluation():
    """Return the previously generated uplift evaluation report."""
    if not UPLIFT_REPORT.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Uplift report is unavailable. "
                "Run the existing uplift analysis first."
            ),
        )

    try:
        import json

        with UPLIFT_REPORT.open("r", encoding="utf-8") as file:
            report = json.load(file)

        return report

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read uplift report: {exc}",
        ) from exc


@app.post("/api/optimization")
def optimize_promotions(request: OptimizationRequest):
    """Allocate a budget using supplied predicted treatment effects."""
    try:
        candidates = pd.DataFrame(
            [
                candidate.model_dump()
                for candidate in request.candidates
            ]
        )

        problem = build_optimization_problem(
            candidates=candidates,
            budget=request.budget,
            discount_pct=request.discount_pct,
        )

        result = solve_optimization_problem(problem)

        return {
            "objective": problem.objective_description,
            "constraint": problem.constraint_description,
            "solver_status": result.solver_status,
            "selected_indices": list(result.selected_indices),
            "selected_count": result.selected_count,
            "total_cost": result.total_cost,
            "budget": result.budget,
            "unused_budget": result.unused_budget,
            "discount_pct": result.discount_pct,
            "expected_incremental_purchases": (
                result.expected_incremental_purchases
            ),
            "interpretation": (
                "Expected incremental purchases are model predictions, "
                "not guaranteed or observed purchases. Costs use the "
                "configured discount percentage as a cost proxy."
            ),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {exc}",
        ) from exc
