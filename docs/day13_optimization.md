# Day 13: SciPy Budget-Constrained Promotion Optimization

## Objective

Implement a budget-constrained promotion allocator using SciPy's
mixed-integer linear optimization solver (`scipy.optimize.milp`).

## Method

- Train the existing EconML LinearDML model on the synthetic training split.
- Predict treatment effects for the held-out test candidates.
- Assign a fixed 10% discount for optimization cost accounting.
- Model each candidate's promotion decision as a binary variable.
- Maximize the sum of predicted individual treatment effects.
- Enforce the total promotion-cost budget as a hard constraint.
- Do not select candidates with non-positive predicted treatment effects.

## Evaluation Results

| Metric | Result |
|---|---:|
| Training rows | 6,000 |
| Test candidates | 2,000 |
| Fixed discount | 10% |
| Budget | 1,385.60 |
| Selected promotions | 526 |
| Total estimated cost | 1,385.57 |
| Unused budget | 0.03 |
| Sum of predicted treatment effects | 46.9187 |
| Solver status | Optimal |
| Automated tests | 16 passed |

## Budget Assumption

The budget is set to 10% of the hypothetical total cost of applying a
10% discount to all test candidates. This is a project evaluation
assumption, not a budget observed from a real business.

## Interpretation and Limitations

The solver finds an allocation that maximizes the sum of the model's
predicted treatment effects under the specified budget and binary
selection constraints.

The value 46.9187 is the sum of predicted individual treatment effects,
interpreted as expected incremental purchases only under the model's
causal interpretation and assumptions. It is not a count of observed
incremental purchases and is not a real-world guarantee.

The evaluation uses synthetic retail data. The model estimates a
promotion-versus-control effect; it does not estimate a separate causal
effect for each discount percentage. The fixed 10% discount is used
for cost accounting, not to claim a discount dose-response relationship.

## Files

- `backend/optimization.py`: optimization formulation and SciPy MILP solver.
- `backend/run_optimization.py`: model fitting, candidate scoring, solving,
  and report generation.
- `tests/test_optimization.py`: optimization formulation tests.
- `tests/test_optimization_solver.py`: solver and constraint tests.
- `data/synthetic/optimization_report.json`: evaluation summary.
- `data/synthetic/optimization_selected_candidates.csv`: selected candidates.

## Validation

Command: `python -m pytest -q`

Result: 16 passed.
