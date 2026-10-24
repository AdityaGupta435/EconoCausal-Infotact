# Day 14 — Week 2 Checkpoint

## 1. Checkpoint Summary

EconoCausal has completed the documented DoWhy causal audit, uplift evaluation, and SciPy budget-constrained optimization work. This checkpoint consolidates the current results and identifies limitations before API and frontend integration.

## 2. DoWhy Causal Audit

- Analysis dataset: 6,000 rows.
- Treatment: `treatment`.
- Outcome: `purchase`.
- Covariates: `baseline_price`, `customer_history`, `seasonality`, and `category`.
- DoWhy ATE: approximately 0.079692.
- Regression-adjustment ATE: approximately 0.079692.
- Known synthetic ATE: 0.0541125.
- Random-common-cause refutation p-value: 0.88.
- Placebo refutation effect: approximately zero.

### Interpretation and limitations

The DoWhy and regression-adjustment estimates agree numerically on this synthetic dataset. The estimated effect differs from the known synthetic effect. Refutation tests do not prove causal identification. Repeated customer rows, the binary promotion-versus-control treatment, and the unconfoundedness assumption remain important limitations.

The audit recorded 500 warnings: 400 `Pandas4Warning` and 100 `SingularMatrixWarning`. These warnings should be considered when reviewing the analysis environment and diagnostic results.

## 3. Uplift Evaluation

- Training rows: 6,000.
- Test rows: 2,000.
- ITE MAE: 0.02906.
- ITE RMSE: 0.03536.
- Observed Qini area: 27.78958.
- Mean Qini area across 30 random rankings: 29.05260.
- Difference from the random-ranking mean: -1.26302.
- Top-20% mean true effect: model ranking 0.05526, random targeting 0.05380, oracle 0.07329.

### Interpretation and limitations

The observed Qini area is below the random-ranking mean, so this evaluation does not demonstrate superior observed-outcome uplift ranking. The top-20% ground-truth comparison is a separate synthetic evaluation and does not establish real-world targeting performance. The random comparison is not a formal significance test, and the oracle ranking is not available in practice.

## 4. Budget-Constrained Optimization

- Solver: `scipy.optimize.milp`.
- Candidate test rows: 2,000.
- Hypothetical budget: 1,385.60.
- Selected candidates: 526.
- Reported total cost: 1,385.57.
- Unused budget: 0.03.
- Sum of predicted ITE for selected candidates: 46.9187.
- Recorded solver outcome: Optimal.
- Existing optimization tests: 16 tests passed across the project suite.

### Interpretation and limitations

The objective maximizes the sum of predicted ITE subject to the budget constraint, while excluding non-positive predicted effects. The cost model uses a fixed 10% discount as a cost proxy. The reported predicted ITE sum is not observed purchases, realized revenue, profit, or a guarantee of business impact.

The implementation checks solver success and the presence of a solution, but the returned `solver_status` is hardcoded as `optimal` rather than derived from the solver result. This is an auditability improvement to consider; it has not been established as a current runtime failure.

## 5. Verified Repository State

- Branch: `infotact-initial-only`.
- Last confirmed Git state: clean and synchronized with `origin/infotact-initial-only`.
- Latest known commit: `4a85307` — `docs: improve Day 10 DoWhy causal audit`.
- Latest test run: `python -m pytest -q` — 16 passed.

These checks reflect the latest observed command outputs and should not be interpreted as checks rerun by this report.

## 6. Open Issues and Next Steps

1. Preserve the causal assumptions and limitations in any API responses and dashboard explanations.
2. Do not claim uplift-ranking superiority based on the current Qini comparison.
3. Keep the optimization cost proxy explicit until appropriate margin and offer-cost data are available.
4. Consider mapping the actual SciPy solver status into the optimization result and adding focused tests.
5. Proceed to Day 15 FastAPI endpoints only after the Week 2 checkpoint findings are reviewed.

## 7. Checkpoint Decision

The existing synthetic-data pipeline, causal audit documentation, uplift evaluation, and optimization implementation provide a documented baseline for the next phase. The project does not yet establish real-world causal performance or superior uplift targeting. Those limitations must remain visible during API and dashboard integration.

Day 14 report prepared; Git checkpoint/commit remains pending until the report is reviewed.
