# Day 11: Uplift and Qini Evaluation

## Objective

Evaluate whether the EconML LinearDML model ranks customers by treatment-effect heterogeneity better than random targeting on a held-out synthetic retail dataset.

## Experimental setup

- Model: EconML LinearDML
- Training rows: 6,000
- Held-out test rows: 2,000
- Treatment rows: 774
- Control rows: 1,226
- Evaluation data: synthetic retail benchmark
- Ground-truth treatment effects are used only for evaluation, never as model features.

## Results

| Metric | Result |
|---|---:|
| ITE MAE | 0.02906 |
| ITE RMSE | 0.03536 |
| Observed Qini area | 27.78958 |
| Mean random Qini area, 30 rankings | 29.05260 |
| Qini difference versus random | -1.26302 |
| Top 20% mean true effect, model ranking | 0.05526 |
| Top 20% mean true effect, random targeting | 0.05380 |
| Top 20% mean true effect, oracle ranking | 0.07329 |

## Interpretation

The model's individual treatment-effect estimates have an MAE of approximately 2.91 percentage points on this synthetic test set.

The observed Qini area is lower than the mean from 30 randomized rankings. Therefore, this run does not demonstrate improved observed-outcome uplift ranking over random targeting.

When evaluated against synthetic ground truth, the model-ranked top 20% has a mean true effect of 0.05526, compared with 0.05380 for random targeting. The oracle ranking reaches 0.07329 and is an idealized upper benchmark because it uses known ground-truth effects.

These metrics answer different questions. The observed Qini curve uses realized outcomes, while the top-segment ground-truth metric evaluates ranking against known simulated effects. The latter is not available in real-world data.

## Limitations

- This is a synthetic benchmark, not evidence of real-world causal validity.
- Qini estimates based on observed binary outcomes are noisy.
- The randomized comparison uses 30 random rankings; it is not a formal significance test.
- Oracle ranking is not an implementable production strategy.
- Results do not establish that a particular discount level is optimal.

## Artifacts

- Evaluation module: `backend/uplift_analysis.py`
- Unit tests: `tests/test_uplift_analysis.py`
- Machine-readable report: `data/synthetic/uplift_evaluation.json`

## Conclusion

Day 11 evaluation is implemented. The current model shows reasonable ITE prediction error, but the observed Qini comparison does not establish uplift-ranking superiority. Further model and metric validation is required before using these scores for budget allocation.
