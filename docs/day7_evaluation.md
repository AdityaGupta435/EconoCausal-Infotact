
# Day 7 — Causal Model Evaluation Checkpoint

## Objective

Compare the existing manual Double Machine Learning (DML)
baseline with the EconML LinearDML implementation.

All reported results below use synthetic data with known
treatment effects.

## Evaluation Results

| Model | Estimated ATE | Ground-Truth ATE | Absolute Error |
|---|---:|---:|---:|
| Manual DML | 0.0771 | 0.0538 | 0.0233 |
| Customer-grouped manual DML | 0.0703 | 0.0538 | 0.0165 |
| EconML LinearDML | 0.0763 | 0.0538 | 0.0225 |

## EconML Individual Treatment Effect Evaluation

- Test observations: 2,000
- ITE MAE: 0.0291
- ITE RMSE: 0.0354
- Encoded features: 6
- Treatment: Promotion assignment (binary)
- Outcome: Purchase (binary)

## Interpretation

The customer-grouped manual DML run produced the lowest
reported ATE absolute error.

However, this is not a controlled head-to-head comparison.

The customer-grouped evaluation uses 8,000 training
observations and a customer-disjoint split.

The standard manual DML and EconML evaluations use the
existing 6,000-row training split.

Therefore, these results should not be used to declare
one estimator universally superior.

## Identified Limitations

1. The standard train/test split is not customer-grouped.
2. Repeated customer observations may create dependence.
3. LinearDML estimates a linear treatment-effect function
   over the encoded features.
4. All current quantitative results are from synthetic data.
5. Known synthetic effects are used for evaluation only,
   never as model inputs.
6. ATE and ITE errors do not establish real-world causal
   identification.
7. Valid uncertainty estimates require appropriate
   inference assumptions and additional diagnostics.

## Day 7 Checkpoint

- Manual DML baseline executed successfully.
- Customer-grouped baseline executed successfully.
- EconML held-out test evaluation executed successfully.
- Evaluation metrics recorded and compared.
- Fair customer-grouped model comparison remains necessary.
- Real-world causal analysis remains separate from the
  synthetic benchmark.

## Next Steps

- Validate the estimators under a common evaluation protocol.
- Investigate treatment-effect heterogeneity by segment.
- Perform propensity overlap and causal robustness checks.
