# Day 10: DoWhy Causal Audit

## Objective
Audit the synthetic retail treatment-effect estimate using DoWhy,
compare it with the known synthetic ground truth, and evaluate
robustness using refutation tests.

## Dataset and Method
- Dataset: Synthetic retail promotion benchmark
- Rows: 6,000
- Treatment: `treatment`
- Outcome: `purchase`
- Covariates: `baseline_price`, `customer_history`, `seasonality`, `category`
- Identification approach: Backdoor adjustment under the unconfoundedness assumption

The backdoor identification approach relies on the assumption that
the relevant confounders are adequately controlled. The analysis
does not independently prove that this assumption holds.

## Results
- DoWhy linear-regression ATE: 0.079692
- Scikit-learn regression-adjustment ATE: 0.079692
- Known synthetic ATE: 0.054113
- Absolute estimation error: 0.025579

The estimated ATE is higher than the known synthetic ATE. Agreement
between the two regression implementations is a consistency check,
not independent evidence that the causal estimate is unbiased.

## Refutation Tests

### Random Common Cause
- Original estimate: 0.079692
- Refuted estimate: 0.079658
- p-value: 0.88

The estimate changed only slightly after adding a random common
cause. This supports stability under this particular refutation
procedure but does not prove that the causal assumptions are valid.

### Placebo Treatment
- Original estimate: 0.079692
- Placebo estimate: approximately 0
- p-value: 0.0

The placebo treatment produced an effect close to zero, consistent
with the intended placebo check. The p-value is specific to the
refutation procedure and should not be interpreted as proof of causality.

## Warning Audit
The generated `dowhy_audit_report.json` records 500 warnings:

| Refutation | Total warnings | Pandas4Warning | SingularMatrixWarning |
|---|---:|---:|---:|
| Random common cause | 200 | 200 | 0 |
| Placebo treatment | 300 | 200 | 100 |
| Total | 500 | 400 | 100 |

- `Pandas4Warning`: Warnings concerning the deprecated `copy` keyword
  in the pandas 3 Copy-on-Write context.
- `SingularMatrixWarning`: Warnings indicating a rank-deficient
  design matrix during the refutation procedure.

These warnings are recorded in the report and have not been eliminated
by this documentation update. Their impact on the analysis should be
investigated separately.

## Limitations
- The data is synthetic; results do not establish real-world pricing effects.
- The estimated ATE differs from the known synthetic ground truth.
- Refutation tests assess robustness under specific checks; they do not
  prove causal identification or eliminate all confounding.
- Rows repeat customers, so row-level uncertainty may understate
  dependence between observations.
- Binary treatment estimates promotion versus no promotion; they do not
  estimate separate causal effects for every discount percentage.
- Random common cause and placebo refutations are sensitivity checks,
  not proof that confounding has been eliminated.
- The recorded warning counts indicate that warning investigation remains
  part of the audit.

## Conclusion
The DoWhy identification, estimation, and refutation procedures completed.
The estimate is reasonably stable under the tested random-common-cause
refutation, and the placebo estimate is approximately zero. However,
the ATE differs from the known synthetic ground truth, and the report
contains pandas deprecation and singular-matrix warnings.

These results are diagnostic evidence from a synthetic experiment,
not proof of causal identification or real-world effectiveness.
