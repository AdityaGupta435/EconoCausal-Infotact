
# EconoCausal — Causal DAG

## Research Question

What is the causal effect of offering a discount on the
probability that a customer purchases a product?

## Variables

- T: Discount or promotion assignment (treatment)
- Y: Purchase outcome
- X1: Customer purchase history before treatment
- X2: Product category
- X3: Baseline price before treatment
- X4: Seasonality at the time of the shopping opportunity

## Proposed Causal Relationships

- T -> Y: A discount may change purchase probability.
- X1 -> T: Purchase history may influence promotion assignment.
- X1 -> Y: Purchase history may influence purchase probability.
- X2 -> T: Product category may influence promotion assignment.
- X2 -> Y: Product category may influence purchase probability.
- X3 -> T: Baseline price may influence promotion assignment.
- X3 -> Y: Baseline price may influence purchase probability.
- X4 -> T: Seasonality may influence promotion assignment.
- X4 -> Y: Seasonality may influence purchase probability.

## Identification Assumptions

1. Relevant pre-treatment confounders are measured adequately.
2. Treatment overlap: comparable observations exist with and
   without the promotion.
3. Treatment and outcome definitions are consistent.
4. No post-treatment variables are incorrectly treated as
   ordinary pre-treatment confounders.

## Limitations

These edges are hypotheses, not proven facts. The graph must be
revised if the actual promotion assignment process differs.
Unmeasured confounding may remain.
