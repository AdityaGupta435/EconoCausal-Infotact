
# EconoCausal — Dataset Schema

## 1. Unit of Observation

One row represents one customer-product interaction
during a specific shopping opportunity.

## 2. Proposed Columns

| Column | Data Type | Description |
|---|---|---|
| customer_id | string | Anonymous customer identifier |
| product_id | string | Product identifier |
| date | date | Shopping opportunity date |
| category | string | Product category |
| baseline_price | float | Price before the promotion |
| discount_pct | float | Applied discount percentage |
| treatment | integer | 1 = promotion, 0 = no promotion |
| purchase_history | integer | Previous purchases before this opportunity |
| seasonality | integer | Encoded seasonal indicator |
| purchase | integer | 1 = purchased, 0 = not purchased |
| revenue | float | Revenue from the interaction |
| promotion_cost | float | Cost of providing the promotion |

## 3. Treatment and Outcome

- Treatment (T): `treatment`
- Outcome (Y): `purchase`
- Pre-treatment covariates (X): `baseline_price`, `category`,
  `purchase_history`, `seasonality`, and other justified variables.

## 4. Data Quality Rules

- Customer and product identifiers must not be missing.
- `treatment` and `purchase` must contain only 0 or 1.
- `discount_pct` must be between 0 and 100.
- Prices and promotion costs must be non-negative.
- Missing values and duplicate observations must be checked.
- Post-treatment variables must not be used as ordinary confounders.

## 5. Synthetic Data Ground Truth

The synthetic data generator will define a known treatment effect.
The estimated treatment effect will be compared with this known
effect to assess whether the causal model recovers the signal.

## 6. Limitations

Synthetic results demonstrate the methodology, not real-world
business performance. Real causal estimates require appropriate
data, overlap, justified assumptions, and careful validation.
