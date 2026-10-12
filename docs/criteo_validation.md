# Criteo Dataset — Initial Validation Report

## 1. Dataset Details

- Source: Criteo Uplift Prediction Dataset
- Sample file: `data/real/processed/criteo_sample.csv`
- Sampling method: Reservoir sampling
- Random seed: 42
- Sample size: 100,000 rows
- Columns: 16

## 2. Schema

- `f0`–`f11`: Anonymized numerical features
- `treatment`: Treatment assignment indicator
- `conversion`: Binary conversion outcome
- `visit`: Binary visit outcome
- `exposure`: Exposure indicator

## 3. Data Quality

- Missing values: 0
- Treatment, conversion, visit and exposure are integer-valued binary indicators in the inspected sample.
- All 12 feature columns are numeric.

## 4. Initial Statistics

| Metric | Count |
|---|---:|
| Treatment group | 85,022 |
| Control group | 14,978 |
| Conversions | 302 |
| Visits | 4,740 |
| Exposures | 3,032 |

Observed conversion rate:
- Control: approximately 0.214%
- Treatment: approximately 0.318%
- Unadjusted difference: approximately 0.104 percentage points

## 5. Modelling Considerations

The treatment and control groups are imbalanced, and conversion is a rare outcome. The raw conversion-rate difference is descriptive and must not be interpreted as a causal effect without further analysis.

Treatment assignment and actual exposure are separate variables. Their roles must be handled according to the dataset's design and the intended estimand.

The anonymized features do not provide explicit discount amounts, revenue or profit. This dataset will be used for uplift analysis, while the existing synthetic retail dataset will support controlled discount and budget-optimization experiments.

## 6. Current Status

Initial sample validation is complete. Before fitting a causal model, inspect treatment assignment, exposure semantics, outcome definitions and overlap. Record the chosen estimand and assumptions before model evaluation.