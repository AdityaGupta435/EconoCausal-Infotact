# Criteo Dataset — EDA Findings

## 1. Dataset

- Source: Criteo Uplift Prediction Dataset
- Sample: 100,000 rows
- Features: `f0` through `f11`
- Treatment: `treatment`
- Primary outcome: `conversion`
- Secondary outcome: `visit`
- Exposure indicator: `exposure`

## 2. Data Quality

- Missing cells: 0
- Exact duplicate rows reported by pandas: 79
- Rows involved in exact duplicate groups: 158
- No rows have been removed at this stage.

Duplicate rows will be retained until the source data design and duplicate patterns have been sufficiently examined. Exact duplicates do not automatically imply invalid observations.

## 3. Treatment and Outcomes

| Metric | Control | Treatment |
|---|---:|---:|
| Rows | 14,978 | 85,022 |
| Conversions | 32 | 270 |
| Conversion rate | 0.214% | 0.318% |
| Visits | 598 | 4,142 |
| Visit rate | 3.993% | 4.872% |
| Exposures | 0 | 3,032 |

The observed conversion-rate difference is approximately 0.104 percentage points in favour of the treatment-assigned group. This is a descriptive comparison, not a causal-effect estimate.

## 4. Feature Distributions

The 12 anonymized features have no missing values in the inspected sample. Their distributions and ranges differ. Some features have concentrated quartiles, while others have wider ranges.

Feature scaling, outlier handling and feature selection will not be decided from ranges alone. These decisions must be appropriate for the estimator and evaluated using training-only preprocessing where applicable.

## 5. Exposure Consideration

The exposure indicator is zero for all control rows and positive for some treatment-assigned rows. Exposure is kept separate from the initial feature matrix because it may occur after treatment assignment.

The primary analysis will first define the estimand around treatment assignment. Any exposure-based analysis requires a separate justification.

## 6. Preprocessing Decisions

- Do not automatically delete duplicate rows.
- Do not remove outliers solely because they exceed an IQR threshold.
- Preserve the original sample.
- Keep treatment, outcomes and exposure distinct from the 12-feature matrix.
- Use a reproducible train/test strategy for later model evaluation.
- Report uncertainty because conversion is rare.
- Do not interpret anonymized features as named customer or business attributes.

## 7. Current Status

Initial EDA and descriptive findings are documented. Data-cleaning decisions and causal-model assumptions still require validation. Day 4 is not complete until preprocessing checks, output review and the final GitHub push are complete.