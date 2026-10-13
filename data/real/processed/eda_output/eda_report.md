# Criteo Sample — Exploratory Data Analysis

## Dataset overview
- Rows: 100,000
- Columns: 16
- Duplicate rows: 79
- Missing cells: 0

## Treatment and outcome rates

               rate
conversion  0.00302
visit       0.04740
exposure    0.03032

## Outcome summary by treatment assignment

          conversion                 visit                 exposure                
               count      mean  sum  count      mean   sum    count      mean   sum
treatment                                                                          
0              14978  0.002136   32  14978  0.039925   598    14978  0.000000     0
1              85022  0.003176  270  85022  0.048717  4142    85022  0.035661  3032

## Data-quality notes
- Treatment-group comparisons are descriptive, not proof of causality.
- Conversion is a rare outcome; uncertainty must be considered.
- IQR outlier counts are diagnostic flags, not automatic deletion rules.
- Anonymized features cannot be interpreted as named business variables.
- Exposure is reported separately and is not included in the feature matrix.

## Generated outputs
- `conversion_by_treatment.png`
- `data_quality.json`
- `feature_correlation.csv`
- `feature_correlation.png`
- `feature_distributions.png`
- `feature_means_by_treatment.csv`
- `feature_summary.csv`
- `numeric_summary.csv`
- `outcomes_by_treatment.csv`
- `overall_outcome_rates.csv`
- `treatment_counts.csv`
- `treatment_counts.png`
