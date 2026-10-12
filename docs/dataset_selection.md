# EconoCausal — Dataset Selection

## 1. Project Objective

EconoCausal estimates the incremental impact of a marketing intervention on customer conversion and uses treatment-effect estimates to support budget-aware promotional decisions.

## 2. Selected Real-World Dataset

**Name:** Criteo Uplift Prediction Dataset  
**Publisher:** Criteo AI Lab  
**Official source:** https://ailab.criteo.com/criteo-uplift-prediction-dataset/

The dataset was developed for uplift modelling and contains randomized advertising treatment information, user features, visits, and conversions.

### Relevant Columns

- `f0` to `f11`: Anonymized user features
- `treatment`: Treatment assignment indicator
- `conversion`: Binary conversion outcome
- `visit`: Binary visit outcome
- `exposure`: Whether the user was effectively exposed to the advertising treatment

## 3. Intended Use

The real dataset will be used for:

- Treatment and control group analysis
- Conversion-rate comparison
- Uplift modelling and heterogeneous treatment-effect experiments
- Evaluation against suitable targeting baselines
- Causal-model diagnostics, subject to the dataset's design and assumptions

## 4. Existing Synthetic Dataset

The existing synthetic retail dataset will be retained for:

- Controlled DML evaluation against known synthetic effects
- Discount-level experiments
- Budget-constrained promotional allocation
- Testing the optimization pipeline

## 5. Important Limitations

The Criteo dataset represents advertising treatment, not explicit discount amounts. It cannot directly establish the effect of a $10 versus $20 discount.

Its features are anonymized, and its conversion outcome is not the same as realized revenue or profit. Discount optimization will therefore require separate pricing assumptions and appropriate cost or margin inputs.

## 6. Processing Strategy

The initial implementation will use a manageable sample, preserve the treatment and outcome columns, record the sampling method and random seed, and keep evaluation data separate from training data.

The full dataset will not be loaded into memory unnecessarily.

## 7. Research Reference

Diemert, E., Betlei, A., Renaudin, C., and Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling*. AdKDD Workshop.

Dataset source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/