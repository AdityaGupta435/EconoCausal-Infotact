
# EconoCausal — Project Requirements

## 1. Problem Statement

Retail businesses use discounts to increase customer purchases.
However, not every customer responds positively to a discount.
EconoCausal estimates the causal effect of discounts on purchase
behavior and helps allocate a limited promotional budget.

## 2. Project Objectives

- Analyze retail and promotional data.
- Estimate heterogeneous treatment effects using Double Machine Learning.
- Identify potential confounders using a causal graph.
- Evaluate model performance and causal assumptions.
- Recommend promotional allocation under a fixed budget.
- Expose analysis through a REST API and interactive dashboard.

## 3. Proposed Variables

- Treatment (T): Whether a discount or promotion was applied.
- Outcome (Y): Whether a purchase occurred.
- Covariates (X): Customer history, product category, baseline price,
  seasonality and other relevant pre-treatment variables.
- Treatment Effect: Estimated change in purchase probability due to
  the promotion.

## 4. Proposed Technology Stack

- Python, Pandas and NumPy
- Scikit-learn and EconML
- DoWhy for causal graph and assumption checks
- SciPy for budget optimization
- FastAPI for REST endpoints
- React and Vite for the dashboard
- Plotly for interactive visualizations
- Pytest for automated tests

## 5. Expected Deliverables

- Reproducible dataset preparation pipeline
- Exploratory data analysis
- Causal graph and assumption documentation
- Double Machine Learning model and evaluation
- Budget allocation and what-if analysis
- REST API and dashboard
- Tests, final report, presentation and demo

## 6. Data and Evaluation Principles

- Clearly distinguish synthetic data from real observations.
- Avoid data leakage from post-treatment variables.
- Document assumptions and limitations of causal estimates.
- Compare estimated effects with known effects when using
  synthetic data with a defined ground truth.
- Do not present demo metrics as validated business results.
