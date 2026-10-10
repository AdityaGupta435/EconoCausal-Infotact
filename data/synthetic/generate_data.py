
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


SEED = 42
N_CUSTOMERS = 2000
OPPORTUNITIES_PER_CUSTOMER = 5

OUTPUT_PATH = (
    Path(__file__).resolve().parent / "retail_data.csv"
)


def generate_retail_data():
    rng = np.random.default_rng(SEED)

    n = N_CUSTOMERS * OPPORTUNITIES_PER_CUSTOMER

    customer_id = np.repeat(
        np.arange(1, N_CUSTOMERS + 1), OPPORTUNITIES_PER_CUSTOMER
    )
    customer_history = rng.poisson(3, n)
    category = rng.choice(
        ["Electronics", "Fashion", "Home", "Beauty"],
        size=n,
        p=[0.20, 0.35, 0.25, 0.20],
    )
    baseline_price = np.round(
        rng.lognormal(mean=4.0, sigma=0.7, size=n), 2
    )
    seasonality = rng.integers(0, 4, n)

    # Promotion assignment depends on observed customer and product factors.
    assignment_score = (
        -0.5
        + 0.12 * customer_history
        - 0.25 * (category == "Electronics")
        + 0.15 * (seasonality == 3)
        - 0.08 * np.log1p(baseline_price)
    )
    propensity = expit(assignment_score)
    treatment = rng.binomial(1, propensity)

    discount_pct = np.where(
        treatment == 1,
        rng.choice([5, 10, 15, 20], size=n),
        0,
    )


    # Known heterogeneous causal effect of receiving a promotion.
    # The effect is measured in purchase-probability points.
    true_effect = (
        0.04
        + 0.025 * (category == "Fashion")
        + 0.015 * (customer_history >= 4)
    )

    baseline_log_odds = (
        -1.5
        + 0.16 * customer_history
        - 0.0008 * baseline_price
        + 0.20 * (category == "Beauty")
        + 0.12 * (seasonality == 3)
    )

    baseline_probability = expit(baseline_log_odds)

    # Binary treatment has one clearly defined causal effect:
    # promotion changes purchase probability by true_effect.
    purchase_probability = np.clip(
        baseline_probability + treatment * true_effect,
        0.01,
        0.99,
    )

    purchase = rng.binomial(1, purchase_probability)


    revenue = np.where(
        purchase == 1,
        baseline_price * (1 - discount_pct / 100),
        0.0,
    )

    promotion_cost = np.where(
        treatment == 1,
        baseline_price * discount_pct / 100,
        0.0,
    )

    data = pd.DataFrame({
        "customer_id": customer_id,
        "product_id": [
            f"P{value:04d}"
            for value in rng.integers(1, 501, n)
        ],
        "category": category,
        "baseline_price": baseline_price,
        "customer_history": customer_history,
        "seasonality": seasonality,
        "treatment": treatment,
        "discount_pct": discount_pct,
        "purchase": purchase,
        "revenue": np.round(revenue, 2),
        "promotion_cost": np.round(promotion_cost, 2),
        "true_effect": true_effect,
        "baseline_probability": baseline_probability,
        "propensity_score": propensity,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved: {OUTPUT_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Treatment rate: {data['treatment'].mean():.1%}")
    print(f"Purchase rate: {data['purchase'].mean():.1%}")
    print(
        "Average known treatment effect: "
        f"{data['true_effect'].mean():.2%}"
    )


if __name__ == "__main__":
    generate_retail_data()
