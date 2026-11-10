"""Comprehensive test suite for Phase 6: Dynamic Pricing & Price Elasticity Engine."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from backend.config import (
    CUSTOMER_PRICING_RECOMMENDATIONS_PATH,
    DEFAULT_MAX_PRICE,
    DEFAULT_MAX_PRICE_CHANGE_PCT,
    DEFAULT_MIN_PRICE,
    PRICING_MODEL_ARTIFACT,
    PRICING_REPORT,
    TRANSACTION_PRICING_RECOMMENDATIONS_PATH,
)
from backend.dynamic_pricing import (
    DynamicPricingEngine,
    PriceDemandModel,
    PriceElasticityEstimator,
    calculate_arc_elasticity,
    generate_transaction_recommendations,
    aggregate_customer_recommendations,
)


@pytest.fixture
def sample_pricing_data():
    """Generates a controlled synthetic dataset for pricing tests."""
    np.random.seed(42)
    n = 200
    cust_ids = np.repeat(np.arange(1, 41), 5)  # 40 customers, 5 transactions each
    cat = np.random.choice(["Electronics", "Fashion", "Home", "Beauty"], size=n)
    base_price = np.random.uniform(20.0, 100.0, size=n)
    discount = np.random.choice([0, 5, 10, 15, 20], size=n)
    treatment = (discount > 0).astype(int)
    history = np.random.randint(1, 10, size=n)
    seasonality = np.random.randint(0, 4, size=n)

    # Ground truth demand probability: slightly negative price effect
    p_true = np.clip(0.35 - 0.001 * base_price + 0.05 * treatment, 0.05, 0.95)
    purchase = np.random.binomial(1, p_true)

    df = pd.DataFrame({
        "customer_id": cust_ids,
        "product_id": [f"P{i:04d}" for i in range(n)],
        "category": cat,
        "baseline_price": base_price,
        "discount_pct": discount,
        "treatment": treatment,
        "customer_history": history,
        "seasonality": seasonality,
        "purchase": purchase,
        "revenue": np.where(purchase == 1, base_price * (1 - discount / 100.0), 0.0),
    })
    return df


def test_calculate_arc_elasticity():
    # Price increases from 100 to 110 (+10%), demand drops from 0.40 to 0.36 (-10%)
    e = calculate_arc_elasticity(p0=100.0, q0=0.40, p1=110.0, q1=0.36)
    assert np.isclose(e, -1.105, atol=0.05)

    # Identical prices
    assert calculate_arc_elasticity(100.0, 0.40, 100.0, 0.40) == 0.0

    # Zero demand
    assert calculate_arc_elasticity(100.0, 0.0, 120.0, 0.0) == 0.0


def test_price_elasticity_estimators(sample_pricing_data):
    df = sample_pricing_data

    # Observational OLS
    obs = PriceElasticityEstimator.estimate_observational(df)
    assert obs.model_type == "observational_ols"
    assert not obs.is_causally_identified
    assert np.isfinite(obs.elasticity)
    assert obs.ci_lower <= obs.ci_upper

    # Confounder-Adjusted OLS
    adj = PriceElasticityEstimator.estimate_adjusted(df)
    assert adj.model_type == "confounder_adjusted_ols"
    assert adj.is_causally_identified
    assert np.isfinite(adj.elasticity)
    assert np.isfinite(adj.standard_error)

    # Category-level estimation
    by_cat = PriceElasticityEstimator.estimate_all_categories(df)
    assert "overall" in by_cat
    assert "by_category" in by_cat
    for cat in ["Electronics", "Fashion", "Home", "Beauty"]:
        assert cat in by_cat["by_category"]
        assert "causal_dml" in by_cat["by_category"][cat]


def test_price_demand_model_fit_and_prediction(sample_pricing_data):
    df = sample_pricing_data
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    probs = model.predict_purchase_probability(df)
    assert len(probs) == len(df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Evaluate at higher price -> demand should not wildly extrapolate
    higher_price = df["baseline_price"] * 1.5
    probs_higher = model.predict_purchase_probability(df, candidate_price=higher_price)
    assert len(probs_higher) == len(df)
    assert np.all((probs_higher >= 0.0) & (probs_higher <= 1.0))

    # Point elasticity
    pe = model.compute_point_elasticity(df)
    assert len(pe) == len(df)
    assert np.all(np.isfinite(pe))


def test_dynamic_pricing_engine_constraints_and_bounds(sample_pricing_data):
    df = sample_pricing_data
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(
        demand_model=model,
        support_data=df,
        min_price=10.0,
        max_price=200.0,
        max_price_change_pct=0.20,
    )

    row = df.iloc[[0]].copy()
    curr_price = float(row["baseline_price"].iloc[0])

    # Evaluate candidate prices
    cands = [curr_price * 0.5, curr_price * 0.9, curr_price, curr_price * 1.1, curr_price * 1.5, 5.0, 300.0]
    evals = engine.evaluate_candidates(row, cands)

    for ev in evals:
        if ev.candidate_price < 10.0:
            assert not ev.is_feasible
            assert "below floor" in ev.infeasibility_reason
        elif ev.candidate_price > 200.0:
            assert not ev.is_feasible
            assert "above ceiling" in ev.infeasibility_reason
        elif abs(ev.price_change_pct) > 0.20 + 1e-4:
            assert not ev.is_feasible
            assert "exceeds limit" in ev.infeasibility_reason

    # Recommendation respects constraints
    rec = engine.recommend_price(row)
    assert rec.recommended_price >= 10.0
    assert rec.recommended_price <= 200.0
    assert abs(rec.price_change_pct) <= 0.20 + 1e-4
    assert np.isfinite(rec.recommended_demand)
    assert np.isfinite(rec.recommended_expected_revenue)


def test_cost_margin_profit_handling(sample_pricing_data):
    df = sample_pricing_data
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(demand_model=model, support_data=df)
    row = df.iloc[[0]].copy()

    # 1. Without cost: profit must be None
    rec_no_cost = engine.recommend_price(row, unit_cost=None)
    assert rec_no_cost.expected_profit is None
    assert "Profit not calculated" in rec_no_cost.recommendation_explanation

    # 2. With valid unit cost: profit must be computed
    rec_with_cost = engine.recommend_price(row, unit_cost=15.0)
    assert rec_with_cost.expected_profit is not None
    assert np.isclose(
        rec_with_cost.expected_profit,
        (rec_with_cost.recommended_price - 15.0) * rec_with_cost.recommended_demand,
        atol=0.01,
    )
    assert "Expected unit profit" in rec_with_cost.recommendation_explanation


def test_extrapolation_detection(sample_pricing_data):
    df = sample_pricing_data
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(demand_model=model, support_data=df)
    cat = "Electronics"

    # Extreme price outside historical support
    is_extrap_high, tier_high = engine.check_extrapolation(cat, price=5000.0)
    assert is_extrap_high is True
    assert tier_high == "low"

    # Median price inside historical support
    median_p = engine.support_bounds[cat]["median"]
    is_extrap_med, tier_med = engine.check_extrapolation(cat, price=median_p)
    assert is_extrap_med is False
    assert tier_med == "high"


def test_invalid_price_raises_error(sample_pricing_data):
    df = sample_pricing_data
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(demand_model=model)
    invalid_row = df.iloc[[0]].copy()
    invalid_row["baseline_price"] = -10.0

    with pytest.raises(ValueError, match="Invalid current price"):
        engine.evaluate_candidates(invalid_row, [10.0, 20.0])


def test_batch_and_single_recommendations_consistency(sample_pricing_data):
    df = sample_pricing_data.head(10).copy()
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(demand_model=model, support_data=df)

    # Batch recommendation
    df_batch = engine.recommend_batch(df)
    assert len(df_batch) == 10

    # Single-row recommendation on 1st row
    rec_single = engine.recommend_price(df.iloc[[0]])

    assert np.isclose(df_batch.iloc[0]["recommended_price"], rec_single.recommended_price, atol=0.05)
    assert np.isclose(df_batch.iloc[0]["recommended_demand"], rec_single.recommended_demand, atol=0.05)


def test_customer_level_aggregation_integrity(sample_pricing_data):
    df = sample_pricing_data  # 200 rows, 40 customers (5 transactions each)
    model = PriceDemandModel(max_iter=30, max_leaf_nodes=10, random_state=42)
    model.fit(df, df["purchase"])

    engine = DynamicPricingEngine(demand_model=model, support_data=df)
    trans_recs = generate_transaction_recommendations(df, engine)

    # Uplift segments mock
    mock_uplift = pd.DataFrame({
        "customer_id": np.arange(1, 41),
        "segment": ["Persuadables"] * 20 + ["Sure Things"] * 10 + ["Lost Causes"] * 8 + ["Sleeping Dogs"] * 2,
        "targeting_recommendation": ["Target"] * 20 + ["Do Not Target"] * 20,
    })

    cust_recs = aggregate_customer_recommendations(trans_recs, uplift_scores_df=mock_uplift)

    # Critical check: exactly 40 unique customers, not 200 transaction rows
    assert len(cust_recs) == 40
    assert len(trans_recs) == 200
    assert (cust_recs["transaction_count"] == 5).all()

    # Strategic guidance exists
    assert "strategic_pricing_guidance" in cust_recs.columns
    assert cust_recs["uplift_segment"].nunique() >= 3


def test_phase6_artifacts_and_report_integrity():
    assert PRICING_MODEL_ARTIFACT.exists(), f"Missing model: {PRICING_MODEL_ARTIFACT}"
    assert PRICING_REPORT.exists(), f"Missing report: {PRICING_REPORT}"
    assert TRANSACTION_PRICING_RECOMMENDATIONS_PATH.exists(), f"Missing: {TRANSACTION_PRICING_RECOMMENDATIONS_PATH}"
    assert CUSTOMER_PRICING_RECOMMENDATIONS_PATH.exists(), f"Missing: {CUSTOMER_PRICING_RECOMMENDATIONS_PATH}"

    # Load model artifact
    loaded_model = joblib.load(PRICING_MODEL_ARTIFACT)
    assert hasattr(loaded_model, "predict_purchase_probability")

    # Load and check JSON report
    with open(PRICING_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["execution_summary"]["within_timeout"] is True
    assert report["execution_summary"]["total_elapsed_seconds"] < 180.0
    assert report["train_rows"] == 6000
    assert report["test_rows"] == 2000
    assert report["unique_test_customers"] == 400
    assert "elasticity_results" in report
    assert "causal_dml" in report["elasticity_results"]["overall"]

    # Load transaction CSV
    df_trans = pd.read_csv(TRANSACTION_PRICING_RECOMMENDATIONS_PATH)
    assert len(df_trans) == 2000
    assert "recommended_price" in df_trans.columns
    assert "incremental_revenue" in df_trans.columns

    # Load customer CSV
    df_cust = pd.read_csv(CUSTOMER_PRICING_RECOMMENDATIONS_PATH)
    assert len(df_cust) == 400
    assert (df_cust["transaction_count"] == 5).all()
    assert "strategic_pricing_guidance" in df_cust.columns

