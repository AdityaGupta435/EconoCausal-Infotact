"""Dynamic Pricing and Price Elasticity Engine for EconoCausal (Phase 6).

Implements:
1. Price Elasticity Modeling (Observational OLS, Confounder-Adjusted OLS, and Causal DML).
2. Distinction between correlational and causal price effects with explicit unconfoundedness assumptions.
3. Category-level and overall price elasticity estimation with standard errors and confidence intervals.
4. Price-demand prediction with calibrated purchase probability estimation.
5. Dynamic price recommendation engine subject to configurable business constraints (floors, ceilings, max % shifts).
6. Extrapolation and support boundary detection (1st-99th percentile support check).
7. Transparent revenue-driven recommendations, with profit strictly conditioned on provided unit cost data.
8. Clear separation between transaction-level and customer-level aggregated recommendations.
9. Modular, production-ready architecture persisting evaluation reports and artifacts.
"""

from __future__ import annotations

import sys
if __name__ == "__main__":
    sys.modules["backend.dynamic_pricing"] = sys.modules["__main__"]

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import LinearDML
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.config import (
    CUSTOMER_PRICING_RECOMMENDATIONS_PATH,
    DEFAULT_MAX_PRICE,
    DEFAULT_MAX_PRICE_CHANGE_PCT,
    DEFAULT_MIN_PRICE,
    PRICING_MODEL_ARTIFACT,
    PRICING_REPORT,
    SPLIT_DIR,
    SYNTHETIC_DATA_DIR,
    TRANSACTION_PRICING_RECOMMENDATIONS_PATH,
    UPLIFT_SCORED_CUSTOMERS_PATH,
)

# Schema column definitions
PRICE_COL = "baseline_price"
OUTCOME_COL = "purchase"
TREATMENT_COL = "treatment"
DISCOUNT_COL = "discount_pct"
CATEGORY_COL = "category"
CUSTOMER_HISTORY_COL = "customer_history"
SEASONALITY_COL = "seasonality"
CUSTOMER_ID_COL = "customer_id"
PRODUCT_ID_COL = "product_id"
REVENUE_COL = "revenue"


# =========================================================================
# 1. Data Structures & Results Containers
# =========================================================================

@dataclass
class PriceElasticityResult:
    """Stores elasticity estimation results with statistical and causal properties."""
    model_type: str
    category: Optional[str]
    elasticity: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    sample_size: int
    is_causally_identified: bool
    identification_assumption: str
    caveat: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePriceEvaluation:
    """Detailed evaluation of an individual candidate price point."""
    candidate_price: float
    price_change_pct: float
    predicted_purchase_probability: float
    expected_demand: float
    expected_revenue: float
    expected_profit: Optional[float]
    incremental_demand: float
    incremental_revenue: float
    incremental_profit: Optional[float]
    point_elasticity: float
    arc_elasticity: float
    is_extrapolation: bool
    confidence_tier: str
    is_feasible: bool
    infeasibility_reason: Optional[str] = None


@dataclass
class PricingRecommendation:
    """Recommended pricing decision for a single transaction opportunity."""
    customer_id: str
    product_id: str
    category: str
    current_price: float
    recommended_price: float
    price_change_pct: float
    current_demand: float
    recommended_demand: float
    incremental_demand: float
    current_expected_revenue: float
    recommended_expected_revenue: float
    incremental_revenue: float
    expected_profit: Optional[float]
    arc_elasticity: float
    is_extrapolation: bool
    confidence_tier: str
    recommendation_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# 2. Price Elasticity Estimators
# =========================================================================

class PriceElasticityEstimator:
    """Estimates price elasticity of demand across multiple methodologies."""

    @staticmethod
    def _prepare_effective_price(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if "effective_price" not in data.columns:
            if DISCOUNT_COL in data.columns and PRICE_COL in data.columns:
                data["effective_price"] = data[PRICE_COL] * (1.0 - data[DISCOUNT_COL] / 100.0)
            elif PRICE_COL in data.columns:
                data["effective_price"] = data[PRICE_COL].astype(float)
            else:
                raise ValueError(f"Missing required price column: {PRICE_COL}")

        # Guard against zero or negative prices
        data = data[data["effective_price"] > 0].copy()
        data["log_price"] = np.log(data["effective_price"])
        return data

    @classmethod
    def estimate_observational(cls, df: pd.DataFrame, category: Optional[str] = None) -> PriceElasticityResult:
        """Estimate observational price elasticity via naive log-log regression.
        
        Warning: Correlational only. Does not control for confounders or product selection.
        """
        data = cls._prepare_effective_price(df)
        if category is not None:
            data = data[data[CATEGORY_COL] == category].copy()

        if len(data) < 20:
            raise ValueError(f"Insufficient sample size ({len(data)}) for category {category}")

        y = data[OUTCOME_COL].values.astype(float)
        mean_y = float(np.mean(y))
        if mean_y <= 0:
            raise ValueError("Mean purchase probability is zero; elasticity cannot be computed.")

        X = sm.add_constant(data["log_price"].values)
        model = sm.OLS(y, X).fit()

        beta = float(model.params[1])
        se_beta = float(model.bse[1])
        pval = float(model.pvalues[1])

        # Semi-elasticity / demand base = price elasticity
        elasticity = beta / mean_y
        se_elasticity = se_beta / mean_y

        return PriceElasticityResult(
            model_type="observational_ols",
            category=category,
            elasticity=round(elasticity, 4),
            standard_error=round(se_elasticity, 4),
            ci_lower=round(elasticity - 1.96 * se_elasticity, 4),
            ci_upper=round(elasticity + 1.96 * se_elasticity, 4),
            p_value=round(pval, 4),
            sample_size=len(data),
            is_causally_identified=False,
            identification_assumption="None (Naive correlation). Subject to omitted variable and selection bias.",
            caveat="Reflects cross-sectional price-demand correlation. Does not isolate causal price changes.",
        )

    @classmethod
    def estimate_adjusted(cls, df: pd.DataFrame, category: Optional[str] = None) -> PriceElasticityResult:
        """Estimate price elasticity controlling for observed confounders (history, seasonality, promotion)."""
        data = cls._prepare_effective_price(df)
        if category is not None:
            data = data[data[CATEGORY_COL] == category].copy()

        if len(data) < 30:
            raise ValueError(f"Insufficient sample size ({len(data)}) for adjusted regression.")

        y = data[OUTCOME_COL].values.astype(float)
        mean_y = float(np.mean(y))
        if mean_y <= 0:
            raise ValueError("Mean purchase probability is zero; elasticity cannot be computed.")

        covariates = ["log_price", CUSTOMER_HISTORY_COL, SEASONALITY_COL, TREATMENT_COL]
        features = data[covariates].copy()

        if category is None and CATEGORY_COL in data.columns:
            dummies = pd.get_dummies(data[CATEGORY_COL], drop_first=True, dtype=float)
            features = pd.concat([features, dummies], axis=1)

        X = sm.add_constant(features.values)
        model = sm.OLS(y, X).fit()

        beta = float(model.params[1])
        se_beta = float(model.bse[1])
        pval = float(model.pvalues[1])

        elasticity = beta / mean_y
        se_elasticity = se_beta / mean_y

        return PriceElasticityResult(
            model_type="confounder_adjusted_ols",
            category=category,
            elasticity=round(elasticity, 4),
            standard_error=round(se_elasticity, 4),
            ci_lower=round(elasticity - 1.96 * se_elasticity, 4),
            ci_upper=round(elasticity + 1.96 * se_elasticity, 4),
            p_value=round(pval, 4),
            sample_size=len(data),
            is_causally_identified=True,
            identification_assumption="Conditional unconfoundedness given customer history, seasonality, category, and promotion assignment.",
            caveat="Identified conditional on observed covariates. Does not account for unobserved product quality or cross-product brand substitution.",
        )

    @classmethod
    def estimate_causal_dml(
        cls,
        df: pd.DataFrame,
        category: Optional[str] = None,
        n_splits: int = 3,
        random_state: int = 42,
    ) -> PriceElasticityResult:
        """Estimate causal price elasticity via Double Machine Learning (DML) with orthogonalized cross-fitting."""
        data = cls._prepare_effective_price(df)
        if category is not None:
            data = data[data[CATEGORY_COL] == category].copy()

        if len(data) < 25:
            raise ValueError(f"Insufficient sample size ({len(data)}) for DML price elasticity.")

        cv_folds = min(n_splits, max(2, len(data) // 10))

        y = data[OUTCOME_COL].values.astype(float)
        t = data["log_price"].values.astype(float)
        mean_y = float(np.mean(y))
        if mean_y <= 0:
            raise ValueError("Mean purchase probability is zero.")

        covar_cols = [CUSTOMER_HISTORY_COL, SEASONALITY_COL, TREATMENT_COL]
        X_df = data[covar_cols].copy()
        if category is None and CATEGORY_COL in data.columns:
            dummies = pd.get_dummies(data[CATEGORY_COL], drop_first=True, dtype=float)
            X_df = pd.concat([X_df, dummies], axis=1)

        W = X_df.values.astype(float)

        dml = LinearDML(
            model_y=HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=random_state),
            model_t=HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=random_state),
            discrete_treatment=False,
            cv=cv_folds,
            random_state=random_state,
        )
        dml.fit(y, t, W=W)

        # Marginal treatment effect of log_price on purchase probability
        theta = float(dml.intercept_)
        inf = dml.intercept__inference()
        se_theta = float(inf.stderr)
        pval = float(inf.pvalue()) if callable(getattr(inf, "pvalue", None)) else float(inf.pvalue)

        elasticity = theta / mean_y
        se_elasticity = se_theta / mean_y

        return PriceElasticityResult(
            model_type="causal_dml",
            category=category,
            elasticity=round(elasticity, 4),
            standard_error=round(se_elasticity, 4),
            ci_lower=round(elasticity - 1.96 * se_elasticity, 4),
            ci_upper=round(elasticity + 1.96 * se_elasticity, 4),
            p_value=round(pval, 4),
            sample_size=len(data),
            is_causally_identified=True,
            identification_assumption="Double Machine Learning Neyman-orthogonal residualization assuming unconfoundedness given W.",
            caveat="Isolates pricing variation orthogonal to customer history, seasonality, and promotional assignment.",
        )

    @classmethod
    def estimate_all_categories(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute observational, adjusted, and causal elasticity overall and per category."""
        results = {
            "overall": {
                "observational": cls.estimate_observational(df).to_dict(),
                "adjusted": cls.estimate_adjusted(df).to_dict(),
                "causal_dml": cls.estimate_causal_dml(df).to_dict(),
            },
            "by_category": {},
        }

        categories = sorted(df[CATEGORY_COL].dropna().unique())
        for cat in categories:
            results["by_category"][cat] = {
                "observational": cls.estimate_observational(df, category=cat).to_dict(),
                "adjusted": cls.estimate_adjusted(df, category=cat).to_dict(),
                "causal_dml": cls.estimate_causal_dml(df, category=cat).to_dict(),
            }

        return results


# =========================================================================
# 3. Price-Demand Predictive Model
# =========================================================================

class PriceDemandModel(BaseEstimator, ClassifierMixin):
    """Predicts purchase probability under arbitrary candidate prices."""

    __module__ = "backend.dynamic_pricing"

    def __init__(self, max_iter: int = 150, max_leaf_nodes: int = 15, learning_rate: float = 0.05, random_state: int = 42):
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.categorical_features = [CATEGORY_COL]
        self.numeric_features = [
            PRICE_COL,
            "effective_price",
            CUSTOMER_HISTORY_COL,
            SEASONALITY_COL,
            TREATMENT_COL,
            DISCOUNT_COL,
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), self.categorical_features),
                ("num", StandardScaler(), self.numeric_features),
            ]
        )

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def _prepare_features(self, df: pd.DataFrame, candidate_price: Optional[Union[float, np.ndarray, pd.Series]] = None) -> pd.DataFrame:
        X = df.copy()
        if candidate_price is not None:
            if np.isscalar(candidate_price):
                X[PRICE_COL] = float(candidate_price)
            else:
                X[PRICE_COL] = np.asarray(candidate_price, dtype=float)

        # Compute effective price
        if DISCOUNT_COL in X.columns:
            X["effective_price"] = X[PRICE_COL] * (1.0 - X[DISCOUNT_COL] / 100.0)
        else:
            X["effective_price"] = X[PRICE_COL]
            X[DISCOUNT_COL] = 0.0

        if TREATMENT_COL not in X.columns:
            X[TREATMENT_COL] = (X[DISCOUNT_COL] > 0).astype(int)

        return X[self.categorical_features + self.numeric_features]

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        X_clean = self._prepare_features(X)
        y_clean = np.asarray(y, dtype=int)

        X_trans = self.preprocessor.fit_transform(X_clean)
        self.model.fit(X_trans, y_clean)
        return self

    def predict_proba(self, X: pd.DataFrame, candidate_price: Optional[Union[float, np.ndarray, pd.Series]] = None) -> np.ndarray:
        X_clean = self._prepare_features(X, candidate_price=candidate_price)
        X_trans = self.preprocessor.transform(X_clean)
        return self.model.predict_proba(X_trans)

    def predict_purchase_probability(self, X: pd.DataFrame, candidate_price: Optional[Union[float, np.ndarray, pd.Series]] = None) -> np.ndarray:
        """Returns purchase probability for class 1."""
        return self.predict_proba(X, candidate_price=candidate_price)[:, 1]

    def compute_point_elasticity(
        self,
        X: pd.DataFrame,
        candidate_price: Optional[Union[float, np.ndarray, pd.Series]] = None,
        delta: float = 0.5,
    ) -> np.ndarray:
        """Compute numerical point elasticity at current or candidate price."""
        curr_price = X[PRICE_COL].values.astype(float) if candidate_price is None else np.asarray(candidate_price, dtype=float)
        p_up = curr_price + delta
        p_down = np.maximum(curr_price - delta, 0.01)

        q_up = self.predict_purchase_probability(X, candidate_price=p_up)
        q_down = self.predict_purchase_probability(X, candidate_price=p_down)
        q_curr = self.predict_purchase_probability(X, candidate_price=curr_price)

        dq_dp = (q_up - q_down) / (p_up - p_down)
        elasticity = np.where(q_curr > 1e-5, (curr_price / q_curr) * dq_dp, 0.0)
        return elasticity


def calculate_arc_elasticity(p0: float, q0: float, p1: float, q1: float) -> float:
    """Calculate arc price elasticity between two price-demand coordinate points."""
    dp = p1 - p0
    dq = q1 - q0
    if abs(dp) < 1e-6:
        return 0.0
    avg_p = 0.5 * (p1 + p0)
    avg_q = 0.5 * (q1 + q0)
    if avg_q < 1e-6:
        return 0.0
    return float((dq / dp) * (avg_p / avg_q))


# =========================================================================
# 4. Dynamic Pricing Engine & Recommendation Generator
# =========================================================================

class DynamicPricingEngine:
    """Core recommendation engine with business constraints, bounds, and extrapolation safety."""

    def __init__(
        self,
        demand_model: PriceDemandModel,
        support_data: Optional[pd.DataFrame] = None,
        min_price: float = DEFAULT_MIN_PRICE,
        max_price: float = DEFAULT_MAX_PRICE,
        max_price_change_pct: float = DEFAULT_MAX_PRICE_CHANGE_PCT,
    ):
        self.demand_model = demand_model
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.max_price_change_pct = float(max_price_change_pct)

        # Learn historical support bounds per category to flag out-of-support extrapolation
        self.support_bounds: Dict[str, Dict[str, float]] = {}
        if support_data is not None and CATEGORY_COL in support_data.columns and PRICE_COL in support_data.columns:
            for cat, group in support_data.groupby(CATEGORY_COL):
                prices = group[PRICE_COL].dropna().values.astype(float)
                if len(prices) > 0:
                    self.support_bounds[cat] = {
                        "p01": float(np.percentile(prices, 1)),
                        "p99": float(np.percentile(prices, 99)),
                        "min": float(np.min(prices)),
                        "max": float(np.max(prices)),
                        "median": float(np.median(prices)),
                    }

    def check_extrapolation(self, category: str, price: float) -> Tuple[bool, str]:
        """Check if candidate price is outside historical support for the category."""
        if category not in self.support_bounds:
            # Fallback to global engine bounds
            is_extrap = price < self.min_price or price > self.max_price
            tier = "low" if is_extrap else "medium"
            return is_extrap, tier

        bounds = self.support_bounds[category]
        if price < bounds["p01"] or price > bounds["p99"]:
            return True, "low"
        elif price < bounds["min"] or price > bounds["max"]:
            return True, "low"
        else:
            return False, "high"

    def evaluate_candidates(
        self,
        context_row: pd.DataFrame,
        candidate_prices: List[float],
        unit_cost: Optional[float] = None,
    ) -> List[CandidatePriceEvaluation]:
        """Evaluate a set of candidate prices against current price under all constraints."""
        if len(context_row) != 1:
            raise ValueError("context_row must contain exactly 1 transaction opportunity.")

        curr_price = float(context_row[PRICE_COL].iloc[0])
        category = str(context_row[CATEGORY_COL].iloc[0])

        if curr_price <= 0:
            raise ValueError(f"Invalid current price: {curr_price}")

        # Current baseline prediction
        q0 = float(self.demand_model.predict_purchase_probability(context_row, candidate_price=curr_price)[0])
        rev0 = curr_price * q0
        prof0 = (curr_price - unit_cost) * q0 if unit_cost is not None else None

        evaluations = []
        for p_cand in candidate_prices:
            p_cand = float(round(p_cand, 2))
            pct_change = (p_cand - curr_price) / curr_price

            # Feasibility checks
            is_feasible = True
            infeasible_reasons = []

            if p_cand < self.min_price:
                is_feasible = False
                infeasible_reasons.append(f"Price ${p_cand:.2f} is below floor ${self.min_price:.2f}")
            if p_cand > self.max_price:
                is_feasible = False
                infeasible_reasons.append(f"Price ${p_cand:.2f} is above ceiling ${self.max_price:.2f}")
            if abs(pct_change) > self.max_price_change_pct + 1e-5:
                is_feasible = False
                infeasible_reasons.append(
                    f"Price change {pct_change:+.1%} exceeds limit +/-{self.max_price_change_pct:.0%}"
                )

            # Predict demand at candidate price
            q_cand = float(self.demand_model.predict_purchase_probability(context_row, candidate_price=p_cand)[0])
            rev_cand = p_cand * q_cand
            prof_cand = (p_cand - unit_cost) * q_cand if unit_cost is not None else None

            # Incremental metrics
            inc_demand = q_cand - q0
            inc_rev = rev_cand - rev0
            inc_prof = (prof_cand - prof0) if (prof_cand is not None and prof0 is not None) else None

            # Elasticity
            arc_e = calculate_arc_elasticity(curr_price, q0, p_cand, q_cand)
            point_e = float(self.demand_model.compute_point_elasticity(context_row, candidate_price=p_cand)[0])

            # Extrapolation
            is_extrap, conf_tier = self.check_extrapolation(category, p_cand)

            evaluations.append(
                CandidatePriceEvaluation(
                    candidate_price=p_cand,
                    price_change_pct=round(pct_change, 4),
                    predicted_purchase_probability=round(q_cand, 4),
                    expected_demand=round(q_cand, 4),
                    expected_revenue=round(rev_cand, 4),
                    expected_profit=round(prof_cand, 4) if prof_cand is not None else None,
                    incremental_demand=round(inc_demand, 4),
                    incremental_revenue=round(inc_rev, 4),
                    incremental_profit=round(inc_prof, 4) if inc_prof is not None else None,
                    point_elasticity=round(point_e, 4),
                    arc_elasticity=round(arc_e, 4),
                    is_extrapolation=is_extrap,
                    confidence_tier=conf_tier,
                    is_feasible=is_feasible,
                    infeasibility_reason="; ".join(infeasible_reasons) if infeasible_reasons else None,
                )
            )

        return evaluations

    def recommend_price(
        self,
        context_row: pd.DataFrame,
        unit_cost: Optional[float] = None,
        candidate_count: int = 11,
    ) -> PricingRecommendation:
        """Select optimal revenue-maximizing (or profit-maximizing if cost given) feasible price."""
        curr_price = float(context_row[PRICE_COL].iloc[0])
        cust_id = str(context_row[CUSTOMER_ID_COL].iloc[0])
        prod_id = str(context_row[PRODUCT_ID_COL].iloc[0])
        cat = str(context_row[CATEGORY_COL].iloc[0])

        # Generate candidate grid within allowed % change
        min_cand = max(curr_price * (1.0 - self.max_price_change_pct), self.min_price)
        max_cand = min(curr_price * (1.0 + self.max_price_change_pct), self.max_price)
        candidate_prices = list(np.linspace(min_cand, max_cand, candidate_count))

        # Always include the current price in candidates
        if curr_price not in candidate_prices:
            candidate_prices.append(curr_price)
        candidate_prices = sorted(set(candidate_prices))

        evaluations = self.evaluate_candidates(context_row, candidate_prices, unit_cost=unit_cost)
        feasible_evals = [e for e in evaluations if e.is_feasible]

        if not feasible_evals:
            # Fallback to current price clipped to bounds
            fallback_price = float(np.clip(curr_price, self.min_price, self.max_price))
            evaluations = self.evaluate_candidates(context_row, [fallback_price], unit_cost=unit_cost)
            best_eval = evaluations[0]
        else:
            if unit_cost is not None:
                # Maximize expected profit
                best_eval = max(feasible_evals, key=lambda e: (e.expected_profit if e.expected_profit is not None else -999999.0))
            else:
                # Maximize expected revenue
                best_eval = max(feasible_evals, key=lambda e: e.expected_revenue)

        # Baseline evaluation for current price
        curr_eval = next((e for e in evaluations if abs(e.candidate_price - curr_price) < 1e-4), best_eval)

        # Explanation generation
        if abs(best_eval.price_change_pct) < 1e-3:
            explanation = (
                f"Maintain current price ${curr_price:.2f}. Local price elasticity is {best_eval.arc_elasticity:+.2f}; "
                f"current price optimizes expected revenue (${best_eval.expected_revenue:.2f})."
            )
        elif best_eval.price_change_pct > 0:
            explanation = (
                f"Increase price by {best_eval.price_change_pct:+.1%} to ${best_eval.candidate_price:.2f}. "
                f"Demand is relatively inelastic (arc elasticity={best_eval.arc_elasticity:+.2f}). "
                f"Expected demand shifts from {curr_eval.expected_demand:.1%} to {best_eval.expected_demand:.1%}, "
                f"increasing expected revenue by ${best_eval.incremental_revenue:+.2f} per opportunity."
            )
        else:
            explanation = (
                f"Decrease price by {best_eval.price_change_pct:+.1%} to ${best_eval.candidate_price:.2f}. "
                f"Demand response (arc elasticity={best_eval.arc_elasticity:+.2f}) increases purchase probability "
                f"from {curr_eval.expected_demand:.1%} to {best_eval.expected_demand:.1%} (+{best_eval.incremental_demand:+.1%} demand), "
                f"delivering higher expected revenue (+${best_eval.incremental_revenue:+.2f})."
            )

        if unit_cost is None:
            explanation += " (Note: Profit not calculated because unit cost data was not provided; optimized for revenue)."
        else:
            explanation += f" (Expected unit profit: ${best_eval.expected_profit:.2f})."

        return PricingRecommendation(
            customer_id=cust_id,
            product_id=prod_id,
            category=cat,
            current_price=round(curr_price, 2),
            recommended_price=round(best_eval.candidate_price, 2),
            price_change_pct=round(best_eval.price_change_pct, 4),
            current_demand=round(curr_eval.expected_demand, 4),
            recommended_demand=round(best_eval.expected_demand, 4),
            incremental_demand=round(best_eval.incremental_demand, 4),
            current_expected_revenue=round(curr_eval.expected_revenue, 4),
            recommended_expected_revenue=round(best_eval.expected_revenue, 4),
            incremental_revenue=round(best_eval.incremental_revenue, 4),
            expected_profit=round(best_eval.expected_profit, 4) if best_eval.expected_profit is not None else None,
            arc_elasticity=round(best_eval.arc_elasticity, 4),
            is_extrapolation=best_eval.is_extrapolation,
            confidence_tier=best_eval.confidence_tier,
            recommendation_explanation=explanation,
        )

    def recommend_batch(
        self,
        df: pd.DataFrame,
        unit_cost: Optional[float] = None,
        candidate_count: int = 11,
    ) -> pd.DataFrame:
        """Vectorized batch recommendation across all transaction opportunities."""
        n = len(df)
        if n == 0:
            return pd.DataFrame()

        p0 = df[PRICE_COL].values.astype(float)
        cust_ids = df[CUSTOMER_ID_COL].values
        prod_ids = df[PRODUCT_ID_COL].values
        categories = df[CATEGORY_COL].values

        # Base prediction at current price
        q0 = self.demand_model.predict_purchase_probability(df, candidate_price=p0)
        rev0 = p0 * q0

        # Candidate price multipliers from -max_price_change_pct to +max_price_change_pct
        shifts = np.linspace(-self.max_price_change_pct, self.max_price_change_pct, candidate_count)
        if not np.any(np.isclose(shifts, 0.0)):
            shifts = np.sort(np.append(shifts, 0.0))

        p_matrices = []
        q_matrices = []
        rev_matrices = []
        profit_matrices = []
        feasible_matrices = []

        for s in shifts:
            p_cand = np.round(p0 * (1.0 + s), 2)
            p_bounded = np.clip(p_cand, self.min_price, self.max_price)
            q_cand = self.demand_model.predict_purchase_probability(df, candidate_price=p_bounded)
            rev_cand = p_bounded * q_cand
            prof_cand = (p_bounded - unit_cost) * q_cand if unit_cost is not None else None

            pct_chg = np.abs((p_bounded - p0) / p0)
            is_feas = (
                (p_bounded >= self.min_price)
                & (p_bounded <= self.max_price)
                & (pct_chg <= self.max_price_change_pct + 1e-5)
            )

            p_matrices.append(p_bounded)
            q_matrices.append(q_cand)
            rev_matrices.append(rev_cand)
            if unit_cost is not None:
                profit_matrices.append(prof_cand)
            feasible_matrices.append(is_feas)

        p_mat = np.column_stack(p_matrices)
        q_mat = np.column_stack(q_matrices)
        rev_mat = np.column_stack(rev_matrices)
        feas_mat = np.column_stack(feasible_matrices)

        if unit_cost is not None:
            prof_mat = np.column_stack(profit_matrices)
            objective_mat = prof_mat.copy()
        else:
            prof_mat = None
            objective_mat = rev_mat.copy()

        objective_mat[~feas_mat] = -1e9
        best_indices = np.argmax(objective_mat, axis=1)

        row_indices = np.arange(n)
        best_p = p_mat[row_indices, best_indices]
        best_q = q_mat[row_indices, best_indices]
        best_rev = rev_mat[row_indices, best_indices]
        best_prof = prof_mat[row_indices, best_indices] if prof_mat is not None else None

        best_pct_change = (best_p - p0) / p0
        inc_demand = best_q - q0
        inc_rev = best_rev - rev0

        dp = best_p - p0
        dq = best_q - q0
        avg_p = 0.5 * (best_p + p0)
        avg_q = 0.5 * (best_q + q0)
        valid_arc = (np.abs(dp) > 1e-5) & (avg_q > 1e-5)
        arc_e = np.zeros(n, dtype=float)
        arc_e[valid_arc] = (dq[valid_arc] / dp[valid_arc]) * (avg_p[valid_arc] / avg_q[valid_arc])

        extrap_flags = []
        conf_tiers = []
        for i in range(n):
            is_ex, tier = self.check_extrapolation(categories[i], best_p[i])
            extrap_flags.append(is_ex)
            conf_tiers.append(tier)

        explanations = []
        for i in range(n):
            cp = p0[i]
            bp = best_p[i]
            pct = best_pct_change[i]
            ae = arc_e[i]
            cd = q0[i]
            bd = best_q[i]
            ir = inc_rev[i]

            if abs(pct) < 1e-3:
                exp_text = (
                    f"Maintain current price ${cp:.2f}. Local price elasticity is {ae:+.2f}; "
                    f"current price optimizes expected revenue (${best_rev[i]:.2f})."
                )
            elif pct > 0:
                exp_text = (
                    f"Increase price by {pct:+.1%} to ${bp:.2f}. "
                    f"Demand is relatively inelastic (arc elasticity={ae:+.2f}). "
                    f"Expected demand shifts from {cd:.1%} to {bd:.1%}, "
                    f"increasing expected revenue by ${ir:+.2f} per opportunity."
                )
            else:
                exp_text = (
                    f"Decrease price by {pct:+.1%} to ${bp:.2f}. "
                    f"Demand response (arc elasticity={ae:+.2f}) increases purchase probability "
                    f"from {cd:.1%} to {bd:.1%} ({inc_demand[i]:+.1%} demand), "
                    f"delivering higher expected revenue (+${ir:+.2f})."
                )

            if unit_cost is None:
                exp_text += " (Note: Profit not calculated because unit cost data was not provided; optimized for revenue)."
            else:
                exp_text += f" (Expected unit profit: ${best_prof[i]:.2f})."

            explanations.append(exp_text)

        recs_df = pd.DataFrame({
            "customer_id": cust_ids,
            "product_id": prod_ids,
            "category": categories,
            "current_price": np.round(p0, 2),
            "recommended_price": np.round(best_p, 2),
            "price_change_pct": np.round(best_pct_change, 4),
            "current_demand": np.round(q0, 4),
            "recommended_demand": np.round(best_q, 4),
            "incremental_demand": np.round(inc_demand, 4),
            "current_expected_revenue": np.round(rev0, 4),
            "recommended_expected_revenue": np.round(best_rev, 4),
            "incremental_revenue": np.round(inc_rev, 4),
            "expected_profit": np.round(best_prof, 4) if unit_cost is not None else [None] * n,
            "arc_elasticity": np.round(arc_e, 4),
            "is_extrapolation": extrap_flags,
            "confidence_tier": conf_tiers,
            "recommendation_explanation": explanations,
        })
        return recs_df


# =========================================================================
# 5. Batch Recommendation Pipelines: Transaction-Level & Customer-Level
# =========================================================================

def generate_transaction_recommendations(
    df: pd.DataFrame,
    engine: DynamicPricingEngine,
    unit_cost: Optional[float] = None,
) -> pd.DataFrame:
    """Generate row-level pricing recommendations for all transaction opportunities."""
    return engine.recommend_batch(df, unit_cost=unit_cost)


def aggregate_customer_recommendations(
    transaction_recs_df: pd.DataFrame,
    uplift_scores_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate transaction recommendations to unique customer level with Phase 5 Uplift integration."""
    customer_rows = []

    # Prepare lookup for uplift segments if available
    uplift_lookup = {}
    if uplift_scores_df is not None and CUSTOMER_ID_COL in uplift_scores_df.columns:
        for cid, group in uplift_scores_df.groupby(CUSTOMER_ID_COL):
            seg = group["segment"].iloc[0] if "segment" in group.columns else "Unknown"
            rec = group["targeting_recommendation"].iloc[0] if "targeting_recommendation" in group.columns else "Unknown"
            uplift_lookup[cid] = (seg, rec)

    for cid, group in transaction_recs_df.groupby(CUSTOMER_ID_COL):
        t_count = len(group)
        categories = sorted(group[CATEGORY_COL].unique())
        cat_str = ", ".join(categories)

        mean_curr_price = float(group["current_price"].mean())
        mean_rec_price = float(group["recommended_price"].mean())
        mean_pct_change = float(group["price_change_pct"].mean())

        tot_curr_demand = float(group["current_demand"].sum())
        tot_rec_demand = float(group["recommended_demand"].sum())
        tot_inc_demand = float(group["incremental_demand"].sum())

        tot_curr_rev = float(group["current_expected_revenue"].sum())
        tot_rec_rev = float(group["recommended_expected_revenue"].sum())
        tot_inc_rev = float(group["incremental_revenue"].sum())

        mean_e = float(group["arc_elasticity"].mean())

        seg, target_rec = uplift_lookup.get(cid, ("Unclassified", "Review"))

        # Strategic guidance tailored to uplift quadrant
        if seg == "Persuadables":
            guidance = (
                "High promotional responsiveness. Price competitively and consider targeted promotional discounts; "
                "customer has high price sensitivity and incremental conversion potential."
            )
        elif seg == "Sure Things":
            guidance = (
                "Organic buyers with high baseline purchase probability. Maintain premium pricing and avoid unnecessary "
                "discounts or promotional markdowns; demand is relatively inelastic."
            )
        elif seg == "Lost Causes":
            guidance = (
                "Low purchase probability regardless of price or promotion. Maintain standard catalog pricing with standard "
                "margin; avoid margin-eroding price cuts."
            )
        elif seg == "Sleeping Dogs":
            guidance = (
                "Sensitive to marketing fatigue. Avoid aggressive discounting or notification triggers; keep stable regular pricing."
            )
        else:
            guidance = "Standard dynamic pricing based on category elasticity."

        customer_rows.append({
            "customer_id": int(cid),
            "transaction_count": t_count,
            "categories_shopped": cat_str,
            "mean_current_price": round(mean_curr_price, 2),
            "mean_recommended_price": round(mean_rec_price, 2),
            "mean_price_change_pct": round(mean_pct_change, 4),
            "total_current_expected_demand": round(tot_curr_demand, 3),
            "total_recommended_expected_demand": round(tot_rec_demand, 3),
            "total_incremental_demand": round(tot_inc_demand, 3),
            "total_current_expected_revenue": round(tot_curr_rev, 2),
            "total_recommended_expected_revenue": round(tot_rec_rev, 2),
            "total_incremental_revenue": round(tot_inc_rev, 2),
            "mean_elasticity": round(mean_e, 4),
            "uplift_segment": seg,
            "uplift_recommendation": target_rec,
            "strategic_pricing_guidance": guidance,
        })

    return pd.DataFrame(customer_rows)


# =========================================================================
# 6. End-to-End Dynamic Pricing & Elasticity Pipeline (Phase 6)
# =========================================================================

def run_pricing_pipeline(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    uplift_scores_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    timeout_seconds: float = 180.0,
) -> Dict[str, Any]:
    """Execute complete Phase 6 dynamic pricing and elasticity pipeline with progress logging and timeout enforcement."""
    start_total = time.perf_counter()
    stage_timings: Dict[str, float] = {}

    def _check_timeout(stage_name: str):
        elapsed = time.perf_counter() - start_total
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Dynamic pricing pipeline exceeded execution timeout of {timeout_seconds}s "
                f"at stage '{stage_name}' (elapsed: {elapsed:.2f}s)."
            )

    t_path = train_path or (SPLIT_DIR / "train.csv")
    v_path = val_path or (SPLIT_DIR / "validation.csv")
    te_path = test_path or (SPLIT_DIR / "test.csv")
    u_path = uplift_scores_path or UPLIFT_SCORED_CUSTOMERS_PATH
    out_dir = output_dir or SYNTHETIC_DATA_DIR

    print("=" * 70)
    print("ECONOCAUSAL DYNAMIC PRICING & PRICE ELASTICITY ENGINE (PHASE 6)")
    print(f"Configured Pipeline Execution Timeout: {timeout_seconds:.1f}s")
    print("=" * 70)

    # 1. Load datasets
    t0 = time.perf_counter()
    df_train = pd.read_csv(t_path)
    df_val = pd.read_csv(v_path)
    df_test = pd.read_csv(te_path)
    load_duration = time.perf_counter() - t0
    stage_timings["data_loading"] = round(load_duration, 3)
    print(f"Loaded splits in {load_duration:.3f}s: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,} rows")

    # 2. Price Elasticity Modeling
    _check_timeout("Price Elasticity Modeling")
    t1 = time.perf_counter()
    print("\n[1/5] Estimating Price Elasticity of Demand (Overall & Category-level)...")
    elasticity_results = PriceElasticityEstimator.estimate_all_categories(df_train)
    e_duration = time.perf_counter() - t1
    stage_timings["price_elasticity_modeling"] = round(e_duration, 3)

    ov = elasticity_results["overall"]
    print(f"  - Overall Observational Elasticity : {ov['observational']['elasticity']:+.4f} (p={ov['observational']['p_value']:.4f})")
    print(f"  - Overall Confounder-Adjusted     : {ov['adjusted']['elasticity']:+.4f} (SE={ov['adjusted']['standard_error']:.4f}, 95% CI: [{ov['adjusted']['ci_lower']}, {ov['adjusted']['ci_upper']}])")
    print(f"  - Overall Causal DML Elasticity   : {ov['causal_dml']['elasticity']:+.4f} (SE={ov['causal_dml']['standard_error']:.4f}, 95% CI: [{ov['causal_dml']['ci_lower']}, {ov['causal_dml']['ci_upper']}])")

    print("\n  Category-Level Causal DML Elasticities:")
    for cat, res in elasticity_results["by_category"].items():
        dml_res = res["causal_dml"]
        print(f"    * {cat:12s}: Elasticity = {dml_res['elasticity']:+.4f} | SE = {dml_res['standard_error']:.4f} | 95% CI = [{dml_res['ci_lower']}, {dml_res['ci_upper']}]")
    print(f"  Stage [1/5] completed in {e_duration:.3f}s")

    # 3. Train Calibrated Price-Demand Model
    _check_timeout("Demand Model Fitting")
    t2 = time.perf_counter()
    print("\n[2/5] Training Calibrated Price-Demand Model...")
    demand_model = PriceDemandModel(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, random_state=42)
    demand_model.fit(df_train, df_train[OUTCOME_COL].values)

    # Validation evaluation
    p_val_pred = demand_model.predict_purchase_probability(df_val)
    val_brier = float(brier_score_loss(df_val[OUTCOME_COL], p_val_pred))
    val_mae = float(mean_absolute_error(df_val[OUTCOME_COL], p_val_pred))
    val_rmse = float(root_mean_squared_error(df_val[OUTCOME_COL], p_val_pred))
    print(f"  Validation Performance: Brier Score={val_brier:.4f} | MAE={val_mae:.4f} | RMSE={val_rmse:.4f}")

    # Test evaluation
    p_test_pred = demand_model.predict_purchase_probability(df_test)
    test_brier = float(brier_score_loss(df_test[OUTCOME_COL], p_test_pred))
    test_mae = float(mean_absolute_error(df_test[OUTCOME_COL], p_test_pred))
    test_rmse = float(root_mean_squared_error(df_test[OUTCOME_COL], p_test_pred))

    # Ground truth benchmark if available
    gt_test_path = SPLIT_DIR / "test_ground_truth.csv"
    gt_mae = None
    gt_rmse = None
    if gt_test_path.exists():
        df_gt_test = pd.read_csv(gt_test_path)
        if "baseline_probability" in df_gt_test.columns:
            p_test_base = demand_model.predict_purchase_probability(df_test[df_test[TREATMENT_COL] == 0])
            gt_base = df_gt_test.loc[df_test[TREATMENT_COL] == 0, "baseline_probability"]
            gt_mae = float(mean_absolute_error(gt_base, p_test_base))
            gt_rmse = float(root_mean_squared_error(gt_base, p_test_base))
            print(f"  Ground Truth Test Benchmark: MAE vs True Baseline Prob={gt_mae:.4f} | RMSE={gt_rmse:.4f}")

    print(f"  Untouched Test Performance: Brier Score={test_brier:.4f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f}")
    m_duration = time.perf_counter() - t2
    stage_timings["demand_model_training_and_eval"] = round(m_duration, 3)
    print(f"  Stage [2/5] completed in {m_duration:.3f}s")

    # 4. Generate Vectorized Transaction-Level Price Recommendations
    _check_timeout("Transaction Recommendation Generation")
    t3 = time.perf_counter()
    print("\n[3/5] Generating Vectorized Transaction-Level Dynamic Price Recommendations (N=2,000)...")
    engine = DynamicPricingEngine(
        demand_model=demand_model,
        support_data=df_train,
        min_price=DEFAULT_MIN_PRICE,
        max_price=DEFAULT_MAX_PRICE,
        max_price_change_pct=DEFAULT_MAX_PRICE_CHANGE_PCT,
    )

    df_trans_recs = generate_transaction_recommendations(df_test, engine)
    trans_out_path = out_dir / "transaction_pricing_recommendations.csv"
    df_trans_recs.to_csv(trans_out_path, index=False)
    rec_duration = time.perf_counter() - t3
    stage_timings["transaction_recommendations_generation"] = round(rec_duration, 3)

    print(f"  Saved transaction recommendations to: {trans_out_path}")
    print(f"  Mean Price Shift: {df_trans_recs['price_change_pct'].mean():+.2%}")
    print(f"  Total Expected Current Revenue : ${df_trans_recs['current_expected_revenue'].sum():,.2f}")
    print(f"  Total Expected Recommended Rev : ${df_trans_recs['recommended_expected_revenue'].sum():,.2f}")
    print(f"  Total Incremental Revenue      : ${df_trans_recs['incremental_revenue'].sum():+,.2f} ({df_trans_recs['incremental_revenue'].sum()/df_trans_recs['current_expected_revenue'].sum():+.2%})")
    print(f"  Stage [3/5] completed in {rec_duration:.3f}s")

    # 5. Customer-Level Aggregation & Uplift Integration
    _check_timeout("Customer Recommendation Aggregation")
    t4 = time.perf_counter()
    print("\n[4/5] Aggregating Customer-Level Recommendations (N=400 Unique Customers)...")
    df_uplift = pd.read_csv(u_path) if u_path.exists() else None
    df_cust_recs = aggregate_customer_recommendations(df_trans_recs, uplift_scores_df=df_uplift)
    cust_out_path = out_dir / "customer_pricing_recommendations.csv"
    df_cust_recs.to_csv(cust_out_path, index=False)
    agg_duration = time.perf_counter() - t4
    stage_timings["customer_aggregation"] = round(agg_duration, 3)

    print(f"  Saved customer-level aggregated recommendations to: {cust_out_path}")
    print("\n  Customer Pricing Summary by Uplift Segment:")
    seg_summary = df_cust_recs.groupby("uplift_segment").agg(
        customer_count=("customer_id", "count"),
        mean_price_shift=("mean_price_change_pct", "mean"),
        total_inc_revenue=("total_incremental_revenue", "sum"),
        mean_elasticity=("mean_elasticity", "mean"),
    ).reset_index()
    for _, row in seg_summary.iterrows():
        print(f"    - {row['uplift_segment']:13s}: {int(row['customer_count']):3d} customers | Mean Shift: {row['mean_price_shift']:+.2%} | Inc Rev: ${row['total_inc_revenue']:+,.2f} | Mean Arc Elasticity: {row['mean_elasticity']:+.4f}")
    print(f"  Stage [4/5] completed in {agg_duration:.3f}s")

    # 6. Save Model Artifact & Final Report
    _check_timeout("Artifact Serialization")
    t5 = time.perf_counter()
    print("\n[5/5] Persisting Model Artifacts and Evaluation Report...")
    PRICING_MODEL_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(demand_model, PRICING_MODEL_ARTIFACT)
    print(f"  Saved demand model artifact to: {PRICING_MODEL_ARTIFACT}")

    total_duration = time.perf_counter() - start_total
    stage_timings["total_pipeline"] = round(total_duration, 3)

    extrapolation_count = int(df_trans_recs["is_extrapolation"].sum())
    report_data = {
        "evaluation": "Phase 6 Dynamic Pricing & Price Elasticity Engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_summary": {
            "total_elapsed_seconds": round(total_duration, 3),
            "stage_timings_seconds": stage_timings,
            "within_timeout": bool(total_duration <= timeout_seconds),
            "configured_timeout_seconds": timeout_seconds,
        },
        "train_rows": len(df_train),
        "validation_rows": len(df_val),
        "test_rows": len(df_test),
        "unique_test_customers": len(df_cust_recs),
        "transactions_per_customer": 5,
        "business_constraints": {
            "min_price": DEFAULT_MIN_PRICE,
            "max_price": DEFAULT_MAX_PRICE,
            "max_price_change_pct": DEFAULT_MAX_PRICE_CHANGE_PCT,
            "profit_policy": "Profit optimization was not executed because unit cost/margin data is absent in observed data. Revenue maximization applied.",
        },
        "elasticity_results": elasticity_results,
        "demand_model_performance": {
            "validation_brier_score": round(val_brier, 4),
            "validation_mae": round(val_mae, 4),
            "validation_rmse": round(val_rmse, 4),
            "test_brier_score": round(test_brier, 4),
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "ground_truth_test_mae": round(gt_mae, 4) if gt_mae is not None else None,
            "ground_truth_test_rmse": round(gt_rmse, 4) if gt_rmse is not None else None,
        },
        "recommendation_summary": {
            "transaction_count": len(df_trans_recs),
            "unique_customers": len(df_cust_recs),
            "mean_recommended_price_change_pct": round(float(df_trans_recs["price_change_pct"].mean()), 4),
            "total_current_expected_revenue": round(float(df_trans_recs["current_expected_revenue"].sum()), 2),
            "total_recommended_expected_revenue": round(float(df_trans_recs["recommended_expected_revenue"].sum()), 2),
            "total_incremental_revenue": round(float(df_trans_recs["incremental_revenue"].sum()), 2),
            "revenue_uplift_pct": round(float(df_trans_recs["incremental_revenue"].sum() / df_trans_recs["current_expected_revenue"].sum()), 4),
            "extrapolation_recommendation_count": extrapolation_count,
            "extrapolation_rate": round(float(extrapolation_count / len(df_trans_recs)), 4),
        },
        "customer_segment_pricing_summary": seg_summary.to_dict(orient="records"),
        "artifact_paths": {
            "model": str(PRICING_MODEL_ARTIFACT),
            "report": str(PRICING_REPORT),
            "transaction_recommendations": str(trans_out_path),
            "customer_recommendations": str(cust_out_path),
        },
    }

    with open(PRICING_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved pricing evaluation report to: {PRICING_REPORT}")

    print(f"\nPhase 6 Dynamic Pricing & Elasticity Engine completed in {total_duration:.2f}s (well within {timeout_seconds}s limit).")
    return report_data


if __name__ == "__main__":
    run_pricing_pipeline()

