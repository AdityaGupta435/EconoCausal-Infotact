"""Centralized configuration and environment settings for EconoCausal."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base filesystem directories
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent

# Load environment variables (.env in backend or root)
if (BACKEND_DIR / ".env").exists():
    load_dotenv(dotenv_path=BACKEND_DIR / ".env")
elif (ROOT_DIR / ".env").exists():
    load_dotenv(dotenv_path=ROOT_DIR / ".env")
else:
    load_dotenv()


# Base filesystem directories
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SPLIT_DIR = SYNTHETIC_DATA_DIR / "splits"
REAL_DATA_DIR = DATA_DIR / "real"
PROCESSED_REAL_DATA_DIR = REAL_DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Standard artifact paths
UPLIFT_REPORT = SYNTHETIC_DATA_DIR / "uplift_evaluation.json"
UPLIFT_EVALUATION_PATH = UPLIFT_REPORT
DOWHY_AUDIT_REPORT = SYNTHETIC_DATA_DIR / "dowhy_audit_report.json"
OPTIMIZATION_REPORT = SYNTHETIC_DATA_DIR / "optimization_report.json"
CAUSAL_ENGINE_REPORT = SYNTHETIC_DATA_DIR / "causal_engine_report.json"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
MODEL_EVALUATION_REPORT = MODELS_DIR / "evaluation_report.json"
CAUSAL_MODEL_ARTIFACT = MODEL_ARTIFACTS_DIR / "causal_dml_model.joblib"
UPLIFT_MODEL_ARTIFACT = MODEL_ARTIFACTS_DIR / "uplift_model.joblib"
UPLIFT_SCORED_CUSTOMERS_PATH = SYNTHETIC_DATA_DIR / "customer_uplift_scores.csv"
UPLIFT_DECILE_REPORT_PATH = SYNTHETIC_DATA_DIR / "uplift_deciles.csv"
PRICING_REPORT = SYNTHETIC_DATA_DIR / "pricing_evaluation.json"
PRICING_MODEL_ARTIFACT = MODEL_ARTIFACTS_DIR / "pricing_demand_model.joblib"
TRANSACTION_PRICING_RECOMMENDATIONS_PATH = SYNTHETIC_DATA_DIR / "transaction_pricing_recommendations.csv"
CUSTOMER_PRICING_RECOMMENDATIONS_PATH = SYNTHETIC_DATA_DIR / "customer_pricing_recommendations.csv"
PROMOTION_ALLOCATION_REPORT = SYNTHETIC_DATA_DIR / "promotion_allocation_report.json"
PROMOTION_CUSTOMER_ALLOCATIONS_PATH = SYNTHETIC_DATA_DIR / "promotion_customer_allocations.csv"
PROMOTION_SCENARIO_COMPARISON_PATH = SYNTHETIC_DATA_DIR / "promotion_scenario_comparison.json"

# Application runtime settings
APP_ENV = os.getenv("ECONOCAUSAL_ENV", os.getenv("APP_ENV", "development"))
APP_DEBUG = os.getenv("ECONOCAUSAL_DEBUG", "false").lower() in ("1", "true", "yes")
API_TITLE = "EconoCausal API"
API_DESCRIPTION = "Causal inference for dynamic pricing"
API_VERSION = "0.1.0"

# MongoDB Atlas Settings
MONGODB_URI = os.getenv("MONGODB_URI", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "econocausal")
MONGODB_MAX_POOL_SIZE = int(os.getenv("MONGODB_MAX_POOL_SIZE", "50"))
MONGODB_MIN_POOL_SIZE = int(os.getenv("MONGODB_MIN_POOL_SIZE", "5"))
MONGODB_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000"))
MONGODB_CONNECT_TIMEOUT_MS = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000"))

# Authentication & Security
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "econocausal_default_development_secret_key_at_least_32_chars")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRES_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "30"))

# Operational thresholds & constraints
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))
DEFAULT_DISCOUNT_PCT = float(os.getenv("DEFAULT_DISCOUNT_PCT", "10.0"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
DEFAULT_MIN_PRICE = float(os.getenv("DEFAULT_MIN_PRICE", "5.0"))
DEFAULT_MAX_PRICE = float(os.getenv("DEFAULT_MAX_PRICE", "500.0"))
DEFAULT_MAX_PRICE_CHANGE_PCT = float(os.getenv("DEFAULT_MAX_PRICE_CHANGE_PCT", "0.20"))
DEFAULT_PROMOTION_BUDGET = float(os.getenv("DEFAULT_PROMOTION_BUDGET", "2500.0"))
DEFAULT_ALLOWED_DISCOUNTS = (5.0, 10.0, 15.0, 20.0)

# CORS Configuration
_raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:8000")
CORS_ORIGINS = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]


