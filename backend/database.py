"""MongoDB Atlas connection and lifecycle management for EconoCausal."""

import logging
import re
from typing import Optional, Dict, Any
from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError

from backend import config

logger = logging.getLogger("econocausal.database")

# Global client and DB references
_mongo_client: Optional[MongoClient] = None
_database_override: Optional[Database] = None


def sanitize_uri(uri: str) -> str:
    """Mask credentials in MongoDB connection string for safe logging."""
    if not uri:
        return "<empty>"
    # Mask password if present: mongodb+srv://user:pass@host -> mongodb+srv://user:***@host
    return re.sub(r":([^/@]+)@", ":***@", uri)


def get_mongo_client() -> Optional[MongoClient]:
    """Obtain or initialize the singleton MongoClient instance."""
    global _mongo_client

    if _database_override is not None and hasattr(_database_override, "client"):
        return _database_override.client

    if _mongo_client is not None:
        return _mongo_client

    uri = config.MONGODB_URI.strip() if config.MONGODB_URI else ""
    if not uri or "<DB_PASSWORD>" in uri or "<EXISTING_CLUSTER_HOST>" in uri:
        logger.warning(
            "MongoDB URI is not configured or contains placeholder credentials. "
            "Database features will operate in disconnected/mock mode."
        )
        return None

    try:
        _mongo_client = MongoClient(
            uri,
            maxPoolSize=config.MONGODB_MAX_POOL_SIZE,
            minPoolSize=config.MONGODB_MIN_POOL_SIZE,
            serverSelectionTimeoutMS=config.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=config.MONGODB_CONNECT_TIMEOUT_MS,
            retryWrites=True,
            appname="EconoCausal",
        )
        logger.info("Initialized MongoDB client with connection pooling.")
        return _mongo_client
    except Exception as exc:
        logger.error("Failed to initialize MongoClient: %s", str(exc))
        _mongo_client = None
        return None


def get_db(db_name: Optional[str] = None) -> Optional[Database]:
    """Retrieve the target database instance."""
    global _database_override
    if _database_override is not None:
        return _database_override

    client = get_mongo_client()
    if client is None:
        return None

    target_name = db_name or config.DATABASE_NAME
    try:
        return client[target_name]
    except Exception as exc:
        logger.error("Failed to access database '%s': %s", target_name, str(exc))
        return None


def set_db_override(db: Optional[Database]) -> None:
    """Inject a mock or custom database instance (primarily for testing)."""
    global _database_override
    _database_override = db


def close_db_connection() -> None:
    """Gracefully close the active MongoClient connection."""
    global _mongo_client, _database_override
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("Closed MongoDB client connection pool.")
        except Exception as exc:
            logger.warning("Error while closing MongoDB connection: %s", str(exc))
        finally:
            _mongo_client = None
    _database_override = None


def check_db_health() -> Dict[str, Any]:
    """Check MongoDB connection health and return structured status."""
    db = get_db()
    if db is None:
        return {
            "status": "disconnected",
            "message": "Database client is not initialized or configured.",
            "database": config.DATABASE_NAME,
        }

    try:
        # Ping the admin database or command
        client = db.client if hasattr(db, "client") else None
        if client:
            client.admin.command("ping")
        else:
            db.command("ping")

        return {
            "status": "healthy",
            "message": "Successfully connected to MongoDB cluster.",
            "database": db.name,
        }
    except (ServerSelectionTimeoutError, ConnectionFailure, PyMongoError) as exc:
        logger.warning("MongoDB ping failed: %s", str(exc))
        return {
            "status": "unhealthy",
            "message": f"Connection check failed: {type(exc).__name__}",
            "database": db.name,
        }
    except Exception as exc:
        logger.warning("Unexpected error during MongoDB ping: %s", str(exc))
        return {
            "status": "error",
            "message": str(exc),
            "database": db.name if db else config.DATABASE_NAME,
        }


def init_db_indexes(db: Database) -> Dict[str, Any]:
    """Ensure all required collection indexes exist with correct constraints."""
    if db is None:
        return {"status": "skipped", "message": "Database unavailable."}

    created = {}
    try:
        # 1. users: unique email and unique username
        users = db["users"]
        users.create_index([("email", ASCENDING)], unique=True, name="idx_users_email_unique")
        users.create_index([("username", ASCENDING)], unique=True, name="idx_users_username_unique")
        created["users"] = ["idx_users_email_unique", "idx_users_username_unique"]

        # 2. datasets: name, created_at
        datasets = db["datasets"]
        datasets.create_index([("name", ASCENDING)], name="idx_datasets_name")
        datasets.create_index([("created_at", ASCENDING)], name="idx_datasets_created_at")
        created["datasets"] = ["idx_datasets_name", "idx_datasets_created_at"]

        # 3. training_runs: run_id (unique), model_type, created_at
        training_runs = db["training_runs"]
        training_runs.create_index([("run_id", ASCENDING)], unique=True, name="idx_training_run_id_unique")
        training_runs.create_index([("model_type", ASCENDING)], name="idx_training_model_type")
        training_runs.create_index([("created_at", ASCENDING)], name="idx_training_created_at")
        created["training_runs"] = ["idx_training_run_id_unique", "idx_training_model_type", "idx_training_created_at"]

        # 4. model_registry: model_name, version, is_active
        model_registry = db["model_registry"]
        model_registry.create_index([("model_name", ASCENDING), ("version", ASCENDING)], unique=True, name="idx_model_name_ver_unique")
        model_registry.create_index([("is_active", ASCENDING)], name="idx_model_is_active")
        created["model_registry"] = ["idx_model_name_ver_unique", "idx_model_is_active"]

        # 5. pricing_recommendations: created_at, strategy
        pricing = db["pricing_recommendations"]
        pricing.create_index([("created_at", ASCENDING)], name="idx_pricing_created_at")
        pricing.create_index([("strategy", ASCENDING)], name="idx_pricing_strategy")
        created["pricing_recommendations"] = ["idx_pricing_created_at", "idx_pricing_strategy"]

        # 6. promotion_campaigns: campaign_name, status
        campaigns = db["promotion_campaigns"]
        campaigns.create_index([("campaign_name", ASCENDING)], name="idx_campaign_name")
        campaigns.create_index([("status", ASCENDING)], name="idx_campaign_status")
        created["promotion_campaigns"] = ["idx_campaign_name", "idx_campaign_status"]

        # 7. optimization_results: campaign_id, created_at
        opt_results = db["optimization_results"]
        opt_results.create_index([("campaign_id", ASCENDING)], name="idx_opt_campaign_id")
        opt_results.create_index([("created_at", ASCENDING)], name="idx_opt_created_at")
        created["optimization_results"] = ["idx_opt_campaign_id", "idx_opt_created_at"]

        # 8. audit_logs: user_id, action, created_at
        audit_logs = db["audit_logs"]
        audit_logs.create_index([("user_id", ASCENDING)], name="idx_audit_user_id")
        audit_logs.create_index([("action", ASCENDING)], name="idx_audit_action")
        audit_logs.create_index([("created_at", ASCENDING)], name="idx_audit_created_at")
        created["audit_logs"] = ["idx_audit_user_id", "idx_audit_action", "idx_audit_created_at"]

        # 9. pipeline_runs: pipeline_run_id (unique), status, created_at
        pipeline_runs = db["pipeline_runs"]
        pipeline_runs.create_index([("pipeline_run_id", ASCENDING)], unique=True, name="idx_pipeline_run_id_unique")
        pipeline_runs.create_index([("status", ASCENDING)], name="idx_pipeline_status")
        pipeline_runs.create_index([("created_at", ASCENDING)], name="idx_pipeline_created_at")
        created["pipeline_runs"] = ["idx_pipeline_run_id_unique", "idx_pipeline_status", "idx_pipeline_created_at"]

        # 10. causal_runs: analysis_id (unique), created_at
        causal_runs = db["causal_runs"]
        causal_runs.create_index([("analysis_id", ASCENDING)], unique=True, name="idx_causal_analysis_id_unique")
        causal_runs.create_index([("created_at", ASCENDING)], name="idx_causal_created_at")
        created["causal_runs"] = ["idx_causal_analysis_id_unique", "idx_causal_created_at"]

        # 11. uplift_runs: analysis_id (unique), created_at
        uplift_runs = db["uplift_runs"]
        uplift_runs.create_index([("analysis_id", ASCENDING)], unique=True, name="idx_uplift_analysis_id_unique")
        uplift_runs.create_index([("created_at", ASCENDING)], name="idx_uplift_created_at")
        created["uplift_runs"] = ["idx_uplift_analysis_id_unique", "idx_uplift_created_at"]

        logger.info("Successfully created/verified all database collection indexes.")
        return {"status": "success", "indexes": created}
    except Exception as exc:
        logger.error("Failed creating database indexes: %s", str(exc))
        return {"status": "error", "error": str(exc)}
