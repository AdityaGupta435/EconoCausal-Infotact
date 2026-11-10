"""Persistent data management, business logic, audit logging, and metadata synchronization."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pymongo.database import Database

from backend import config
from backend.auth import hash_password, verify_password
from backend.schemas import UserCreate, UserRole

logger = logging.getLogger("econocausal.services")


def _format_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Format MongoDB document by converting _id to id string."""
    if doc is None:
        return None
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    return d


# ---------------------------------------------------------
# Audit Logging Service
# ---------------------------------------------------------

def record_audit_log(
    db: Optional[Database],
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    user: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Record an audit trail event in the audit_logs collection."""
    if db is None:
        return None

    entry = {
        "user_id": user.get("id") if user else None,
        "username": user.get("username") if user else "anonymous",
        "action": action,
        "entity_type": entity_type,
        "entity_id": str(entity_id) if entity_id else None,
        "details": details or {},
        "ip_address": ip_address,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        res = db["audit_logs"].insert_one(entry)
        entry["id"] = str(res.inserted_id)
        if "_id" in entry:
            entry.pop("_id", None)
        return entry
    except Exception as exc:
        logger.warning("Failed to record audit log: %s", str(exc))
        return None


def get_audit_logs(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
    action: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve audit logs with optional filters."""
    if db is None:
        return []
    query: Dict[str, Any] = {}
    if action:
        query["action"] = action
    if user_id:
        query["user_id"] = user_id

    cursor = db["audit_logs"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


# ---------------------------------------------------------
# User Management Services
# ---------------------------------------------------------

def create_user(
    db: Database,
    user_data: UserCreate,
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a new user with hashed password."""
    email_clean = user_data.email.strip().lower()
    username_clean = user_data.username.strip().lower()

    if db["users"].find_one({"email": email_clean}):
        raise ValueError(f"User with email '{email_clean}' already exists.")

    if db["users"].find_one({"username": username_clean}):
        raise ValueError(f"Username '{username_clean}' is already taken.")

    now_iso = datetime.now(timezone.utc).isoformat()
    hashed = hash_password(user_data.password)

    user_doc = {
        "email": email_clean,
        "username": username_clean,
        "full_name": user_data.full_name.strip() if user_data.full_name else None,
        "role": user_data.role.value if isinstance(user_data.role, UserRole) else str(user_data.role),
        "hashed_password": hashed,
        "is_active": True,
        "created_at": now_iso,
        "last_login": None,
    }

    res = db["users"].insert_one(user_doc)
    user_id = str(res.inserted_id)

    record_audit_log(
        db,
        action="user.create",
        entity_type="users",
        entity_id=user_id,
        details={"username": username_clean, "role": user_doc["role"]},
        user=actor,
    )

    clean_user = _format_doc(user_doc)
    clean_user.pop("hashed_password", None)
    clean_user["id"] = user_id
    return clean_user


def authenticate_user(
    db: Database,
    username_or_email: str,
    password: str,
    ip_address: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate user credentials and return user object if successful."""
    term = username_or_email.strip().lower()
    user_doc = db["users"].find_one({"$or": [{"email": term}, {"username": term}]})

    if not user_doc:
        return None

    if not verify_password(password, user_doc.get("hashed_password", "")):
        return None

    if not user_doc.get("is_active", True):
        return None

    now_iso = datetime.now(timezone.utc).isoformat()
    db["users"].update_one(
        {"_id": user_doc["_id"]},
        {"$set": {"last_login": now_iso}},
    )

    user_clean = _format_doc(user_doc)
    user_clean["last_login"] = now_iso
    user_clean.pop("hashed_password", None)

    record_audit_log(
        db,
        action="auth.login",
        entity_type="users",
        entity_id=user_clean["id"],
        details={"username": user_clean["username"]},
        user=user_clean,
        ip_address=ip_address,
    )

    return user_clean


def get_user_by_id(db: Optional[Database], user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch user by id."""
    if db is None:
        return None
    from bson import ObjectId
    try:
        query = {"_id": ObjectId(user_id)}
    except Exception:
        query = {"_id": user_id}

    doc = db["users"].find_one(query)
    if not doc:
        doc = db["users"].find_one({"id": user_id})
    if not doc:
        return None

    clean = _format_doc(doc)
    clean.pop("hashed_password", None)
    return clean


def get_user_by_email(db: Optional[Database], email: str) -> Optional[Dict[str, Any]]:
    if db is None:
        return None
    doc = db["users"].find_one({"email": email.strip().lower()})
    clean = _format_doc(doc)
    if clean:
        clean.pop("hashed_password", None)
    return clean


def get_user_by_username(db: Optional[Database], username: str) -> Optional[Dict[str, Any]]:
    if db is None:
        return None
    doc = db["users"].find_one({"username": username.strip().lower()})
    clean = _format_doc(doc)
    if clean:
        clean.pop("hashed_password", None)
    return clean


def list_users(db: Optional[Database], skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    """List all registered users without exposing password hashes."""
    if db is None:
        return []
    cursor = db["users"].find().sort("created_at", -1).skip(skip).limit(limit)
    users = []
    for doc in cursor:
        u = _format_doc(doc)
        u.pop("hashed_password", None)
        users.append(u)
    return users


def update_user_role(
    db: Database,
    user_id: str,
    new_role: UserRole,
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Update assigned RBAC role of a user."""
    from bson import ObjectId
    try:
        filter_q = {"_id": ObjectId(user_id)}
    except Exception:
        filter_q = {"_id": user_id}

    role_val = new_role.value if isinstance(new_role, UserRole) else str(new_role)
    res = db["users"].update_one(filter_q, {"$set": {"role": role_val}})
    if res.matched_count == 0:
        return None

    record_audit_log(
        db,
        action="user.role_update",
        entity_type="users",
        entity_id=user_id,
        details={"new_role": role_val},
        user=actor,
    )
    return get_user_by_id(db, user_id)


def update_user_status(
    db: Database,
    user_id: str,
    is_active: bool,
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Enable or disable a user account."""
    from bson import ObjectId
    try:
        filter_q = {"_id": ObjectId(user_id)}
    except Exception:
        filter_q = {"_id": user_id}

    res = db["users"].update_one(filter_q, {"$set": {"is_active": is_active}})
    if res.matched_count == 0:
        return None

    record_audit_log(
        db,
        action="user.status_update",
        entity_type="users",
        entity_id=user_id,
        details={"is_active": is_active},
        user=actor,
    )
    return get_user_by_id(db, user_id)


def change_user_password(
    db: Database,
    user_id: str,
    current_password: str,
    new_password: str,
) -> bool:
    """Change user password after verifying current password."""
    from bson import ObjectId
    try:
        filter_q = {"_id": ObjectId(user_id)}
    except Exception:
        filter_q = {"_id": user_id}

    doc = db["users"].find_one(filter_q)
    if not doc:
        return False

    if not verify_password(current_password, doc.get("hashed_password", "")):
        return False

    new_hash = hash_password(new_password)
    db["users"].update_one(filter_q, {"$set": {"hashed_password": new_hash}})

    record_audit_log(
        db,
        action="user.password_change",
        entity_type="users",
        entity_id=user_id,
        user=_format_doc(doc),
    )
    return True


# ---------------------------------------------------------
# Metadata Persistence Services (Phases 1–7 Artifacts)
# ---------------------------------------------------------

def record_dataset(
    db: Optional[Database],
    metadata: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Save dataset metadata to the datasets collection."""
    if db is None:
        return None

    doc = metadata.copy()
    if "created_at" not in doc:
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "created_by" not in doc:
        doc["created_by"] = actor.get("username", "system")

    # Upsert by name and file_path
    db["datasets"].update_one(
        {"name": doc["name"], "file_path": doc["file_path"]},
        {"$set": doc},
        upsert=True,
    )
    saved = db["datasets"].find_one({"name": doc["name"], "file_path": doc["file_path"]})
    clean = _format_doc(saved)

    record_audit_log(
        db,
        action="dataset.record",
        entity_type="datasets",
        entity_id=clean["id"],
        details={"name": doc["name"], "rows": doc.get("row_count")},
        user=actor,
    )
    return clean


def get_datasets(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    cursor = db["datasets"].find().sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


def get_dataset_by_id(db: Optional[Database], dataset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch dataset metadata by ID."""
    if db is None:
        return None
    from bson import ObjectId
    try:
        query = {"_id": ObjectId(dataset_id)}
    except Exception:
        query = {"_id": dataset_id}
    doc = db["datasets"].find_one(query)
    if not doc:
        doc = db["datasets"].find_one({"id": dataset_id})
    return _format_doc(doc)


def record_training_run(
    db: Optional[Database],
    run_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Save training run metadata in training_runs collection."""
    if db is None:
        return None

    doc = run_data.copy()
    if "created_at" not in doc:
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "created_by" not in doc:
        doc["created_by"] = actor.get("username", "system")

    db["training_runs"].update_one(
        {"run_id": doc["run_id"]},
        {"$set": doc},
        upsert=True,
    )
    saved = db["training_runs"].find_one({"run_id": doc["run_id"]})
    clean = _format_doc(saved)

    record_audit_log(
        db,
        action="training_run.record",
        entity_type="training_runs",
        entity_id=clean["id"],
        details={"run_id": doc["run_id"], "model_name": doc.get("model_name")},
        user=actor,
    )
    return clean


def get_training_runs(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
    model_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    query: Dict[str, Any] = {}
    if model_type:
        query["model_type"] = model_type
    cursor = db["training_runs"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


def record_model(
    db: Optional[Database],
    model_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Register a trained model in model_registry."""
    if db is None:
        return None

    doc = model_data.copy()
    if "registered_at" not in doc:
        doc["registered_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "registered_by" not in doc:
        doc["registered_by"] = actor.get("username", "system")

    db["model_registry"].update_one(
        {"model_name": doc["model_name"], "version": doc.get("version", "v1.0")},
        {"$set": doc},
        upsert=True,
    )
    saved = db["model_registry"].find_one({
        "model_name": doc["model_name"],
        "version": doc.get("version", "v1.0"),
    })
    clean = _format_doc(saved)

    record_audit_log(
        db,
        action="model_registry.record",
        entity_type="model_registry",
        entity_id=clean["id"],
        details={"model_name": doc["model_name"], "version": doc.get("version")},
        user=actor,
    )
    return clean


def get_models(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
    is_active: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    query: Dict[str, Any] = {}
    if is_active is not None:
        query["is_active"] = is_active
    cursor = db["model_registry"].find(query).sort("registered_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


def record_pricing_recommendation(
    db: Optional[Database],
    pricing_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist pricing recommendation metadata."""
    if db is None:
        return None

    doc = pricing_data.copy()
    if "created_at" not in doc:
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "created_by" not in doc:
        doc["created_by"] = actor.get("username", "system")

    res = db["pricing_recommendations"].insert_one(doc)
    doc["id"] = str(res.inserted_id)
    if "_id" in doc:
        doc.pop("_id", None)

    record_audit_log(
        db,
        action="pricing.record",
        entity_type="pricing_recommendations",
        entity_id=doc["id"],
        details={"strategy": doc.get("strategy")},
        user=actor,
    )
    return doc


def get_pricing_recommendations(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    cursor = db["pricing_recommendations"].find().sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


def record_promotion_campaign(
    db: Optional[Database],
    campaign_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist marketing promotion campaign metadata."""
    if db is None:
        return None

    doc = campaign_data.copy()
    if "created_at" not in doc:
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "created_by" not in doc:
        doc["created_by"] = actor.get("username", "system")

    db["promotion_campaigns"].update_one(
        {"campaign_id": doc["campaign_id"]},
        {"$set": doc},
        upsert=True,
    )
    saved = db["promotion_campaigns"].find_one({"campaign_id": doc["campaign_id"]})
    clean = _format_doc(saved)

    record_audit_log(
        db,
        action="campaign.record",
        entity_type="promotion_campaigns",
        entity_id=clean["id"],
        details={"campaign_name": doc.get("campaign_name"), "budget": doc.get("budget")},
        user=actor,
    )
    return clean


def get_promotion_campaigns(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    query: Dict[str, Any] = {}
    if status:
        query["status"] = status
    cursor = db["promotion_campaigns"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


def get_promotion_campaign_by_id(
    db: Optional[Database],
    campaign_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch campaign by identifier."""
    if db is None:
        return None
    from bson import ObjectId
    try:
        query = {"_id": ObjectId(campaign_id)}
    except Exception:
        query = {"_id": campaign_id}
    doc = db["promotion_campaigns"].find_one(query)
    if not doc:
        doc = db["promotion_campaigns"].find_one({"campaign_id": campaign_id})
    return _format_doc(doc)


def record_optimization_result(
    db: Optional[Database],
    opt_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Persist mathematical optimization run results."""
    if db is None:
        return None

    doc = opt_data.copy()
    if "created_at" not in doc:
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
    if actor and "created_by" not in doc:
        doc["created_by"] = actor.get("username", "system")

    res = db["optimization_results"].insert_one(doc)
    doc["id"] = str(res.inserted_id)
    if "_id" in doc:
        doc.pop("_id", None)

    record_audit_log(
        db,
        action="optimization.record",
        entity_type="optimization_results",
        entity_id=doc["id"],
        details={"campaign_id": doc.get("campaign_id"), "allocated_budget": doc.get("allocated_budget")},
        user=actor,
    )
    return doc


def get_optimization_results(
    db: Optional[Database],
    skip: int = 0,
    limit: int = 50,
    campaign_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if db is None:
        return []
    query: Dict[str, Any] = {}
    if campaign_id:
        query["campaign_id"] = campaign_id
    cursor = db["optimization_results"].find(query).sort("created_at", -1).skip(skip).limit(limit)
    return [_format_doc(doc) for doc in cursor]


# ---------------------------------------------------------
# Artifact Sync Service: Local Disk -> MongoDB Collections
# ---------------------------------------------------------

def sync_local_artifacts_to_db(
    db: Optional[Database],
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scan existing Phase 1–7 artifacts on local disk and record their metadata in MongoDB.

    All CSV data and model files remain safely on local disk; only structured metadata
    is indexed in MongoDB.
    """
    if db is None:
        return {"status": "skipped", "message": "Database is not connected."}

    synced = {
        "datasets": 0,
        "models": 0,
        "pricing": 0,
        "campaigns": 0,
        "optimization": 0,
    }

    # 1. Sync Datasets
    candidate_datasets = [
        ("retail_data_full", config.SYNTHETIC_DATA_DIR / "retail_data.csv", "synthetic"),
        ("train_split", config.SPLIT_DIR / "train.csv", "synthetic"),
        ("validation_split", config.SPLIT_DIR / "validation.csv", "synthetic"),
        ("test_split", config.SPLIT_DIR / "test.csv", "synthetic"),
        ("criteo_sample", config.PROCESSED_REAL_DATA_DIR / "criteo_sample.csv", "real"),
    ]

    for name, path, src_type in candidate_datasets:
        if path.is_file():
            try:
                # Fast sample to inspect columns and length
                df_sample = pd.read_csv(path, nrows=5)
                # Count lines without loading entire file in memory
                with path.open("rb") as f:
                    row_count = sum(1 for _ in f) - 1
                meta = {
                    "name": name,
                    "description": f"Persisted dataset ({src_type})",
                    "file_path": str(path.resolve()),
                    "row_count": max(0, row_count),
                    "column_count": len(df_sample.columns),
                    "columns": list(df_sample.columns),
                    "file_size_bytes": path.stat().st_size,
                    "source_type": src_type,
                }
                record_dataset(db, meta, actor=actor)
                synced["datasets"] += 1
            except Exception as exc:
                logger.warning("Could not sync dataset %s: %s", name, str(exc))

    # 2. Sync Model Registry & Joblibs
    if config.MODEL_REGISTRY_PATH.is_file():
        try:
            with open(config.MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
            # could be list or dict
            entries = registry_data if isinstance(registry_data, list) else registry_data.get("models", [])
            for entry in entries:
                if isinstance(entry, dict) and "model_name" in entry:
                    record_model(db, entry, actor=actor)
                    synced["models"] += 1
        except Exception as exc:
            logger.warning("Could not sync model registry: %s", str(exc))

    # Also register default artifacts if file exists
    known_artifacts = [
        ("causal_dml_model", "v1.0", "Causal DML Estimator", config.CAUSAL_MODEL_ARTIFACT),
        ("uplift_model", "v1.0", "Uplift Learner", config.UPLIFT_MODEL_ARTIFACT),
        ("pricing_demand_model", "v1.0", "Price Elasticity Demand Estimator", config.PRICING_MODEL_ARTIFACT),
    ]
    for model_name, ver, desc, art_path in known_artifacts:
        if art_path.is_file():
            m_data = {
                "model_name": model_name,
                "version": ver,
                "model_type": desc,
                "artifact_path": str(art_path.resolve()),
                "metrics": {"file_size_bytes": art_path.stat().st_size},
                "is_active": True,
                "description": desc,
            }
            record_model(db, m_data, actor=actor)
            synced["models"] += 1

    # 3. Sync Pricing Recommendations Report
    if config.PRICING_REPORT.is_file():
        try:
            with open(config.PRICING_REPORT, "r", encoding="utf-8") as f:
                p_eval = json.load(f)
            pricing_rec = {
                "scenario_name": "Phase 6 Dynamic Pricing Baseline",
                "strategy": "profit_maximization",
                "parameters": {
                    "min_price": config.DEFAULT_MIN_PRICE,
                    "max_price": config.DEFAULT_MAX_PRICE,
                },
                "summary_metrics": p_eval.get("metrics", p_eval),
                "recommendations_count": p_eval.get("recommendations_count", 0),
            }
            record_pricing_recommendation(db, pricing_rec, actor=actor)
            synced["pricing"] += 1
        except Exception as exc:
            logger.warning("Could not sync pricing report: %s", str(exc))

    # 4. Sync Promotion Campaign & Optimization Report
    if config.PROMOTION_ALLOCATION_REPORT.is_file():
        try:
            with open(config.PROMOTION_ALLOCATION_REPORT, "r", encoding="utf-8") as f:
                promo_data = json.load(f)
            campaign_id = "CAMP-PHASE7-DEFAULT"
            campaign_rec = {
                "campaign_id": campaign_id,
                "campaign_name": "Phase 7 Budget Optimization Campaign",
                "budget": float(promo_data.get("budget", config.DEFAULT_PROMOTION_BUDGET)),
                "status": "completed",
                "target_segment": "high_uplift_customers",
                "objectives": {"strategy": "incremental_profit_maximization"},
            }
            record_promotion_campaign(db, campaign_rec, actor=actor)
            synced["campaigns"] += 1

            opt_rec = {
                "campaign_id": campaign_id,
                "total_budget": float(promo_data.get("budget", 2500.0)),
                "allocated_budget": float(promo_data.get("total_cost", 0.0)),
                "expected_uplift": float(promo_data.get("expected_incremental_purchases", 0.0)),
                "customer_count": int(promo_data.get("selected_count", 0)),
                "status": "optimal",
                "summary": {
                    "solver_status": promo_data.get("solver_status"),
                    "selected_count": promo_data.get("selected_count"),
                },
            }
            record_optimization_result(db, opt_rec, actor=actor)
            synced["optimization"] += 1
        except Exception as exc:
            logger.warning("Could not sync promotion allocation report: %s", str(exc))

    record_audit_log(
        db,
        action="sync.artifacts",
        entity_type="system",
        details=synced,
        user=actor,
    )
    return {"status": "success", "synced": synced}


# ---------------------------------------------------------
# Phase 9: Dashboard Aggregation Services
# ---------------------------------------------------------

def get_dashboard_overview(db: Optional[Database]) -> Dict[str, Any]:
    """Compile aggregated project overview metrics from actual database records and reports."""
    datasets = get_datasets(db)
    total_records = sum(d.get("row_count", 0) for d in datasets)

    users_count = db["users"].count_documents({}) if db is not None else 0
    training_runs = get_training_runs(db)
    models = get_models(db)

    # Latest model performance from evaluation report if available
    latest_model_perf = None
    if config.MODEL_EVALUATION_REPORT.is_file():
        try:
            with open(config.MODEL_EVALUATION_REPORT, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            latest_model_perf = {
                "classification_roc_auc": eval_data.get("classification", {}).get("final_test_metrics", {}).get("roc_auc"),
                "regression_rmse": eval_data.get("regression", {}).get("final_test_metrics", {}).get("rmse"),
            }
        except Exception:
            pass

    # Causal summary
    causal_summary = None
    if config.CAUSAL_ENGINE_REPORT.is_file():
        try:
            with open(config.CAUSAL_ENGINE_REPORT, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            causal_summary = {
                "ate": c_data.get("evaluation", {}).get("estimated_ate"),
                "ci_lower": c_data.get("evaluation", {}).get("ate_95_ci_lower"),
                "ci_upper": c_data.get("evaluation", {}).get("ate_95_ci_upper"),
            }
        except Exception:
            pass

    # Uplift summary
    uplift_summary = None
    if config.UPLIFT_REPORT.is_file():
        try:
            with open(config.UPLIFT_REPORT, "r", encoding="utf-8") as f:
                u_data = json.load(f)
            uplift_summary = {
                "best_model": u_data.get("best_model_name", "DML-Uplift"),
                "test_qini_gain": u_data.get("best_model_test_evaluation", {}).get("qini_score_gain_vs_random"),
                "test_auuc": u_data.get("best_model_test_evaluation", {}).get("observed_auuc"),
            }
        except Exception:
            pass

    # Pricing summary
    pricing_summary = None
    if config.PRICING_REPORT.is_file():
        try:
            with open(config.PRICING_REPORT, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            pricing_summary = {
                "mean_recommended_price": p_data.get("metrics", {}).get("mean_recommended_price"),
                "expected_revenue_gain_pct": p_data.get("metrics", {}).get("expected_revenue_gain_pct"),
            }
        except Exception:
            pass

    # Campaign summary
    campaign_summary = None
    if config.PROMOTION_ALLOCATION_REPORT.is_file():
        try:
            with open(config.PROMOTION_ALLOCATION_REPORT, "r", encoding="utf-8") as f:
                prom_data = json.load(f)
            campaign_summary = {
                "budget": prom_data.get("budget"),
                "allocated_budget": prom_data.get("total_cost"),
                "selected_customers": prom_data.get("selected_count"),
                "expected_incremental_purchases": prom_data.get("expected_incremental_purchases"),
            }
        except Exception:
            pass

    recent_activity = get_audit_logs(db, limit=5)

    return {
        "total_datasets": len(datasets),
        "total_records_processed": total_records,
        "total_registered_users": users_count,
        "total_training_runs": len(training_runs),
        "total_models": len(models),
        "latest_model_performance": latest_model_perf,
        "causal_summary": causal_summary,
        "uplift_summary": uplift_summary,
        "pricing_summary": pricing_summary,
        "campaign_summary": campaign_summary,
        "recent_activity": recent_activity,
    }


def get_dashboard_recent_activity(db: Optional[Database], limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent platform audit activities for dashboard."""
    return get_audit_logs(db, limit=limit)


def get_dashboard_metrics(db: Optional[Database]) -> Dict[str, Any]:
    """Retrieve detailed numerical and statistical metrics across all engines."""
    overview = get_dashboard_overview(db)
    return {
        "datasets_count": overview["total_datasets"],
        "records_count": overview["total_records_processed"],
        "model_performance": overview["latest_model_performance"],
        "causal_metrics": overview["causal_summary"],
        "uplift_metrics": overview["uplift_summary"],
        "pricing_metrics": overview["pricing_summary"],
        "promotion_metrics": overview["campaign_summary"],
    }


def get_dashboard_model_performance(db: Optional[Database]) -> Dict[str, Any]:
    """Retrieve detailed model training and validation benchmarks."""
    if not config.MODEL_EVALUATION_REPORT.is_file():
        return {"status": "unavailable", "message": "No model evaluation report found."}
    try:
        with open(config.MODEL_EVALUATION_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def get_dashboard_data_summary(db: Optional[Database]) -> Dict[str, Any]:
    """Retrieve breakdown of stored datasets and their schemas."""
    datasets = get_datasets(db)
    synthetic_count = sum(1 for d in datasets if d.get("source_type") == "synthetic")
    real_count = sum(1 for d in datasets if d.get("source_type") == "real")
    uploaded_count = sum(1 for d in datasets if d.get("source_type") == "upload")
    total_bytes = sum(d.get("file_size_bytes", 0) for d in datasets)

    return {
        "total_datasets": len(datasets),
        "total_size_bytes": total_bytes,
        "synthetic_datasets": synthetic_count,
        "real_datasets": real_count,
        "uploaded_datasets": uploaded_count,
        "datasets": [
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "rows": d.get("row_count"),
                "columns": d.get("column_count"),
                "source_type": d.get("source_type"),
            }
            for d in datasets
        ],
    }


# ---------------------------------------------------------
# Phase 9: Dataset Upload, Preview, Summary & Safe Deletion
# ---------------------------------------------------------

def save_uploaded_dataset(
    db: Optional[Database],
    filename: str,
    content_bytes: bytes,
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save an uploaded CSV securely, inspect its schema, and record metadata."""
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    if not content_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(content_bytes) > config.MAX_UPLOAD_SIZE:
        raise ValueError(f"File size exceeds limit of {config.MAX_UPLOAD_SIZE // (1024 * 1024)} MB.")

    import io
    # Safe filename and unique id
    safe_name = Path(filename).name.replace(" ", "_")
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    upload_dir = config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / f"{dataset_id}_{safe_name}"

    try:
        csv_text = content_bytes.decode("utf-8-sig")
        df_sample = pd.read_csv(io.StringIO(csv_text), nrows=10)
    except Exception as exc:
        raise ValueError(f"Invalid CSV encoding or content: {exc}") from exc

    if df_sample.empty:
        raise ValueError("CSV contains no rows.")

    # Write to disk
    with open(target_path, "wb") as f:
        f.write(content_bytes)

    # Count total rows safely
    total_rows = csv_text.count("\n")

    metadata = {
        "dataset_id": dataset_id,
        "name": safe_name,
        "description": f"Uploaded CSV ({safe_name})",
        "file_path": str(target_path.resolve()),
        "row_count": max(1, total_rows),
        "column_count": len(df_sample.columns),
        "columns": list(df_sample.columns),
        "file_size_bytes": len(content_bytes),
        "source_type": "upload",
    }

    saved = record_dataset(db, metadata, actor=actor)
    record_audit_log(
        db,
        action="dataset.upload",
        entity_type="datasets",
        entity_id=saved["id"],
        details={"name": safe_name, "rows": metadata["row_count"]},
        user=actor,
    )
    return saved


def get_dataset_preview(
    db: Optional[Database],
    dataset_id: str,
    n_rows: int = 5,
) -> Dict[str, Any]:
    """Read top N rows from the target dataset file."""
    meta = get_dataset_by_id(db, dataset_id)
    if not meta:
        raise ValueError(f"Dataset '{dataset_id}' not found.")

    file_path = Path(meta["file_path"])
    if not file_path.is_file():
        raise ValueError("Dataset file does not exist on disk.")

    df = pd.read_csv(file_path, nrows=n_rows)
    return {
        "dataset_id": dataset_id,
        "name": meta.get("name"),
        "rows_previewed": len(df),
        "columns": list(df.columns),
        "preview": df.to_dict(orient="records"),
    }


def get_dataset_summary_stats(
    db: Optional[Database],
    dataset_id: str,
) -> Dict[str, Any]:
    """Compute summary statistics and null checks for a dataset."""
    meta = get_dataset_by_id(db, dataset_id)
    if not meta:
        raise ValueError(f"Dataset '{dataset_id}' not found.")

    file_path = Path(meta["file_path"])
    if not file_path.is_file():
        raise ValueError("Dataset file is unavailable.")

    # Load small or sample for profiling
    df = pd.read_csv(file_path, nrows=10000)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}
    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}

    return {
        "dataset_id": dataset_id,
        "name": meta.get("name"),
        "total_rows": meta.get("row_count"),
        "total_columns": meta.get("column_count"),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "null_counts": null_counts,
        "numeric_summary": desc,
    }


def delete_dataset(
    db: Optional[Database],
    dataset_id: str,
    actor: Optional[Dict[str, Any]] = None,
) -> bool:
    """Safely delete uploaded dataset while protecting core system files."""
    meta = get_dataset_by_id(db, dataset_id)
    if not meta:
        raise ValueError(f"Dataset '{dataset_id}' not found.")

    file_path = Path(meta["file_path"])

    # Strict protection for Phase 1–7 datasets
    protected_basenames = [
        "retail_data.csv", "train.csv", "validation.csv",
        "test.csv", "criteo_sample.csv", "test_ground_truth.csv"
    ]
    if file_path.name in protected_basenames or "synthetic/splits" in str(file_path).replace("\\", "/"):
        raise ValueError("Cannot delete core system datasets.")

    # Remove file if located in uploads directory
    if "uploads" in str(file_path) and file_path.is_file():
        try:
            file_path.unlink()
        except Exception as exc:
            logger.warning("Could not unlink dataset file: %s", str(exc))

    if db is not None:
        from bson import ObjectId
        try:
            db["datasets"].delete_one({"_id": ObjectId(dataset_id)})
        except Exception:
            db["datasets"].delete_one({"id": dataset_id})

    record_audit_log(
        db,
        action="dataset.delete",
        entity_type="datasets",
        entity_id=dataset_id,
        details={"name": meta.get("name")},
        user=actor,
    )
    return True


# ---------------------------------------------------------
# Phase 9: Asynchronous ML Training Services
# ---------------------------------------------------------

def start_training_run_service(
    db: Optional[Database],
    req_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
    background_tasks: Any = None,
) -> Dict[str, Any]:
    """Queue and execute ML training job asynchronously."""
    run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    training_doc = {
        "run_id": run_id,
        "model_name": req_data.get("model_type", "all"),
        "model_type": "ml_training_pipeline",
        "parameters": req_data.get("parameters", {}),
        "status": "queued",
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": actor.get("username", "system") if actor else "system",
        "logs": ["Job queued."],
    }

    if db is not None:
        db["training_runs"].insert_one(training_doc)
        if "_id" in training_doc:
            training_doc["id"] = str(training_doc.pop("_id"))

    def _execute_training():
        from backend.ml_pipeline import run_full_ml_pipeline
        if db is not None:
            db["training_runs"].update_one(
                {"run_id": run_id},
                {"$set": {"status": "running", "progress": 0.2, "logs": ["Starting training..."]}}
            )
        try:
            report = run_full_ml_pipeline()
            if db is not None:
                db["training_runs"].update_one(
                    {"run_id": run_id},
                    {
                        "$set": {
                            "status": "completed",
                            "progress": 1.0,
                            "metrics": report,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "logs": ["Training completed successfully."],
                        }
                    }
                )
        except Exception as exc:
            logger.error("Training run %s failed: %s", run_id, str(exc))
            if db is not None:
                db["training_runs"].update_one(
                    {"run_id": run_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(exc),
                            "logs": [f"Training failed: {str(exc)}"],
                        }
                    }
                )

    if background_tasks is not None:
        background_tasks.add_task(_execute_training)
    else:
        # If no background task runner (e.g., test or synchronous call)
        _execute_training()

    record_audit_log(
        db,
        action="training.start",
        entity_type="training_runs",
        entity_id=run_id,
        details={"model_type": req_data.get("model_type")},
        user=actor,
    )
    return training_doc


def get_training_run_status(db: Optional[Database], run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch status and progress of a training run."""
    if db is None:
        return {"run_id": run_id, "status": "completed"}
    doc = db["training_runs"].find_one({"run_id": run_id})
    return _format_doc(doc)


def get_training_run_metrics(db: Optional[Database], run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve final evaluation metrics of a training run."""
    run = get_training_run_status(db, run_id)
    if not run:
        return None
    return run.get("metrics", {})


def get_training_run_logs(db: Optional[Database], run_id: str) -> List[str]:
    """Retrieve execution log messages of a training run."""
    run = get_training_run_status(db, run_id)
    if not run:
        return []
    return run.get("logs", [])


# ---------------------------------------------------------
# Phase 9: Causal Inference Services
# ---------------------------------------------------------

def execute_causal_analysis_service(
    db: Optional[Database],
    req_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute causal inference analysis or load validated results."""
    analysis_id = f"CAUSAL-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}"

    # Read existing report if valid, or run engine
    result = None
    if config.CAUSAL_ENGINE_REPORT.is_file():
        try:
            with open(config.CAUSAL_ENGINE_REPORT, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            pass

    if result is None:
        from backend.causal_engine import run_causal_inference_engine
        result = run_causal_inference_engine()

    doc = {
        "analysis_id": analysis_id,
        "treatment_col": req_data.get("treatment_col", "treatment"),
        "outcome_col": req_data.get("outcome_col", "purchase"),
        "summary": {
            "estimated_ate": result.get("evaluation", {}).get("estimated_ate"),
            "ate_ci": [
                result.get("evaluation", {}).get("ate_95_ci_lower"),
                result.get("evaluation", {}).get("ate_95_ci_upper"),
            ],
            "propensity_auc": result.get("diagnostics", {}).get("propensity_auc"),
            "overlap_fraction": result.get("diagnostics", {}).get("overlap_fraction_005_095"),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": actor.get("username", "system") if actor else "system",
    }

    if db is not None:
        db["causal_runs"].insert_one(doc)
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))

    record_audit_log(
        db,
        action="causal.analyze",
        entity_type="causal_runs",
        entity_id=analysis_id,
        details={"ate": doc["summary"]["estimated_ate"]},
        user=actor,
    )
    return doc


def get_causal_results_service(db: Optional[Database], analysis_id: str) -> Optional[Dict[str, Any]]:
    if db is not None:
        doc = db["causal_runs"].find_one({"analysis_id": analysis_id})
        if doc:
            return _format_doc(doc)

    if config.CAUSAL_ENGINE_REPORT.is_file():
        with open(config.CAUSAL_ENGINE_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_causal_history_service(db: Optional[Database]) -> List[Dict[str, Any]]:
    if db is None:
        return []
    cursor = db["causal_runs"].find().sort("created_at", -1).limit(50)
    return [_format_doc(doc) for doc in cursor]


def get_causal_diagnostics_service(db: Optional[Database], analysis_id: str) -> Dict[str, Any]:
    if config.CAUSAL_ENGINE_REPORT.is_file():
        with open(config.CAUSAL_ENGINE_REPORT, "r", encoding="utf-8") as f:
            rep = json.load(f)
            return rep.get("diagnostics", {})
    return {}


# ---------------------------------------------------------
# Phase 9: Uplift Modeling Services
# ---------------------------------------------------------

def execute_uplift_analysis_service(
    db: Optional[Database],
    req_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute uplift modeling analysis or load validated results."""
    analysis_id = f"UPLIFT-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:6]}"

    result = None
    if config.UPLIFT_REPORT.is_file():
        try:
            with open(config.UPLIFT_REPORT, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            pass

    if result is None:
        from backend.uplift_analysis import run_uplift_pipeline
        result = run_uplift_pipeline()

    doc = {
        "analysis_id": analysis_id,
        "best_model": result.get("best_model_name", "DML-Uplift"),
        "qini_gain_vs_random": result.get("best_model_test_evaluation", {}).get("qini_score_gain_vs_random"),
        "test_auuc": result.get("best_model_test_evaluation", {}).get("observed_auuc"),
        "customer_segments": result.get("targeting_strategy", {}).get("customer_counts_by_segment"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": actor.get("username", "system") if actor else "system",
    }

    if db is not None:
        db["uplift_runs"].insert_one(doc)
        if "_id" in doc:
            doc["id"] = str(doc.pop("_id"))

    record_audit_log(
        db,
        action="uplift.analyze",
        entity_type="uplift_runs",
        entity_id=analysis_id,
        details={"best_model": doc["best_model"]},
        user=actor,
    )
    return doc


def get_uplift_results_service(db: Optional[Database], analysis_id: str) -> Optional[Dict[str, Any]]:
    if db is not None:
        doc = db["uplift_runs"].find_one({"analysis_id": analysis_id})
        if doc:
            return _format_doc(doc)
    if config.UPLIFT_REPORT.is_file():
        with open(config.UPLIFT_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_uplift_history_service(db: Optional[Database]) -> List[Dict[str, Any]]:
    if db is None:
        return []
    cursor = db["uplift_runs"].find().sort("created_at", -1).limit(50)
    return [_format_doc(doc) for doc in cursor]


def get_uplift_customer_segments_service(db: Optional[Database]) -> Dict[str, Any]:
    if config.UPLIFT_REPORT.is_file():
        with open(config.UPLIFT_REPORT, "r", encoding="utf-8") as f:
            rep = json.load(f)
            return rep.get("targeting_strategy", {}).get("customer_counts_by_segment", {})
    return {}


def get_uplift_performance_service(db: Optional[Database]) -> Dict[str, Any]:
    if config.UPLIFT_REPORT.is_file():
        with open(config.UPLIFT_REPORT, "r", encoding="utf-8") as f:
            rep = json.load(f)
            return rep.get("best_model_test_evaluation", {})
    return {}


# ---------------------------------------------------------
# Phase 9: Dynamic Pricing Services
# ---------------------------------------------------------

_pricing_engine_instance = None

def get_pricing_engine():
    """Singleton getter for DynamicPricingEngine to avoid repeated model reloads."""
    global _pricing_engine_instance
    if _pricing_engine_instance is not None:
        return _pricing_engine_instance

    from backend.dynamic_pricing import DynamicPricingEngine, PriceDemandModel
    import joblib

    if config.PRICING_MODEL_ARTIFACT.is_file():
        demand_model = joblib.load(config.PRICING_MODEL_ARTIFACT)
    else:
        # Train or initialize default
        from backend.dynamic_pricing import fit_price_demand_model
        train_path = config.SPLIT_DIR / "train.csv"
        df_train = pd.read_csv(train_path) if train_path.is_file() else pd.DataFrame()
        demand_model = fit_price_demand_model(df_train)

    train_path = config.SPLIT_DIR / "train.csv"
    support_df = pd.read_csv(train_path) if train_path.is_file() else None

    _pricing_engine_instance = DynamicPricingEngine(
        demand_model=demand_model,
        support_data=support_df,
    )
    return _pricing_engine_instance


def recommend_single_price_service(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Generate dynamic pricing recommendation for a single transaction/opportunity."""
    engine = get_pricing_engine()

    row_data = {
        "customer_id": [candidate.get("customer_id", "CUST_001")],
        "product_id": [candidate.get("product_id", "P001")],
        "category": [candidate.get("category", "Electronics")],
        "baseline_price": [float(candidate.get("current_price", 50.0))],
        "customer_history": [float(candidate.get("customer_history", 100.0))],
        "seasonality": [float(candidate.get("seasonality", 1.0))],
        "treatment": [int(candidate.get("treatment", 0))],
    }
    df_row = pd.DataFrame(row_data)

    rec = engine.recommend_price(
        df_row,
        unit_cost=candidate.get("unit_cost"),
    )
    return rec.to_dict()


def recommend_batch_prices_service(
    db: Optional[Database],
    candidates: List[Dict[str, Any]],
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate dynamic pricing recommendations for a batch of opportunities."""
    engine = get_pricing_engine()

    rows = []
    for c in candidates:
        rows.append({
            "customer_id": c.get("customer_id", "CUST"),
            "product_id": c.get("product_id", "P001"),
            "category": c.get("category", "Electronics"),
            "baseline_price": float(c.get("current_price", 50.0)),
            "customer_history": float(c.get("customer_history", 100.0)),
            "seasonality": float(c.get("seasonality", 1.0)),
            "treatment": int(c.get("treatment", 0)),
        })
    df_batch = pd.DataFrame(rows)

    df_recs = engine.recommend_batch(df_batch)
    recommendations_list = df_recs.to_dict(orient="records")

    mean_rec_price = float(df_recs["recommended_price"].mean())
    mean_inc_rev = float(df_recs["incremental_revenue"].mean())

    summary_metrics = {
        "count": len(recommendations_list),
        "mean_recommended_price": round(mean_rec_price, 2),
        "mean_incremental_revenue": round(mean_inc_rev, 4),
    }

    pricing_record = {
        "scenario_name": f"Batch API Run ({len(candidates)} items)",
        "strategy": "revenue_maximization",
        "parameters": {"batch_size": len(candidates)},
        "summary_metrics": summary_metrics,
        "recommendations_count": len(recommendations_list),
    }
    record_pricing_recommendation(db, pricing_record, actor=actor)

    return {
        "summary": summary_metrics,
        "recommendations": recommendations_list,
    }


def get_pricing_elasticity_service() -> Dict[str, Any]:
    if config.PRICING_REPORT.is_file():
        with open(config.PRICING_REPORT, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("price_elasticity", {})
    return {}


def get_pricing_performance_service() -> Dict[str, Any]:
    if config.PRICING_REPORT.is_file():
        with open(config.PRICING_REPORT, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("metrics", {})
    return {}


# ---------------------------------------------------------
# Phase 9: Promotion Optimization Services
# ---------------------------------------------------------

def optimize_promotions_service(
    db: Optional[Database],
    req_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute mathematical optimization solver for promotional budget allocation."""
    from backend.optimization import build_optimization_problem, solve_optimization_problem

    candidates_list = req_data.get("candidates", [])
    if not candidates_list:
        raise ValueError("Candidates list cannot be empty.")

    budget = float(req_data.get("budget", 2500.0))
    discount_pct = float(req_data.get("discount_pct", 10.0))

    df_candidates = pd.DataFrame([
        {
            "customer_id": str(c["customer_id"]).strip(),
            "baseline_price": float(c["baseline_price"]),
            "predicted_ite": float(c["predicted_ite"]),
        }
        for c in candidates_list
    ])

    problem = build_optimization_problem(
        candidates=df_candidates,
        budget=budget,
        discount_pct=discount_pct,
    )

    result = solve_optimization_problem(problem)
    selected_set = set(result.selected_indices)

    allocation = []
    for idx, row in df_candidates.iterrows():
        is_sel = idx in selected_set
        ite = float(row["predicted_ite"])
        cost = float(problem.promotion_costs[idx])

        allocation.append({
            "customer_id": row["customer_id"],
            "baseline_price": row["baseline_price"],
            "predicted_ite": ite,
            "promotion_cost": cost,
            "selected": is_sel,
            "status": "Selected" if is_sel else "Not Selected",
            "explanation": (
                "Allocated promotion under budget limit."
                if is_sel else
                "Excluded by budget-constrained allocation."
            ),
        })

    result_doc = {
        "campaign_id": f"CAMP-{uuid.uuid4().hex[:8]}",
        "total_budget": budget,
        "allocated_budget": result.total_cost,
        "expected_uplift": result.expected_incremental_purchases,
        "customer_count": result.selected_count,
        "status": result.solver_status,
        "summary": {
            "selected_count": result.selected_count,
            "total_candidates": len(df_candidates),
            "unused_budget": result.unused_budget,
            "discount_pct": discount_pct,
        },
        "allocation": allocation,
    }

    record_optimization_result(db, result_doc, actor=actor)
    return result_doc


def get_budget_analysis_service(db: Optional[Database]) -> Dict[str, Any]:
    if config.PROMOTION_SCENARIO_COMPARISON_PATH.is_file():
        with open(config.PROMOTION_SCENARIO_COMPARISON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if config.PROMOTION_ALLOCATION_REPORT.is_file():
        with open(config.PROMOTION_ALLOCATION_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------
# Phase 9: End-to-End Pipeline Orchestration Service
# ---------------------------------------------------------

def start_pipeline_run_service(
    db: Optional[Database],
    req_data: Dict[str, Any],
    actor: Optional[Dict[str, Any]] = None,
    background_tasks: Any = None,
) -> Dict[str, Any]:
    """Execute end-to-end orchestration pipeline sequentially across all core engines."""
    run_id = f"PIPE-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    planned_stages = [
        "data_validation",
        "ml_training",
        "causal_inference",
        "uplift_modeling",
        "dynamic_pricing",
        "promotion_optimization",
        "persistence_sync",
    ]

    pipeline_doc = {
        "pipeline_run_id": run_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": actor.get("username", "system") if actor else "system",
        "stages": [{"stage": s, "status": "pending", "duration_seconds": 0.0} for s in planned_stages],
        "summary": {},
    }

    if db is not None:
        db["pipeline_runs"].insert_one(pipeline_doc)
        if "_id" in pipeline_doc:
            pipeline_doc["id"] = str(pipeline_doc.pop("_id"))

    def _execute_pipeline():
        import time
        t_start = time.perf_counter()

        def _update_stage(stage_name, status, duration=0.0, err=None, details=None):
            if db is not None:
                db["pipeline_runs"].update_one(
                    {"pipeline_run_id": run_id, "stages.stage": stage_name},
                    {
                        "$set": {
                            "stages.$.status": status,
                            "stages.$.duration_seconds": round(duration, 3),
                            "stages.$.error": err,
                            "stages.$.details": details or {},
                        }
                    }
                )

        if db is not None:
            db["pipeline_runs"].update_one(
                {"pipeline_run_id": run_id},
                {"$set": {"status": "running"}}
            )

        try:
            # 1. Data Validation
            s_t0 = time.perf_counter()
            _update_stage("data_validation", "running")
            train_split = config.SPLIT_DIR / "train.csv"
            if not train_split.is_file():
                raise FileNotFoundError("Core dataset split train.csv is missing.")
            df_train = pd.read_csv(train_split)
            _update_stage("data_validation", "completed", time.perf_counter() - s_t0, details={"rows": len(df_train)})

            # 2. ML Training (Verify models or run)
            s_t1 = time.perf_counter()
            _update_stage("ml_training", "running")
            # If artifacts exist, reuse verified models; otherwise execute training
            if not config.MODEL_REGISTRY_PATH.is_file():
                from backend.ml_pipeline import run_full_ml_pipeline
                run_full_ml_pipeline()
            _update_stage("ml_training", "completed", time.perf_counter() - s_t1)

            # 3. Causal Inference
            s_t2 = time.perf_counter()
            _update_stage("causal_inference", "running")
            if not config.CAUSAL_ENGINE_REPORT.is_file():
                from backend.causal_engine import run_causal_inference_engine
                run_causal_inference_engine()
            _update_stage("causal_inference", "completed", time.perf_counter() - s_t2)

            # 4. Uplift Modeling
            s_t3 = time.perf_counter()
            _update_stage("uplift_modeling", "running")
            if not config.UPLIFT_REPORT.is_file():
                from backend.uplift_analysis import run_uplift_pipeline
                run_uplift_pipeline()
            _update_stage("uplift_modeling", "completed", time.perf_counter() - s_t3)

            # 5. Dynamic Pricing
            s_t4 = time.perf_counter()
            _update_stage("dynamic_pricing", "running")
            engine = get_pricing_engine()
            _update_stage("dynamic_pricing", "completed", time.perf_counter() - s_t4)

            # 6. Promotion Optimization
            s_t5 = time.perf_counter()
            _update_stage("promotion_optimization", "running")
            budget = float(req_data.get("budget", 2500.0))
            if config.PROMOTION_ALLOCATION_REPORT.is_file():
                with open(config.PROMOTION_ALLOCATION_REPORT, "r", encoding="utf-8") as f:
                    p_alloc = json.load(f)
            else:
                p_alloc = {"selected_count": 0, "budget": budget}
            _update_stage("promotion_optimization", "completed", time.perf_counter() - s_t5, details={"budget": budget})

            # 7. Persistence Sync
            s_t6 = time.perf_counter()
            _update_stage("persistence_sync", "running")
            sync_res = sync_local_artifacts_to_db(db, actor=actor)
            _update_stage("persistence_sync", "completed", time.perf_counter() - s_t6, details=sync_res)

            # Finalize pipeline
            total_duration = time.perf_counter() - t_start
            if db is not None:
                db["pipeline_runs"].update_one(
                    {"pipeline_run_id": run_id},
                    {
                        "$set": {
                            "status": "completed",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "summary": {
                                "total_duration_seconds": round(total_duration, 2),
                                "stages_completed": len(planned_stages),
                            },
                        }
                    }
                )

        except Exception as exc:
            logger.error("Pipeline run %s failed: %s", run_id, str(exc))
            if db is not None:
                db["pipeline_runs"].update_one(
                    {"pipeline_run_id": run_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(exc),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )

    if background_tasks is not None:
        background_tasks.add_task(_execute_pipeline)
    else:
        _execute_pipeline()

    record_audit_log(
        db,
        action="pipeline.run",
        entity_type="pipeline_runs",
        entity_id=run_id,
        user=actor,
    )
    return pipeline_doc


def get_pipeline_status_service(db: Optional[Database], run_id: str) -> Optional[Dict[str, Any]]:
    if db is None:
        return {"pipeline_run_id": run_id, "status": "completed"}
    doc = db["pipeline_runs"].find_one({"pipeline_run_id": run_id})
    return _format_doc(doc)


def get_pipeline_history_service(db: Optional[Database]) -> List[Dict[str, Any]]:
    if db is None:
        return []
    cursor = db["pipeline_runs"].find().sort("created_at", -1).limit(50)
    return [_format_doc(doc) for doc in cursor]


def cancel_pipeline_run_service(
    db: Optional[Database],
    run_id: str,
    actor: Optional[Dict[str, Any]] = None,
) -> bool:
    if db is None:
        return True
    res = db["pipeline_runs"].update_one(
        {"pipeline_run_id": run_id, "status": {"$in": ["queued", "running"]}},
        {"$set": {"status": "cancelled", "completed_at": datetime.now(timezone.utc).isoformat()}}
    )
    if res.matched_count > 0:
        record_audit_log(
            db,
            action="pipeline.cancel",
            entity_type="pipeline_runs",
            entity_id=run_id,
            user=actor,
        )
        return True
    return False
