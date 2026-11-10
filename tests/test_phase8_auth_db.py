"""Comprehensive automated tests for Phase 8: MongoDB Atlas Integration, Authentication, RBAC, and Persistent Storage."""

from datetime import timedelta
import mongomock
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.database import (
    check_db_health,
    init_db_indexes,
    sanitize_uri,
    set_db_override,
)
from backend.main import app
from backend.schemas import UserCreate, UserRole
from backend.services import (
    authenticate_user,
    change_user_password,
    create_user,
    get_audit_logs,
    get_datasets,
    get_models,
    get_optimization_results,
    get_pricing_recommendations,
    get_promotion_campaigns,
    get_training_runs,
    record_audit_log,
    record_dataset,
    record_model,
    record_optimization_result,
    record_pricing_recommendation,
    record_promotion_campaign,
    record_training_run,
    sync_local_artifacts_to_db,
    update_user_role,
    update_user_status,
)


@pytest.fixture(autouse=True)
def mock_mongo():
    """Provide an isolated in-memory mongomock database for each test."""
    client = mongomock.MongoClient()
    mock_db = client["econocausal_test"]
    set_db_override(mock_db)
    init_db_indexes(mock_db)
    yield mock_db
    set_db_override(None)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------
# 1. Database Module & Health Check Tests
# ---------------------------------------------------------

def test_sanitize_uri():
    uri_with_pw = "mongodb+srv://admin:SecretPass123@cluster0.abc.mongodb.net/econocausal?retryWrites=true"
    sanitized = sanitize_uri(uri_with_pw)
    assert "SecretPass123" not in sanitized
    assert ":***@" in sanitized

    assert sanitize_uri("") == "<empty>"


def test_db_health_with_mock(mock_mongo):
    health = check_db_health()
    assert health["status"] in ("healthy", "disconnected", "unhealthy")
    assert health["database"] == "econocausal_test"


def test_db_health_disconnected():
    set_db_override(None)
    # Temporarily blank out URI to simulate disconnected
    orig_uri = config.MONGODB_URI
    try:
        config.MONGODB_URI = ""
        health = check_db_health()
        assert health["status"] == "disconnected"
    finally:
        config.MONGODB_URI = orig_uri


def test_init_db_indexes(mock_mongo):
    res = init_db_indexes(mock_mongo)
    assert res["status"] == "success"
    assert "users" in res["indexes"]
    assert "datasets" in res["indexes"]
    assert "training_runs" in res["indexes"]
    assert "model_registry" in res["indexes"]
    assert "pricing_recommendations" in res["indexes"]
    assert "promotion_campaigns" in res["indexes"]
    assert "optimization_results" in res["indexes"]
    assert "audit_logs" in res["indexes"]


# ---------------------------------------------------------
# 2. Password Security & JWT Tokens
# ---------------------------------------------------------

def test_password_hashing_and_verification():
    pw = "SuperSecure123!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_generation_and_decoding():
    payload = {"sub": "user_123", "username": "analyst_bob", "role": "analyst"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=15))
    assert isinstance(token, str)

    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user_123"
    assert decoded["username"] == "analyst_bob"
    assert decoded["role"] == "analyst"


def test_jwt_expired_token():
    payload = {"sub": "user_expired"}
    token = create_access_token(payload, expires_delta=timedelta(seconds=-1))
    decoded = decode_access_token(token)
    assert decoded is None


def test_jwt_invalid_token():
    assert decode_access_token("not-a-valid-jwt-token") is None


# ---------------------------------------------------------
# 3. User Management & Authentication Services
# ---------------------------------------------------------

def test_create_user_and_authenticate(mock_mongo):
    user_in = UserCreate(
        email="analyst@econocausal.ai",
        username="analyst_jane",
        password="ValidPassword123",
        full_name="Jane Doe",
        role=UserRole.ANALYST,
    )
    created = create_user(mock_mongo, user_in)
    assert created["email"] == "analyst@econocausal.ai"
    assert created["username"] == "analyst_jane"
    assert "hashed_password" not in created
    assert created["role"] == "analyst"
    assert created["is_active"] is True

    # Duplicate email rejection
    with pytest.raises(ValueError, match="already exists"):
        create_user(mock_mongo, user_in)

    # Duplicate username rejection
    user_duplicate_username = UserCreate(
        email="other@econocausal.ai",
        username="analyst_jane",
        password="AnotherPassword123",
        role=UserRole.ANALYST,
    )
    with pytest.raises(ValueError, match="already taken"):
        create_user(mock_mongo, user_duplicate_username)

    # Successful authentication by email
    auth_email = authenticate_user(mock_mongo, "analyst@econocausal.ai", "ValidPassword123")
    assert auth_email is not None
    assert auth_email["username"] == "analyst_jane"

    # Successful authentication by username
    auth_user = authenticate_user(mock_mongo, "analyst_jane", "ValidPassword123")
    assert auth_user is not None

    # Failed authentication with bad password
    bad_auth = authenticate_user(mock_mongo, "analyst_jane", "WrongPassword")
    assert bad_auth is None


def test_user_role_and_status_update(mock_mongo):
    user_in = UserCreate(
        email="viewer@econocausal.ai",
        username="viewer_mark",
        password="Password123",
        role=UserRole.VIEWER,
    )
    user = create_user(mock_mongo, user_in)
    uid = user["id"]

    # Update role to admin
    updated = update_user_role(mock_mongo, uid, UserRole.ADMIN)
    assert updated["role"] == "admin"

    # Deactivate user
    deactivated = update_user_status(mock_mongo, uid, is_active=False)
    assert deactivated["is_active"] is False

    # Deactivated user cannot authenticate
    assert authenticate_user(mock_mongo, "viewer_mark", "Password123") is None

    # Reactivate user
    reactivated = update_user_status(mock_mongo, uid, is_active=True)
    assert reactivated["is_active"] is True
    assert authenticate_user(mock_mongo, "viewer_mark", "Password123") is not None


def test_change_user_password(mock_mongo):
    user_in = UserCreate(
        email="user@econocausal.ai",
        username="chg_user",
        password="OldPassword123",
        role=UserRole.ANALYST,
    )
    user = create_user(mock_mongo, user_in)
    uid = user["id"]

    # Wrong old password fails
    assert change_user_password(mock_mongo, uid, "WrongOld", "NewPassword456") is False

    # Correct old password succeeds
    assert change_user_password(mock_mongo, uid, "OldPassword123", "NewPassword456") is True
    assert authenticate_user(mock_mongo, "chg_user", "OldPassword123") is None
    assert authenticate_user(mock_mongo, "chg_user", "NewPassword456") is not None


# ---------------------------------------------------------
# 4. Metadata Persistence Services
# ---------------------------------------------------------

def test_metadata_persistence_services(mock_mongo):
    # Datasets
    ds = record_dataset(mock_mongo, {
        "name": "synthetic_train",
        "description": "Training split",
        "file_path": "data/synthetic/splits/train.csv",
        "row_count": 3000,
        "column_count": 6,
        "columns": ["baseline_price", "category", "treatment", "purchase"],
        "file_size_bytes": 10240,
        "source_type": "synthetic",
    })
    assert ds["name"] == "synthetic_train"
    assert len(get_datasets(mock_mongo)) == 1

    # Training Runs
    tr = record_training_run(mock_mongo, {
        "run_id": "RUN-2026-001",
        "model_name": "HistGradientBoosting",
        "model_type": "uplift",
        "metrics": {"auuc": 0.78, "qini": 0.45},
    })
    assert tr["run_id"] == "RUN-2026-001"
    assert len(get_training_runs(mock_mongo)) == 1

    # Model Registry
    mr = record_model(mock_mongo, {
        "model_name": "causal_dml_model",
        "version": "v1.0",
        "model_type": "DoubleML",
        "artifact_path": "models/artifacts/causal_dml_model.joblib",
        "metrics": {"ate": 0.12},
        "is_active": True,
    })
    assert mr["model_name"] == "causal_dml_model"
    assert len(get_models(mock_mongo)) == 1

    # Pricing Recommendations
    pr = record_pricing_recommendation(mock_mongo, {
        "scenario_name": "Holiday Discount 15%",
        "strategy": "profit_maximization",
        "summary_metrics": {"mean_recommended_price": 55.4},
        "recommendations_count": 500,
    })
    assert pr["scenario_name"] == "Holiday Discount 15%"
    assert len(get_pricing_recommendations(mock_mongo)) == 1

    # Promotion Campaigns & Optimization Results
    camp = record_promotion_campaign(mock_mongo, {
        "campaign_id": "CAMP-SUMMER-2026",
        "campaign_name": "Summer Promotion",
        "budget": 5000.0,
        "status": "active",
    })
    assert camp["campaign_id"] == "CAMP-SUMMER-2026"
    assert len(get_promotion_campaigns(mock_mongo)) == 1

    opt = record_optimization_result(mock_mongo, {
        "campaign_id": "CAMP-SUMMER-2026",
        "total_budget": 5000.0,
        "allocated_budget": 4850.0,
        "expected_uplift": 320.5,
        "customer_count": 850,
    })
    assert opt["allocated_budget"] == 4850.0
    assert len(get_optimization_results(mock_mongo)) == 1

    # Audit Logs
    audit = record_audit_log(
        mock_mongo,
        action="test.action",
        entity_type="test",
        entity_id="123",
        details={"info": "sample"},
    )
    assert audit is not None
    logs = get_audit_logs(mock_mongo)
    assert len(logs) >= 1


def test_sync_local_artifacts(mock_mongo):
    sync_res = sync_local_artifacts_to_db(mock_mongo)
    assert sync_res["status"] == "success"
    # Should have synced datasets, models, pricing, or campaigns if files exist
    assert "synced" in sync_res


# ---------------------------------------------------------
# 5. FastAPI Endpoints & RBAC Integration
# ---------------------------------------------------------

def test_api_auth_flow_and_rbac(client, mock_mongo):
    # 1. Register Admin
    admin_reg = client.post("/api/auth/register", json={
        "email": "admin@econocausal.ai",
        "username": "sysadmin",
        "password": "AdminPassword123!",
        "full_name": "System Administrator",
        "role": "admin",
    })
    assert admin_reg.status_code == 201
    admin_data = admin_reg.json()
    assert admin_data["role"] == "admin"

    # 2. Register Analyst
    analyst_reg = client.post("/api/auth/register", json={
        "email": "analyst@econocausal.ai",
        "username": "data_analyst",
        "password": "AnalystPassword123!",
        "role": "analyst",
    })
    assert analyst_reg.status_code == 201

    # 3. Register Viewer
    viewer_reg = client.post("/api/auth/register", json={
        "email": "viewer@econocausal.ai",
        "username": "exec_viewer",
        "password": "ViewerPassword123!",
        "role": "viewer",
    })
    assert viewer_reg.status_code == 201

    # 4. Login as Analyst
    login_resp = client.post("/api/auth/login", json={
        "username_or_email": "data_analyst",
        "password": "AnalystPassword123!",
    })
    assert login_resp.status_code == 200
    analyst_token = login_resp.json()["access_token"]
    analyst_headers = {"Authorization": f"Bearer {analyst_token}"}

    # 5. Login as Admin
    admin_login = client.post("/api/auth/login", json={
        "username_or_email": "sysadmin",
        "password": "AdminPassword123!",
    })
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 6. Login as Viewer
    viewer_login = client.post("/api/auth/login", json={
        "username_or_email": "exec_viewer",
        "password": "ViewerPassword123!",
    })
    assert viewer_login.status_code == 200
    viewer_token = viewer_login.json()["access_token"]
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    # 7. Test /api/auth/me
    me_resp = client.get("/api/auth/me", headers=analyst_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "data_analyst"

    # Unauthenticated /api/auth/me
    assert client.get("/api/auth/me").status_code == 401

    # 8. RBAC Test: Listing users is Admin-only
    # Analyst attempt -> 403 Forbidden
    analyst_user_list = client.get("/api/users", headers=analyst_headers)
    assert analyst_user_list.status_code == 403

    # Viewer attempt -> 403 Forbidden
    viewer_user_list = client.get("/api/users", headers=viewer_headers)
    assert viewer_user_list.status_code == 403

    # Admin attempt -> 200 OK
    admin_user_list = client.get("/api/users", headers=admin_headers)
    assert admin_user_list.status_code == 200
    assert len(admin_user_list.json()) == 3

    # 9. RBAC Test: Creating dataset records (Admin & Analyst allowed, Viewer forbidden)
    ds_payload = {
        "name": "q3_sales_dataset",
        "description": "Sales data",
        "file_path": "data/sales_q3.csv",
        "row_count": 1500,
        "column_count": 5,
        "columns": ["id", "price", "treatment"],
        "file_size_bytes": 45000,
        "source_type": "synthetic",
    }

    # Viewer attempt -> 403 Forbidden
    viewer_post_ds = client.post("/api/datasets", json=ds_payload, headers=viewer_headers)
    assert viewer_post_ds.status_code == 403

    # Analyst attempt -> 201 Created
    analyst_post_ds = client.post("/api/datasets", json=ds_payload, headers=analyst_headers)
    assert analyst_post_ds.status_code == 201

    # Viewer can read datasets -> 200 OK
    viewer_get_ds = client.get("/api/datasets", headers=viewer_headers)
    assert viewer_get_ds.status_code == 200
    assert len(viewer_get_ds.json()) == 1

    # 10. Audit logs endpoint (Admin only)
    assert client.get("/api/audit-logs", headers=viewer_headers).status_code == 403
    assert client.get("/api/audit-logs", headers=analyst_headers).status_code == 403
    admin_audit = client.get("/api/audit-logs", headers=admin_headers)
    assert admin_audit.status_code == 200
    assert len(admin_audit.json()) >= 1
