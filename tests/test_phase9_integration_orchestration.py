"""Comprehensive automated test suite for Phase 9: Unified Backend API & End-to-End Orchestration."""

import io
import mongomock
import pytest
from fastapi.testclient import TestClient

from backend.database import init_db_indexes, set_db_override
from backend.main import app
from backend.schemas import UserCreate, UserRole
from backend.services import create_user, sync_local_artifacts_to_db


@pytest.fixture(autouse=True)
def mock_db():
    """Isolate each test in an in-memory mongomock database with clean indexes and synced local metadata."""
    client = mongomock.MongoClient()
    db = client["econocausal_phase9_test"]
    set_db_override(db)
    init_db_indexes(db)
    sync_local_artifacts_to_db(db)
    yield db
    set_db_override(None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_tokens(client, mock_db):
    """Create and return tokens for Admin, Analyst, and Viewer roles."""
    users = [
        ("admin@econocausal.ai", "admin_user", "AdminPass123!", UserRole.ADMIN),
        ("analyst@econocausal.ai", "analyst_user", "AnalystPass123!", UserRole.ANALYST),
        ("viewer@econocausal.ai", "viewer_user", "ViewerPass123!", UserRole.VIEWER),
    ]
    tokens = {}
    for email, username, pw, role in users:
        create_user(mock_db, UserCreate(email=email, username=username, password=pw, role=role))
        login_res = client.post("/api/auth/login", json={"username_or_email": username, "password": pw})
        assert login_res.status_code == 200
        tokens[role.value] = login_res.json()["access_token"]
    return tokens


def auth_header(token: str):
    return {"Authorization": f"Bearer {token}"}


# =========================================================
# 1. Dashboard API Tests
# =========================================================

def test_dashboard_apis(client, auth_tokens):
    headers = auth_header(auth_tokens["viewer"])

    # Overview
    res = client.get("/api/dashboard/overview", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "total_datasets" in data["data"]
    assert "total_records_processed" in data["data"]

    # Metrics
    res_m = client.get("/api/dashboard/metrics", headers=headers)
    assert res_m.status_code == 200
    assert res_m.json()["success"] is True

    # Model Performance
    res_mp = client.get("/api/dashboard/model-performance", headers=headers)
    assert res_mp.status_code == 200
    assert res_mp.json()["success"] is True

    # Recent Activity
    res_ra = client.get("/api/dashboard/recent-activity", headers=headers)
    assert res_ra.status_code == 200
    assert res_ra.json()["success"] is True
    assert isinstance(res_ra.json()["data"], list)

    # Data Summary
    res_ds = client.get("/api/dashboard/data-summary", headers=headers)
    assert res_ds.status_code == 200
    assert res_ds.json()["success"] is True


# =========================================================
# 2. Dataset APIs & Protection Tests
# =========================================================

def test_dataset_upload_preview_summary_and_deletion(client, auth_tokens, mock_db):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    # 1. Viewer upload is forbidden (RBAC)
    csv_content = b"customer_id,baseline_price,category\nC1,25.0,Electronics\nC2,40.0,Fashion\n"
    res_viewer = client.post(
        "/api/datasets/upload",
        files={"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")},
        headers=viewer_hdr,
    )
    assert res_viewer.status_code == 403

    # 2. Analyst upload succeeds
    res_upload = client.post(
        "/api/datasets/upload",
        files={"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")},
        headers=analyst_hdr,
    )
    assert res_upload.status_code == 201
    upload_data = res_upload.json()
    assert upload_data["success"] is True
    dataset_id = upload_data["data"]["id"]

    # 3. Preview dataset
    res_prev = client.get(f"/api/datasets/{dataset_id}/preview", headers=viewer_hdr)
    assert res_prev.status_code == 200
    assert res_prev.json()["data"]["rows_previewed"] == 2

    # 4. Summary stats
    res_sum = client.get(f"/api/datasets/{dataset_id}/summary", headers=viewer_hdr)
    assert res_sum.status_code == 200
    assert res_sum.json()["data"]["total_columns"] == 3

    # 5. Core dataset deletion protection
    datasets_list = client.get("/api/datasets", headers=viewer_hdr).json()
    core_ds = next((d for d in datasets_list if "retail_data" in d.get("name", "")), None)
    if core_ds:
        del_core = client.delete(f"/api/datasets/{core_ds['id']}", headers=analyst_hdr)
        assert del_core.status_code == 400
        assert "Cannot delete core system datasets" in del_core.json()["detail"]

    # 6. Delete uploaded dataset succeeds
    res_del = client.delete(f"/api/datasets/{dataset_id}", headers=analyst_hdr)
    assert res_del.status_code == 200
    assert res_del.json()["data"]["deleted"] is True


def test_invalid_dataset_upload_rejected(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])

    # Non-CSV file rejected
    res_txt = client.post(
        "/api/datasets/upload",
        files={"file": ("bad_file.txt", io.BytesIO(b"hello world"), "text/plain")},
        headers=analyst_hdr,
    )
    assert res_txt.status_code == 400

    # Empty file rejected
    res_empty = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        headers=analyst_hdr,
    )
    assert res_empty.status_code == 400


# =========================================================
# 3. ML Training APIs
# =========================================================

def test_training_apis(client, auth_tokens, monkeypatch):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    # Mock full ML pipeline run to avoid repeating long 5-minute training in test
    import backend.ml_pipeline as ml_mod
    monkeypatch.setattr(ml_mod, "run_full_ml_pipeline", lambda *args, **kwargs: {"status": "mock_completed", "roc_auc": 0.88})

    # Start training
    res = client.post(
        "/api/training/start",
        json={"model_type": "all", "version": "v1.0.1"},
        headers=analyst_hdr,
    )
    assert res.status_code == 202
    job = res.json()["data"]
    run_id = job["run_id"]

    # Viewer cannot start training (RBAC)
    assert client.post("/api/training/start", json={}, headers=viewer_hdr).status_code == 403

    # Check status
    res_status = client.get(f"/api/training/status/{run_id}", headers=viewer_hdr)
    assert res_status.status_code == 200
    assert res_status.json()["data"]["run_id"] == run_id

    # Check history
    res_hist = client.get("/api/training/history", headers=viewer_hdr)
    assert res_hist.status_code == 200
    assert len(res_hist.json()["data"]) >= 1

    # Check metrics and logs
    res_metrics = client.get(f"/api/training/{run_id}/metrics", headers=viewer_hdr)
    assert res_metrics.status_code == 200

    res_logs = client.get(f"/api/training/{run_id}/logs", headers=viewer_hdr)
    assert res_logs.status_code == 200
    assert isinstance(res_logs.json()["data"], list)


# =========================================================
# 4. Causal Inference APIs
# =========================================================

def test_causal_apis(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    # Analyze
    res = client.post(
        "/api/causal/analyze",
        json={"treatment_col": "treatment", "outcome_col": "purchase"},
        headers=analyst_hdr,
    )
    assert res.status_code == 200
    causal_data = res.json()["data"]
    analysis_id = causal_data["analysis_id"]
    assert "estimated_ate" in causal_data["summary"]

    # Viewer forbidden from running analysis
    assert client.post("/api/causal/analyze", json={}, headers=viewer_hdr).status_code == 403

    # Results
    res_res = client.get(f"/api/causal/results/{analysis_id}", headers=viewer_hdr)
    assert res_res.status_code == 200

    # History
    res_hist = client.get("/api/causal/history", headers=viewer_hdr)
    assert res_hist.status_code == 200
    assert len(res_hist.json()["data"]) >= 1

    # Diagnostics
    res_diag = client.get(f"/api/causal/diagnostics/{analysis_id}", headers=viewer_hdr)
    assert res_diag.status_code == 200


# =========================================================
# 5. Uplift Modeling APIs
# =========================================================

def test_uplift_apis(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    # Analyze
    res = client.post(
        "/api/uplift/analyze",
        json={"treatment_col": "treatment", "outcome_col": "purchase"},
        headers=analyst_hdr,
    )
    assert res.status_code == 200
    uplift_data = res.json()["data"]
    analysis_id = uplift_data["analysis_id"]
    assert "best_model" in uplift_data

    # Viewer forbidden
    assert client.post("/api/uplift/analyze", json={}, headers=viewer_hdr).status_code == 403

    # Results
    res_res = client.get(f"/api/uplift/results/{analysis_id}", headers=viewer_hdr)
    assert res_res.status_code == 200

    # History
    res_hist = client.get("/api/uplift/history", headers=viewer_hdr)
    assert res_hist.status_code == 200

    # Customer segments
    res_seg = client.get("/api/uplift/customer-segments", headers=viewer_hdr)
    assert res_seg.status_code == 200

    # Performance
    res_perf = client.get("/api/uplift/performance", headers=viewer_hdr)
    assert res_perf.status_code == 200


# =========================================================
# 6. Dynamic Pricing APIs
# =========================================================

def test_dynamic_pricing_apis(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    candidate = {
        "customer_id": "CUST_TEST_01",
        "category": "Electronics",
        "current_price": 50.0,
        "customer_history": 120.0,
        "seasonality": 1.0,
        "treatment": 0,
    }

    # Single Recommendation
    res_single = client.post("/api/pricing/recommend", json=candidate, headers=analyst_hdr)
    assert res_single.status_code == 200
    rec = res_single.json()["data"]
    assert "recommended_price" in rec
    assert "arc_elasticity" in rec

    # Batch Recommendation
    batch_req = {
        "candidates": [
            candidate,
            {
                "customer_id": "CUST_TEST_02",
                "category": "Fashion",
                "current_price": 75.0,
                "customer_history": 80.0,
            },
        ]
    }
    res_batch = client.post("/api/pricing/batch-recommend", json=batch_req, headers=analyst_hdr)
    assert res_batch.status_code == 200
    batch_data = res_batch.json()["data"]
    assert batch_data["summary"]["count"] == 2
    assert len(batch_data["recommendations"]) == 2

    # Viewer cannot run pricing
    assert client.post("/api/pricing/recommend", json=candidate, headers=viewer_hdr).status_code == 403

    # Elasticity summary
    res_e = client.get("/api/pricing/elasticity", headers=viewer_hdr)
    assert res_e.status_code == 200

    # Pricing performance
    res_p = client.get("/api/pricing/performance", headers=viewer_hdr)
    assert res_p.status_code == 200


# =========================================================
# 7. Promotion Optimization APIs
# =========================================================

def test_promotion_optimization_apis(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    opt_req = {
        "candidates": [
            {"customer_id": "CUST_OPT_1", "baseline_price": 100.0, "predicted_ite": 0.12},
            {"customer_id": "CUST_OPT_2", "baseline_price": 50.0, "predicted_ite": 0.18},
            {"customer_id": "CUST_OPT_3", "baseline_price": 80.0, "predicted_ite": -0.05},
        ],
        "budget": 20.0,
        "discount_pct": 10.0,
    }

    # Run Optimization
    res = client.post("/api/promotions/optimize", json=opt_req, headers=analyst_hdr)
    assert res.status_code == 200
    opt_data = res.json()["data"]
    assert opt_data["status"] == "optimal"
    assert opt_data["allocated_budget"] <= 20.0

    # Viewer forbidden from running optimization
    assert client.post("/api/promotions/optimize", json=opt_req, headers=viewer_hdr).status_code == 403

    # Campaigns list
    res_c = client.get("/api/promotions/campaigns", headers=viewer_hdr)
    assert res_c.status_code == 200

    # Results list
    res_r = client.get("/api/promotions/results", headers=viewer_hdr)
    assert res_r.status_code == 200

    # Budget analysis
    res_b = client.get("/api/promotions/budget-analysis", headers=viewer_hdr)
    assert res_b.status_code == 200


# =========================================================
# 8. End-to-End Pipeline Orchestration APIs
# =========================================================

def test_pipeline_orchestration_apis(client, auth_tokens):
    analyst_hdr = auth_header(auth_tokens["analyst"])
    viewer_hdr = auth_header(auth_tokens["viewer"])

    # Run Pipeline
    pipe_req = {"budget": 3000.0}
    res = client.post("/api/pipeline/run", json=pipe_req, headers=analyst_hdr)
    assert res.status_code == 202
    pipe_data = res.json()["data"]
    pipeline_run_id = pipe_data["pipeline_run_id"]
    assert len(pipe_data["stages"]) == 7

    # Viewer forbidden from starting pipeline
    assert client.post("/api/pipeline/run", json={}, headers=viewer_hdr).status_code == 403

    # Check pipeline status
    res_st = client.get(f"/api/pipeline/status/{pipeline_run_id}", headers=viewer_hdr)
    assert res_st.status_code == 200
    assert res_st.json()["data"]["pipeline_run_id"] == pipeline_run_id

    # Check pipeline history
    res_hist = client.get("/api/pipeline/history", headers=viewer_hdr)
    assert res_hist.status_code == 200
    assert len(res_hist.json()["data"]) >= 1

    # Check pipeline results
    res_res = client.get(f"/api/pipeline/results/{pipeline_run_id}", headers=viewer_hdr)
    assert res_res.status_code == 200

