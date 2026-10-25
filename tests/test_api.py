
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "EconoCausal"
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_causal_ate_endpoint(monkeypatch):
    import backend.main as api

    monkeypatch.setattr(
        api,
        "estimate_ate",
        lambda data: 0.08,
    )

    response = client.get("/api/causal/ate")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "DML baseline"
    assert body["training_rows"] > 0
    assert body["estimated_ate"] == 0.08
    assert body["estimated_effect_percentage_points"] == 8.0


def test_uplift_endpoint_returns_report(monkeypatch, tmp_path):
    import json
    import backend.main as api

    report_path = tmp_path / "uplift_evaluation.json"
    expected = {
        "evaluation": "synthetic held-out test set",
        "test_rows": 2000,
    }
    report_path.write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setattr(api, "UPLIFT_REPORT", report_path)

    response = client.get("/api/uplift")

    assert response.status_code == 200
    assert response.json() == expected


def test_optimization_endpoint():
    response = client.post(
        "/api/optimization",
        json={
            "candidates": [
                {"baseline_price": 100, "predicted_ite": 0.08},
                {"baseline_price": 200, "predicted_ite": 0.12},
                {"baseline_price": 150, "predicted_ite": -0.01},
            ],
            "budget": 30,
            "discount_pct": 10,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_indices"] == [0, 1]
    assert body["selected_count"] == 2
    assert body["total_cost"] == 30.0
    assert body["expected_incremental_purchases"] == 0.2
