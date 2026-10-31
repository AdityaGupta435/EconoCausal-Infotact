
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
                {
                    "customer_id": "CUST001",
                    "baseline_price": 100,
                    "predicted_ite": 0.08,
                },
                {
                    "customer_id": "CUST002",
                    "baseline_price": 200,
                    "predicted_ite": 0.12,
                },
                {
                    "customer_id": "CUST003",
                    "baseline_price": 150,
                    "predicted_ite": -0.01,
                },
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

    assert len(body["allocation"]) == 3

    assert body["allocation"][0]["customer_id"] == "CUST001"
    assert body["allocation"][1]["customer_id"] == "CUST002"
    assert body["allocation"][2]["customer_id"] == "CUST003"

    assert body["allocation"][0]["selected"] is True
    assert body["allocation"][1]["selected"] is True
    assert body["allocation"][2]["selected"] is False

    assert body["allocation"][2]["status"] == "Not Selected"

def test_csv_upload_success():
    csv_content = (
        "baseline_price,customer_history,seasonality,category,treatment,purchase\n"
        "100,5,1,A,1,1\n"
        "200,3,0,B,0,0\n"
    )

    response = client.post(
        "/api/data/upload",
        files={"file": ("dataset.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 2
    assert "baseline_price" in body["columns"]
    assert body["saved_to_disk"] is False


def test_csv_upload_rejects_non_csv_file():
    response = client.post(
        "/api/data/upload",
        files={"file": ("dataset.txt", "sample data", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are allowed."


def test_csv_upload_rejects_missing_required_columns():
    csv_content = "baseline_price,category\n100,A\n"

    response = client.post(
        "/api/data/upload",
        files={"file": ("dataset.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 422
    assert "missing_columns" in response.json()["detail"]