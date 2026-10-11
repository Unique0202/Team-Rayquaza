"""Member A integration tests: real B/C functions, with a test-only model bundle.

No trained model or held-out evaluation is used. The small deterministic model
below exists only in tests; the production API has no stub/fallback path.
"""

import csv
import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, text

from src import api


class TestModel:
    __test__ = False

    def predict_interval(self, features):
        p = (0.05 + features.h2_last.fillna(0).to_numpy() * 0.0005).clip(0, 0.8)
        width = np.where(features.has_oil_test == 1, 0.10, 0.25)
        return p, np.maximum(0, p - 0.03), np.minimum(1, p + width)

    def contributions(self, features):
        return [[{"feature": "h2_last", "contribution": 0.02}] for _ in range(len(features))]


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'warehouse.db'}")
    monkeypatch.setattr(api.P, "DATA_DIR", data)
    monkeypatch.setattr(api.P, "DB_PATH", tmp_path / "warehouse.db")
    monkeypatch.setattr(api.P, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(api.P, "engine", engine)
    bundle = {"model": TestModel(), "model_version": "test-only-v1"}
    monkeypatch.setattr(api.M, "load_model", lambda: bundle)
    monkeypatch.setattr(api.M, "model_info", lambda: {"version": "test-only-v1", "algorithm": "test fixture"})
    assets = [["TR-0001", "hospital", 10000, "North", "medium"],
              ["TR-0002", "rural", 400, "West", "small"],
              ["TR-0003", "urban", 2000, "East", "small"]]
    operation = []
    for month in pd.date_range("2023-10-01", "2026-09-01", freq="MS"):
        for unit in ["TR-0001", "TR-0002"]:
            operation.append([unit, month.strftime("%Y-%m-%d"), 75, 60, 25, "none"])
    tables = {
        "customers_served": ([*api.P.REQUIRED_COLUMNS["customers_served"], "rating_class"], assets),
        "operation": ([*api.P.REQUIRED_COLUMNS["operation"], "event"], operation),
        "oil_tests": (api.P.REQUIRED_COLUMNS["oil_tests"],
                      [["TR-0001", "2025-07-01", 30, 20, 0.5, 10, 15, 250],
                       ["TR-0001", "2026-09-01", 40, 30, 0.5, 15, 20, 300],
                       ["TR-0003", "2026-09-01", 40, 30, 0.5, 15, 20, 300]]),
        "costs_budget": (api.P.REQUIRED_COLUMNS["costs_budget"],
                         [[action, size, cost, 5000000]
                          for action, values in [("replace", [150000,350000,700000]), ("inspect", [10000,15000,25000])]
                          for size, cost in zip(["small", "medium", "large"], values)]),
    }
    for table, (headers, rows) in tables.items():
        with (data / f"{table}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
    api.P.run_pipeline()
    with TestClient(api.app) as test_client:
        yield test_client
    engine.dispose()


def oil_upload(unit="TR-0001"):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(api.P.REQUIRED_COLUMNS["oil_tests"])
    writer.writerow([unit, "2026-10-05", 180, 30, 0.5, 15, 20, 300])
    return {"file": ("oil.csv", stream.getvalue().encode(), "text/csv")}


def test_predict_contract_and_no_test_unit(client):
    response = client.post("/predict", json={})
    assert response.status_code == 200
    body = response.json()
    assert {"model_version", "as_of", "horizon_months", "predictions"} <= body.keys()
    assert "mode" not in body and body["horizon_months"] == 12 and body["run_id"]
    assert body["as_of"] == "2026-10-01" and len(body["predictions"]) == 3
    required = {"transformer_id", "status", "failure_prob_12m", "prob_interval_90", "health_score",
                "impact_weight", "risk_score", "top_drivers", "data_quality", "model_version"}
    for prediction in body["predictions"]:
        assert required <= prediction.keys()
        if prediction["status"] == "insufficient_data":
            assert prediction["failure_prob_12m"] is None and prediction["prob_interval_90"] is None
            continue
        low, high = prediction["prob_interval_90"]
        assert 0 <= low <= prediction["failure_prob_12m"] <= high <= 1
        assert prediction["risk_score"] == pytest.approx(prediction["failure_prob_12m"] * prediction["impact_weight"])
        assert "STUB" not in prediction["data_quality"]
    untested = body["predictions"][1]
    assert untested["status"] == "scored_low_confidence" and untested["health_score"] is None


def test_predict_optional_body_and_empty_selection(client):
    assert len(client.post("/predict").json()["predictions"]) == 3
    assert client.post("/predict", json={"transformer_ids": []}).json()["predictions"] == []


def test_predict_selected_ids_and_as_of(client):
    response = client.post("/predict", json={"transformer_ids": [" tr-0002 ", "TR-0002"], "as_of": "2026-10-11"})
    assert response.status_code == 200
    assert response.json()["as_of"] == "2026-10-11"
    assert [p["transformer_id"] for p in response.json()["predictions"]] == ["TR-0002"]


@pytest.mark.parametrize("body", [
    {"transformer_ids": ["TR-9999"]}, {"transformer_ids": [""]},
    {"transformer_ids": "TR-0001"}, {"transformer_ids": [123]},
    {"as_of": "2023-09-30"}, {"as_of": "2026-02-30"}, {"as_of": "11/10/2026"},
    {"as_of": 20261001}, {"unexpected": True}, ["TR-0001"],
])
def test_predict_invalid_input(client, body):
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert response.json()["status"] == "invalid_input"


@pytest.mark.parametrize("path", ["/predict", "/plan"])
def test_malformed_json(client, path):
    response = client.post(path, content='{"broken":', headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


def test_plan_contract_and_approval(client):
    client.post("/predict", json={})
    response = client.post("/plan", json={"budget": 4000000})
    assert response.status_code == 200
    body = response.json()
    assert {"status", "actions", "total_cost", "budget", "risk_before", "risk_removed",
            "risk_reduced_pct", "age_based_plan", "model_version"} <= body.keys()
    assert body["status"] == "plan" and body["run_id"] and body["score_run_id"]
    assert body["actions"] and body["total_cost"] <= body["budget"]
    assert body["total_cost"] == sum(action["cost"] for action in body["actions"])
    for action in body["actions"]:
        assert action["requires_approval"] is True and action["action"] in {"inspect", "replace"}
    assert body["age_based_plan"]["total_cost"] <= body["budget"]
    for action in body["age_based_plan"]["actions"]:
        assert action["requires_approval"] is True


@pytest.mark.parametrize("budget", [0, 1, 9999])
def test_plan_tiny_budget(client, budget):
    client.post("/predict", json={})
    response = client.post("/plan", json={"budget": budget})
    assert response.status_code == 200
    assert response.json()["actions"] == [] and response.json()["total_cost"] == 0


def test_plan_cost_overrides(client):
    client.post("/predict", json={})
    body = client.post("/plan", json={"budget": 5000, "costs": {"inspect": {"medium": 5000}}}).json()
    assert body["total_cost"] == 5000
    assert body["actions"][0]["transformer_id"] == "TR-0001"
    with api.P.get_connection() as conn:
        assert conn.execute(text("SELECT cost FROM silver_costs WHERE action='inspect' AND rating_class='medium'")).scalar() == 15000


@pytest.mark.parametrize("body", [
    {}, {"budget": -1}, {"budget": True}, {"budget": "5000"},
    {"budget": 1, "costs": {"inspect": {"small": 0}}},
    {"budget": 1, "costs": {"inspect": {"small": True}}},
    {"budget": 1, "costs": {"inspect": {"giant": 5}}},
    {"budget": 1, "costs": {"fix": {"small": 5}}},
])
def test_plan_invalid_input(client, body):
    response = client.post("/plan", json=body)
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_plan_nonfinite_numbers(client, value):
    response = client.post("/plan", content='{"budget":' + value + '}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


@pytest.mark.parametrize("table", ["oil_tests", '{"table":"oil_tests"}'])
def test_ingest_rebuilds_and_preserves_history(client, table):
    before = client.post("/predict", json={"as_of": "2026-10-11"}).json()
    response = client.post("/ingest", data={"table": table}, files=oil_upload())
    assert response.status_code == 200
    body = response.json()
    assert body["rows_loaded"] == 1 and body["rows_rejected"] == 0 and body["status"] == "ok"
    assert len(list((api.P.DATA_DIR / "ingested").glob("*.csv"))) == 1
    with api.P.get_connection() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM fact_risk WHERE run_id=:run_id"), {"run_id":before["run_id"]}).scalar() == 3
        assert conn.execute(text("SELECT COUNT(*) FROM silver_oil_test WHERE test_date='2026-10-05'")).scalar() == 1
    assert client.post("/plan", json={"budget":4000000}).status_code == 409
    assert client.get("/explain/TR-0001").status_code == 409
    after = client.post("/predict", json={"as_of":"2026-10-11"}).json()
    assert after["run_id"] != before["run_id"]
    assert after["predictions"][0]["failure_prob_12m"] > before["predictions"][0]["failure_prob_12m"]
    assert client.post("/plan", json={"budget":4000000}).status_code == 200


def test_ingest_unknown_id(client):
    response = client.post("/ingest", data={"table": "oil_tests"}, files=oil_upload("TR-9999"))
    assert response.status_code == 422
    assert response.json()["status"] == "error" and response.json()["rows_rejected"] == 1


@pytest.mark.parametrize("table", ["unknown", '{"table":[]}', "{broken}"])
def test_ingest_unknown_table(client, table):
    response = client.post("/ingest", data={"table": table}, files=oil_upload())
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


@pytest.mark.parametrize("content", [b"", b"wrong,columns\na,b\n", b"\xff\xfe",
                                     (",".join(api.P.REQUIRED_COLUMNS["oil_tests"]) + "\n").encode()])
def test_ingest_invalid_csv(client, content):
    response = client.post("/ingest", data={"table": "oil_tests"}, files={"file": ("bad.csv", content, "text/csv")})
    assert response.status_code == 422 and response.json()["status"] == "error"


def test_ingest_requires_file_and_table(client):
    response = client.post("/ingest", data={"table": "oil_tests"})
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"
    assert client.post("/ingest", files=oil_upload()).status_code == 422


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-only-v1"
    assert response.json()["algorithm"] == "test fixture"


def test_explain(client):
    client.post("/predict", json={"as_of":"2026-09-01"})
    response = client.get("/explain/TR-0001")
    assert response.status_code == 200
    body = response.json()
    assert {"transformer_id", "top_drivers", "oil_tests", "operation", "model_version"} <= body.keys()
    assert body["as_of"] == "2026-09-01"
    assert all(row["test_date"] < body["as_of"] for row in body["oil_tests"])
    assert all(row["month"] < body["as_of"] for row in body["operation"])
    assert client.get("/explain/TR-9999").json()["status"] == "invalid_input"


def test_explain_agrees_for_unit_without_operation(client):
    prediction = client.post("/predict", json={"transformer_ids":["TR-0003"]}).json()["predictions"][0]
    explanation = client.get("/explain/TR-0003").json()
    assert explanation["status"] == prediction["status"] == "insufficient_data"
    assert explanation["failure_prob_12m"] is None


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"]["loaded"] is True
    assert body["ready_for_real_predictions"] is True


def test_missing_warehouse(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api.P, "DB_PATH", tmp_path / "missing.db")
    assert client.get("/health").status_code == 503
    response = client.post("/predict", json={})
    assert response.status_code == 503 and response.json()["status"] == "error"


def test_missing_model(client, monkeypatch):
    def unavailable():
        raise FileNotFoundError("test")
    monkeypatch.setattr(api.M, "load_model", unavailable)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={}).status_code == 503
    assert client.get("/model/info").status_code == 503


def test_model_card_version_mismatch(client, monkeypatch):
    monkeypatch.setattr(api.M, "model_info", lambda: {"version":"wrong"})
    assert client.get("/health").status_code == 503


def test_plan_requires_scores(client):
    assert client.post("/plan", json={"budget": 5000000}).status_code == 409


def test_filtered_prediction_saves_full_fleet_and_plan(client):
    score = client.post("/predict", json={"transformer_ids":["TR-0001"]}).json()
    assert len(score["predictions"]) == 1
    plan = client.post("/plan", json={"budget":5000000}).json()
    assert plan["score_run_id"] == score["run_id"]
    with api.P.get_connection() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM fact_risk WHERE run_id=:id"), {"id": score["run_id"]}).scalar() == 3
        assert conn.execute(text("SELECT COUNT(*) FROM fact_plan WHERE run_id=:id"), {"id":plan["run_id"]}).scalar() == len(plan["actions"])
    assert all(a["transformer_id"] != "TR-0003" for a in plan["actions"])


def test_transaction_rolls_back_failed_score_write(client, monkeypatch):
    original = api.save_risk
    def broken(conn, risk, as_of, version):
        original(conn, risk, as_of, version)
        raise RuntimeError("simulated failure after inserts")
    monkeypatch.setattr(api, "save_risk", broken)
    with pytest.raises(RuntimeError):
        client.post("/predict",json={})
    with api.P.get_connection() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM dim_run")).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM fact_risk")).scalar() == 0


def test_openapi_and_docs(client):
    assert client.get("/docs").status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) == {"/predict", "/plan", "/ingest", "/model/info", "/explain/{transformer_id}", "/health"}
    assert "multipart/form-data" in paths["/ingest"]["post"]["requestBody"]["content"]
