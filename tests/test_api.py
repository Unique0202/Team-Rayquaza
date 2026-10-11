"""Member A: contract tests for the Part 2 stub API; no warehouse/model required."""

import csv
import io

from fastapi.testclient import TestClient
import pytest

from src import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Tiny private inputs keep tests independent of teammates' generated data."""
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    tables = {
        "customers_served": (
            ["transformer_id", "customer_type", "customers", "region", "rating_class"],
            [["TR-0001", "hospital", 10000, "North", "medium"],
             ["TR-0002", "rural", 400, "West", "small"],
             ["tr-0001 ", "hospital", 10000, "North", "medium"]],
        ),
        "operation": (
            [*api.REQUIRED_COLUMNS["operation"], "event"],
            [["TR-0001", "2023-10-01", 75, 60, 25, "none"],
             ["TR-0002", "2023-10-01", 75, 60, 25, "none"]],
        ),
        "oil_tests": (api.REQUIRED_COLUMNS["oil_tests"],
                      [["TR-0001", "2026-09-01", 40, 30, 0.5, 15, 20, 300]]),
        "costs_budget": (api.REQUIRED_COLUMNS["costs_budget"],
                         [["inspect", "small", 10000, 5000000], ["inspect", "medium", 15000, 5000000]]),
    }
    for table, (headers, rows) in tables.items():
        with (tmp_path / f"{table}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
    with TestClient(api.app) as test_client:
        yield test_client


def oil_upload(unit="TR-0001"):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(api.REQUIRED_COLUMNS["oil_tests"])
    writer.writerow([unit, "2026-10-05", 40, 30, 0.5, 15, 20, 300])
    return {"file": ("oil.csv", stream.getvalue().encode(), "text/csv")}


def test_predict_contract_and_no_test_unit(client):
    response = client.post("/predict", json={})
    assert response.status_code == 200
    body = response.json()
    assert {"model_version", "as_of", "horizon_months", "predictions"} <= body.keys()
    assert body["mode"] == "stub" and body["horizon_months"] == 12
    assert body["as_of"] == "2026-10-01" and len(body["predictions"]) == 2
    required = {"transformer_id", "status", "failure_prob_12m", "prob_interval_90", "health_score",
                "impact_weight", "risk_score", "top_drivers", "data_quality", "model_version"}
    for prediction in body["predictions"]:
        assert required <= prediction.keys()
        low, high = prediction["prob_interval_90"]
        assert 0 <= low <= prediction["failure_prob_12m"] <= high <= 1
        assert prediction["risk_score"] == pytest.approx(prediction["failure_prob_12m"] * prediction["impact_weight"])
        assert "STUB" in prediction["data_quality"]
    untested = body["predictions"][1]
    assert untested["status"] == "scored_low_confidence" and untested["health_score"] is None


def test_predict_optional_body_and_empty_selection(client):
    assert len(client.post("/predict").json()["predictions"]) == 2
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
    response = client.post("/plan", json={"budget": 4000000})
    assert response.status_code == 200
    body = response.json()
    assert {"status", "actions", "total_cost", "budget", "risk_before", "risk_removed",
            "risk_reduced_pct", "age_based_plan", "model_version"} <= body.keys()
    assert body["status"] == "plan" and body["mode"] == "stub"
    assert body["actions"] and body["total_cost"] <= body["budget"]
    assert body["total_cost"] == sum(action["cost"] for action in body["actions"])
    for action in body["actions"]:
        assert action["requires_approval"] is True and action["action"] == "inspect"
    assert "STUB" in body["detail"]
    assert body["age_based_plan"]["actions"] == []


@pytest.mark.parametrize("budget", [0, 1, 9999])
def test_plan_tiny_budget(client, budget):
    response = client.post("/plan", json={"budget": budget})
    assert response.status_code == 200
    assert response.json()["actions"] == [] and response.json()["total_cost"] == 0


def test_plan_cost_overrides(client):
    body = client.post("/plan", json={"budget": 500, "costs": {"inspect": {"medium": 500}}}).json()
    assert body["total_cost"] == 500
    assert body["actions"][0]["transformer_id"] == "TR-0001"


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
def test_ingest_contract_and_no_side_effects(client, table):
    before = {p.name: p.read_bytes() for p in api.DATA_DIR.iterdir()}
    response = client.post("/ingest", data={"table": table}, files=oil_upload())
    assert response.status_code == 200
    body = response.json()
    assert {"rows_loaded", "rows_rejected", "status", "detail"} <= body.keys()
    assert body["rows_loaded"] == 0 and body["rows_rejected"] == 0
    assert body["rows_checked"] == 1 and body["mode"] == "stub" and body["status"] == "ok"
    assert before == {p.name: p.read_bytes() for p in api.DATA_DIR.iterdir()}


def test_ingest_unknown_id(client):
    response = client.post("/ingest", data={"table": "oil_tests"}, files=oil_upload("TR-9999"))
    assert response.status_code == 200
    assert response.json()["status"] == "error" and response.json()["rows_rejected"] == 1


@pytest.mark.parametrize("table", ["unknown", '{"table":[]}', "{broken}"])
def test_ingest_unknown_table(client, table):
    response = client.post("/ingest", data={"table": table}, files=oil_upload())
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


@pytest.mark.parametrize("content", [b"", b"wrong,columns\na,b\n", b"\xff\xfe",
                                     (",".join(api.REQUIRED_COLUMNS["oil_tests"]) + "\n").encode()])
def test_ingest_invalid_csv(client, content):
    response = client.post("/ingest", data={"table": "oil_tests"}, files={"file": ("bad.csv", content, "text/csv")})
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"


def test_ingest_requires_file_and_table(client):
    response = client.post("/ingest", data={"table": "oil_tests"})
    assert response.status_code == 422 and response.json()["status"] == "invalid_input"
    assert client.post("/ingest", files=oil_upload()).status_code == 422


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert {"model_version", "algorithm", "features", "training_window", "validation_method", "metrics"} <= body.keys()
    assert body["mode"] == "stub" and body["metrics"] == {}


def test_explain(client):
    response = client.get("/explain/TR-0001")
    assert response.status_code == 200
    body = response.json()
    assert {"transformer_id", "top_drivers", "oil_tests", "operation", "model_version"} <= body.keys()
    assert body["transformer_id"] == "TR-0001" and body["mode"] == "stub"
    assert client.get("/explain/TR-9999").json()["status"] == "invalid_input"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert {"status", "api", "data_store", "model"} <= body.keys()
    assert body["status"] == "ok" and body["mode"] == "stub"
    assert body["model"]["loaded"] is False and body["ready_for_real_predictions"] is False


def test_missing_source_is_service_unavailable(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "missing")
    assert client.get("/health").status_code == 503
    response = client.post("/predict", json={})
    assert response.status_code == 503 and response.json()["status"] == "error"


def test_openapi_and_docs(client):
    assert client.get("/docs").status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert set(paths) == {"/predict", "/plan", "/ingest", "/model/info", "/explain/{transformer_id}", "/health"}
    assert "multipart/form-data" in paths["/ingest"]["post"]["requestBody"]["content"]
