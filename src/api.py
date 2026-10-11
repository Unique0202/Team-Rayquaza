"""Member A, Part 2: contract-first API with explicitly labelled stub results.

Run: python -m uvicorn src.api:app --reload
Part 3 replaces the stub_* helpers with B/C's shared functions. This module
does not train a model, rebuild the warehouse, or persist uploaded data yet.
"""

import csv
from datetime import date
import io
import json
import math
from pathlib import Path
from typing import Literal, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_AS_OF = "2026-10-01"
MODEL_VERSION = "stub-part2-v1"
TYPE_WEIGHTS = {"hospital": 5, "industry": 3, "urban": 2, "rural": 1}
GASES = ["h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm", "co_ppm"]
# Local Annex B schema for the upload stub; B's loader owns real validation in Part 3.
REQUIRED_COLUMNS = {
    "customers_served": ["transformer_id", "customer_type", "customers", "region"],
    "operation": ["transformer_id", "month", "load_pct", "top_oil_temp_c", "age_years"],
    "oil_tests": ["transformer_id", "test_date", *GASES],
    "costs_budget": ["action", "rating_class", "cost", "annual_budget"],
}


def iso_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Use a date in YYYY-MM-DD format")
    return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    # Optional keeps the same JSON contract on both this laptop's Python 3.9 and Docker's 3.13.
    transformer_ids: Optional[list[str]] = None
    as_of: Optional[str] = None

    @field_validator("as_of")
    @classmethod
    def validate_date(cls, value):
        return iso_date(value) if value is not None else None

    @field_validator("transformer_ids")
    @classmethod
    def normalize_ids(cls, value):
        if value is None:
            return value
        normalized = [unit.strip().upper() for unit in value]
        if any(not unit for unit in normalized):
            raise ValueError("Transformer IDs must not be blank")
        # One response per transformer, preserving the order requested.
        return list(dict.fromkeys(normalized))


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    budget: float = Field(ge=0, allow_inf_nan=False)
    costs: Optional[dict[str, dict[str, float]]] = Field(
        default=None, description='Optional overrides, e.g. {"inspect": {"medium": 20000}}'
    )

    @field_validator("costs")
    @classmethod
    def validate_costs(cls, value):
        for action, sizes in (value or {}).items():
            if action not in {"replace", "inspect"}:
                raise ValueError("Costs must use replace or inspect")
            for size, cost in sizes.items():
                if size not in {"small", "medium", "large"}:
                    raise ValueError("Rating class must be small, medium or large")
                if not math.isfinite(cost) or cost <= 0:
                    raise ValueError("Costs must be finite positive numbers")
        return value


class Driver(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    transformer_id: str
    status: Literal["scored", "scored_low_confidence", "insufficient_data"]
    failure_prob_12m: Optional[float] = Field(ge=0, le=1)
    prob_interval_90: Optional[tuple[float, float]]
    health_score: Optional[float] = Field(ge=0, le=100)
    impact_weight: float
    risk_score: Optional[float]
    top_drivers: list[Driver]
    data_quality: str
    model_version: str


class PredictResponse(BaseModel):
    model_version: str
    as_of: str
    horizon_months: Literal[12] = 12
    predictions: list[Prediction]
    mode: Literal["stub"] = "stub"


class PlanAction(BaseModel):
    transformer_id: str
    action: Literal["replace", "inspect"]
    cost: float
    risk_before: float
    risk_removed: float
    rationale: str
    requires_approval: Literal[True] = True


class PlanSummary(BaseModel):
    actions: list[PlanAction]
    total_cost: float
    budget: float
    risk_before: float
    risk_removed: float
    risk_reduced_pct: float


class PlanResponse(PlanSummary):
    status: Literal["plan"] = "plan"
    model_version: str = MODEL_VERSION
    as_of: str = DEFAULT_AS_OF
    mode: Literal["stub"] = "stub"
    age_based_plan: PlanSummary
    detail: str


app = FastAPI(
    title="Transformer Fleet Health API",
    version="0.2.0",
    description="Part 2 contract preview. All model/plan results are stubs. Uploads are not persisted.",
)


@app.exception_handler(RequestValidationError)
async def invalid_body(request: Request, exc: RequestValidationError):
    # Pydantic's raw errors can contain ValueError objects or NaN; expose only JSON-safe fields.
    errors = [{"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"status": "invalid_input", "detail": errors})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    status = "invalid_input" if exc.status_code < 500 else "error"
    return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                        content={"status": status, "detail": exc.detail})


def read_source(table: str) -> list[dict]:
    """Read source CSVs only for IDs and stub context; no warehouse or answer key access."""
    try:
        with (DATA_DIR / f"{table}.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise HTTPException(503, f"Source data unavailable: {table}.csv") from exc


def assets_by_id() -> dict[str, dict]:
    assets = {}
    for row in read_source("customers_served"):
        unit = row["transformer_id"].strip().upper()
        assets[unit] = {**row, "transformer_id": unit}
    if not assets:
        raise HTTPException(503, "Asset list is empty")
    return assets


def stub_predict(request: PredictRequest) -> dict:
    """Part 3: build_features -> score -> compute_risk, then persist fact_risk."""
    assets = assets_by_id()
    units = sorted(assets) if request.transformer_ids is None else request.transformer_ids
    unknown = [unit for unit in units if unit not in assets]
    if unknown:
        raise HTTPException(422, {"message": "Unknown transformer IDs", "transformer_ids": unknown})
    as_of = request.as_of or DEFAULT_AS_OF
    months = [row["month"] for row in read_source("operation")]
    if months and as_of < min(months):
        raise HTTPException(422, f"as_of must be on or after {min(months)}")
    predictions = []
    for unit in units:
        asset = assets[unit]
        impact = TYPE_WEIGHTS[asset["customer_type"]] * int(asset["customers"])
        # Fixed placeholders exercise the response shape; they are not model estimates.
        predictions.append({
            "transformer_id": unit, "status": "scored_low_confidence",
            "failure_prob_12m": 0.10, "prob_interval_90": [0.02, 0.30],
            "health_score": None, "impact_weight": impact, "risk_score": 0.10 * impact,
            "top_drivers": [{"feature": "age_years", "contribution": 0.0}],
            "data_quality": "STUB: illustrative probability and driver; no trained model or health assessment.",
            "model_version": MODEL_VERSION,
        })
    return {"model_version": MODEL_VERSION, "as_of": as_of, "horizon_months": 12,
            "predictions": predictions, "mode": "stub"}


def stub_plan(request: PlanRequest) -> dict:
    """Part 3: latest fact_risk -> C's make_plan + age_plan; persist fact_plan."""
    predictions = stub_predict(PredictRequest())["predictions"]
    assets = assets_by_id()
    prices = {}
    for row in read_source("costs_budget"):
        prices.setdefault(row["action"], {})[row["rating_class"]] = float(row["cost"])
    for action, overrides in (request.costs or {}).items():
        prices.setdefault(action, {}).update(overrides)
    risk_before = sum(row["risk_score"] for row in predictions)
    actions = []
    # One illustrative inspection, in ID order. No optimiser or model comparison is claimed.
    for row in predictions:
        size = assets[row["transformer_id"]].get("rating_class") or "medium"
        cost = prices["inspect"][size]
        if cost <= request.budget:
            actions.append({"transformer_id": row["transformer_id"], "action": "inspect",
                            "cost": cost, "risk_before": row["risk_score"],
                            "risk_removed": (row["prob_interval_90"][1] - row["failure_prob_12m"]) * row["impact_weight"],
                            "rationale": "STUB: sample inspection for testing the API contract.",
                            "requires_approval": True})
            break
    removed = sum(action["risk_removed"] for action in actions)
    baseline = {"actions": [], "total_cost": 0.0, "budget": request.budget,
                "risk_before": risk_before, "risk_removed": 0.0, "risk_reduced_pct": 0.0}
    return {"status": "plan", "mode": "stub", "model_version": MODEL_VERSION, "as_of": DEFAULT_AS_OF,
            "actions": actions, "total_cost": sum(action["cost"] for action in actions),
            "budget": request.budget, "risk_before": risk_before, "risk_removed": removed,
            "risk_reduced_pct": 100 * removed / risk_before if risk_before else 0.0,
            "age_based_plan": baseline,
            "detail": "STUB: illustrative plan and empty age-based comparison; neither optimiser has run."}


def stub_ingest(table: str, content: bytes) -> dict:
    """Part 3: save a temporary CSV and call B's load_csv(table, csv_path)."""
    # Also accept a JSON object in the form field for clients following HCL's wording.
    if table.lstrip().startswith("{"):
        try:
            metadata = json.loads(table)
            table = metadata["table"]
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, 'table must name a source table or contain {"table": "oil_tests"}') from exc
    if not isinstance(table, str) or table not in REQUIRED_COLUMNS:
        raise HTTPException(422, "Unknown table; use oil_tests, operation, customers_served or costs_budget")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        headers = reader.fieldnames or []
        missing = sorted(set(REQUIRED_COLUMNS[table]) - set(headers))
        if missing or len(headers) != len(set(headers)):
            raise HTTPException(422, {"message": "Missing or duplicate CSV headers", "missing_columns": missing})
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise HTTPException(422, "Upload must be a valid UTF-8 CSV") from exc
    if not rows:
        raise HTTPException(422, "CSV must contain at least one data row")
    known_ids = assets_by_id() if table in {"operation", "oil_tests"} else {}
    rejected = 0
    for row in rows:
        # These are shape/ID checks only. B's real loader will perform domain validation.
        invalid = None in row or any(row.get(column) is None for column in REQUIRED_COLUMNS[table])
        if table in {"operation", "oil_tests"}:
            invalid = invalid or (row.get("transformer_id") or "").strip().upper() not in known_ids
        rejected += int(invalid)
    return {"rows_loaded": 0, "rows_rejected": rejected, "status": "error" if rejected else "ok",
            "mode": "stub", "rows_checked": len(rows),
            "detail": f"STUB: checked headers and IDs for {len(rows)} rows; no data saved or pipeline run."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: Optional[PredictRequest] = Body(default=None)):
    return stub_predict(request or PredictRequest())


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
    return stub_plan(request)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), table: str = Form(...)):
    try:
        return stub_ingest(table, await file.read())
    finally:
        await file.close()


@app.get("/model/info")
def model_info():
    # Part 3 returns C's model_info() unchanged.
    return {"model_version": MODEL_VERSION, "mode": "stub", "algorithm": "stub (not trained)",
            "features": [], "training_window": None, "validation_method": "not run", "metrics": {},
            "horizon_months": 12, "detail": "Placeholder model card; no fitted model loaded."}


@app.get("/explain/{transformer_id}")
def explain(transformer_id: str):
    prediction = stub_predict(PredictRequest(transformer_ids=[transformer_id]))["predictions"][0]
    # Sensor detail is deliberately empty until B's shared data access is wired in Part 3.
    return {**prediction, "as_of": DEFAULT_AS_OF, "mode": "stub", "oil_tests": [], "operation": [],
            "detail": "STUB: illustrative driver; sensor history is not connected yet."}


@app.get("/health")
def health():
    missing = [table for table in REQUIRED_COLUMNS if not (DATA_DIR / f"{table}.csv").is_file()]
    result = {"status": "error" if missing else "ok", "mode": "stub", "api": "ok",
              "data_store": {"status": "unavailable" if missing else "source_csvs_available",
                             "warehouse_connected": False, "missing_tables": missing},
              "model": {"loaded": False, "model_version": MODEL_VERSION},
              "ready_for_real_predictions": False}
    return JSONResponse(status_code=503 if missing else 200, content=result)
