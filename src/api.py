"""Member A: real serving through B's features/loader and C's model/planner.

Run the pipeline first, obtain C's saved model, then run:
python -m uvicorn src.api:app --reload
Serving never trains or substitutes a fake model.
"""

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from threading import RLock
from typing import Literal, Optional
from uuid import uuid4

import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import model as M, pipeline as P
from src.features import build_features
from src.loader import load_csv
from src.plan import age_plan, compute_risk, make_plan


DEFAULT_AS_OF = "2026-10-01"
# A single API worker serializes scoring/planning with warehouse rebuilds during ingest.
WAREHOUSE_LOCK = RLock()
RISK_COLUMNS = ["transformer_id", "failure_prob_12m", "prob_low", "prob_high", "status",
                "health_score", "impact_weight", "risk_score", "risk_low", "risk_high", "model_version"]


def iso_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Use a date in YYYY-MM-DD format")
    return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
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
    run_id: str


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
    model_version: str
    as_of: str
    run_id: str
    score_run_id: str
    age_based_plan: PlanSummary
    detail: str


app = FastAPI(title="Transformer Fleet Health API", version="0.3.0",
              description="Shared snapshot features, saved model probabilities and budget planning.")


@app.exception_handler(RequestValidationError)
async def invalid_body(request: Request, exc: RequestValidationError):
    # Raw Pydantic errors may contain ValueError objects or NaN; expose only safe fields.
    errors = [{"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"status": "invalid_input", "detail": errors})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    status = "invalid_input" if exc.status_code < 500 else "error"
    return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                        content={"status": status, "detail": exc.detail})


@app.exception_handler(SQLAlchemyError)
async def database_error(request: Request, exc: SQLAlchemyError):
    return JSONResponse(status_code=503, content={"status": "error",
                        "detail": "Warehouse unavailable; run python -m src.pipeline and retry."})


def require_warehouse():
    if not Path(P.DB_PATH).is_file():
        raise HTTPException(503, "Warehouse missing; run python -m src.pipeline first.")


def get_bundle():
    try:
        bundle = M.load_model()
        if not bundle.get("model_version") or not callable(getattr(bundle.get("model"), "predict_interval", None)):
            raise ValueError("Saved model has an unexpected structure")
        return bundle
    except Exception as exc:
        raise HTTPException(503, "Saved model unavailable or incompatible; obtain C's model.joblib and matching dependencies.") from exc


def records(frame: pd.DataFrame) -> list[dict]:
    # pandas converts missing numeric values to JSON null instead of invalid JSON NaN.
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def score_fleet(conn, as_of: str, bundle: dict):
    """Use exactly the shared training-time features and C's scoring/risk functions."""
    first = conn.execute(text("SELECT MIN(month) FROM silver_operation")).scalar()
    if first and as_of < first:
        raise HTTPException(422, f"as_of must be on or after {first}")
    features = build_features(conn, [as_of])
    if features.empty:
        raise HTTPException(503, "No transformers available to score")
    dimension = pd.read_sql("SELECT * FROM dim_transformer", conn)
    risk = compute_risk(M.score(features, bundle=bundle), dimension)
    # The API contract requires insufficient_data whenever no operation precedes this snapshot.
    observed = set(conn.execute(text("SELECT DISTINCT transformer_id FROM silver_operation WHERE month < :as_of"),
                                {"as_of": as_of}).scalars())
    missing = ~risk.transformer_id.isin(observed)
    risk.loc[missing, "status"] = "insufficient_data"
    risk.loc[missing, ["failure_prob_12m", "prob_low", "prob_high", "risk_score", "risk_low", "risk_high", "health_score"]] = float("nan")
    risk.loc[missing, "data_quality"] = "No operation rows before the snapshot"
    return risk, features


def prediction_rows(risk):
    output = []
    for row in records(risk):
        interval = None if row["prob_low"] is None else [row["prob_low"], row["prob_high"]]
        output.append({key: row[key] for key in ["transformer_id", "status", "failure_prob_12m", "health_score",
                                                "impact_weight", "risk_score", "top_drivers", "data_quality", "model_version"]}
                      | {"prob_interval_90": interval})
    return output


def write_run(conn, kind, as_of, version, budget=None):
    run_id = f"{kind}-{uuid4().hex}"
    # Run metadata and its fact rows are appended in the caller's single transaction.
    conn.execute(text("""INSERT INTO dim_run (run_id, created_at, kind, as_of, budget, model_version)
                         VALUES (:run_id, :created_at, :kind, :as_of, :budget, :model_version)"""),
                 {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "kind": kind,
                  "as_of": as_of, "budget": budget, "model_version": version})
    return run_id


def save_risk(conn, risk, as_of, version):
    run_id = write_run(conn, "score", as_of, version)
    rows = records(risk[RISK_COLUMNS])
    for row in rows:
        row["run_id"] = run_id
    columns = ["run_id", *RISK_COLUMNS]
    sql = f"INSERT INTO fact_risk ({', '.join(columns)}) VALUES ({', '.join(':' + c for c in columns)})"
    conn.execute(text(sql), rows)
    return run_id


def source_modified_at():
    # Read source-file timestamps, not warehouse mtime (history inserts change the latter).
    paths = list(P.DATA_DIR.glob("*.csv")) + list((P.DATA_DIR / "ingested").glob("*.csv"))
    return max((path.stat().st_mtime for path in paths), default=0)


@app.post("/predict", response_model=PredictResponse)
def predict(request: Optional[PredictRequest] = Body(default=None)):
    request = request or PredictRequest()
    as_of = request.as_of or DEFAULT_AS_OF
    with WAREHOUSE_LOCK:
        require_warehouse()
        with P.engine.begin() as conn:
            known = set(conn.execute(text("SELECT transformer_id FROM dim_transformer")).scalars())
            unknown = sorted(set(request.transformer_ids or []) - known)
            if unknown:
                raise HTTPException(422, {"message": "Unknown transformer IDs", "transformer_ids": unknown})
            bundle = get_bundle()
            risk, _ = score_fleet(conn, as_of, bundle)
            # Save the full fleet even for a filtered response, so /plan never plans a partial fleet by accident.
            run_id = save_risk(conn, risk, as_of, bundle["model_version"])
            if request.transformer_ids is not None:
                risk = risk.set_index("transformer_id").loc[request.transformer_ids].reset_index()
            response = {"model_version": bundle["model_version"], "as_of": as_of, "horizon_months": 12,
                        "predictions": prediction_rows(risk), "run_id": run_id}
            return PredictResponse.model_validate(response)


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
    with WAREHOUSE_LOCK:
        require_warehouse()
        bundle = get_bundle()
        with P.engine.begin() as conn:
            # Only full API scoring runs use kind='score'; UI previews have their own history kind.
            latest = conn.execute(text("SELECT * FROM dim_run WHERE kind = 'score' ORDER BY created_at DESC LIMIT 1")).mappings().first()
            if latest is None:
                raise HTTPException(409, "Call /predict before requesting a plan.")
            created = datetime.fromisoformat(latest["created_at"])
            if created.tzinfo is None:
                created = created.astimezone()
            if source_modified_at() > created.timestamp() or latest["model_version"] != bundle["model_version"]:
                raise HTTPException(409, "Data or model changed; call /predict again before /plan.")
            risk = pd.read_sql(text("SELECT * FROM fact_risk WHERE run_id = :run_id"), conn,
                               params={"run_id": latest["run_id"]})
            features = build_features(conn, [latest["as_of"]])
            dimension = pd.read_sql("SELECT transformer_id, rating_class FROM dim_transformer", conn)
            if set(risk.transformer_id) != set(dimension.transformer_id):
                raise HTTPException(409, "Latest score does not cover this fleet; call /predict again.")
            # Eligibility fields are rebuilt by B's function at the SAME snapshot as the saved scores.
            risk = risk.merge(features[["transformer_id", "age_years", "has_oil_test", "months_since_last_test"]],
                              on="transformer_id", validate="one_to_one").merge(dimension, on="transformer_id", validate="one_to_one")
            costs = pd.read_sql("SELECT * FROM silver_costs", conn)
            for action, sizes in (request.costs or {}).items():
                for size, cost in sizes.items():
                    selected = (costs.action == action) & (costs.rating_class == size)
                    if not selected.any():
                        raise HTTPException(422, f"No configured cost for {action}/{size}")
                    costs.loc[selected, "cost"] = cost
            result = make_plan(risk, costs, request.budget)
            baseline = age_plan(risk, costs, request.budget)
            for proposal in [result, baseline]:
                for action in proposal["actions"]:
                    action["requires_approval"] = True
            run_id = write_run(conn, "plan", latest["as_of"], latest["model_version"], request.budget)
            action_keys = pd.read_sql("SELECT action_key, action, rating_class FROM dim_action", conn)
            key_lookup = action_keys.set_index(["action", "rating_class"]).action_key.to_dict()
            sizes = dimension.set_index("transformer_id").rating_class.to_dict()
            rows = []
            for action in result["actions"]:
                rows.append({"run_id": run_id, "transformer_id": action["transformer_id"],
                             "action_key": int(key_lookup[(action["action"], sizes[action["transformer_id"]])]),
                             **{key: action[key] for key in ["action", "cost", "risk_before", "risk_removed"]}})
            if rows:
                conn.execute(text("""INSERT INTO fact_plan (run_id, transformer_id, action_key, action, cost, risk_before, risk_removed)
                                     VALUES (:run_id, :transformer_id, :action_key, :action, :cost, :risk_before, :risk_removed)"""), rows)
            return PlanResponse.model_validate({**result, "status": "plan", "run_id": run_id,
                      "score_run_id": latest["run_id"], "as_of": latest["as_of"], "model_version": latest["model_version"],
                      "age_based_plan": baseline,
                      "detail": "Replacement value is risk removed; inspection value is uncertain risk that an inspection may reveal."})


@app.post("/ingest")
def ingest(file: UploadFile = File(...), table: str = Form(...)):
    # Support the team's plain table field and HCL's JSON-in-form spelling.
    try:
        if table.lstrip().startswith("{"):
            try:
                table = json.loads(table)["table"]
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(422, 'Use a table name or {"table": "oil_tests"}.') from exc
        if not isinstance(table, str) or table not in P.REQUIRED_COLUMNS:
            raise HTTPException(422, "Unknown source table")
        with WAREHOUSE_LOCK, tempfile.TemporaryDirectory(prefix="rayquaza-ingest-") as directory:
            require_warehouse()
            # Ignore the client filename; keep the temporary path under our control.
            path = Path(directory) / "upload.csv"
            with path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    target.write(chunk)
            result = load_csv(table, str(path))
            return JSONResponse(status_code=200 if result["status"] == "ok" else 422, content=result)
    finally:
        file.file.close()


@app.get("/model/info")
def model_info():
    bundle = get_bundle()
    try:
        card = M.model_info()
    except (OSError, ValueError) as exc:
        raise HTTPException(503, "Model card missing or unreadable; obtain C's model_card.json.") from exc
    # C's file follows HCL Annex C: {"model_card": {...}} - read the card inside the wrapper.
    card = card.get("model_card", card)
    # C's card uses 'version'; expose the API's required model_version too.
    if card.get("version", card.get("model_version")) != bundle["model_version"]:
        raise HTTPException(503, "Model card and saved model versions differ.")
    return {**card, "model_version": bundle["model_version"]}


@app.get("/explain/{transformer_id}")
def explain(transformer_id: str):
    unit = transformer_id.strip().upper()
    with WAREHOUSE_LOCK:
        require_warehouse()
        with P.get_connection() as conn:
            if conn.execute(text("SELECT 1 FROM dim_transformer WHERE transformer_id = :unit"), {"unit": unit}).first() is None:
                raise HTTPException(422, "Unknown transformer ID")
            latest = conn.execute(text("""SELECT d.as_of, d.created_at, r.model_version FROM dim_run d JOIN fact_risk r ON d.run_id = r.run_id
                                          WHERE r.transformer_id = :unit ORDER BY d.created_at DESC LIMIT 1"""), {"unit": unit}).mappings().first()
            as_of = latest["as_of"] if latest else DEFAULT_AS_OF
            bundle = get_bundle()
            if latest:
                created = datetime.fromisoformat(latest["created_at"])
                if created.tzinfo is None:
                    created = created.astimezone()
                if source_modified_at() > created.timestamp() or latest["model_version"] != bundle["model_version"]:
                    raise HTTPException(409, "Data or model changed; call /predict again before requesting its explanation.")
            risk, _ = score_fleet(conn, as_of, bundle)
            risk = risk.loc[risk.transformer_id == unit]
            # Parameterized queries filter out observations on/after the scoring date.
            params = {"unit": unit, "as_of": as_of}
            oil = pd.read_sql(text("SELECT * FROM silver_oil_test WHERE transformer_id = :unit AND test_date < :as_of ORDER BY test_date"), conn, params=params)
            operation = pd.read_sql(text("SELECT * FROM silver_operation WHERE transformer_id = :unit AND month < :as_of ORDER BY month"), conn, params=params)
            return {**prediction_rows(risk)[0], "as_of": as_of, "oil_tests": records(oil), "operation": records(operation)}


@app.get("/health")
def health():
    database, model = {"status": "unavailable"}, {"loaded": False}
    with WAREHOUSE_LOCK:
        try:
            require_warehouse()
            with P.get_connection() as conn:
                for table in ["dim_transformer", "silver_operation", "silver_oil_test", "silver_costs",
                              "dim_run", "fact_risk", "fact_plan", "dim_action"]:
                    conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                count = conn.execute(text("SELECT COUNT(*) FROM dim_transformer")).scalar()
                if not count:
                    raise ValueError("Fleet is empty")
            database = {"status": "ok", "warehouse_connected": True, "transformers": count}
        except (HTTPException, SQLAlchemyError, ValueError):
            database["detail"] = "Warehouse missing, empty or incomplete; run the pipeline."
        try:
            card = model_info()
            model = {"loaded": True, "model_version": card["model_version"]}
        except HTTPException as exc:
            model["detail"] = exc.detail
    ready = database["status"] == "ok" and model["loaded"]
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ok" if ready else "not_ready", "api": "ok", "data_store": database,
                                 "model": model, "ready_for_real_predictions": ready})
