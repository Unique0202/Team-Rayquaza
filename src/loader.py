"""
Load one CSV in the HCL Annex B schema into the warehouse (Member B).

    python -m src.loader --table oil_tests --file data/late_drop/oil_tests_late.csv

Used by the API's POST /ingest, the UI's upload screen, and the judges' test data.
1. check the columns and every row; bad rows are counted and left out, never silently "fixed"
2. save the good rows to data/ingested/<table>__<timestamp>.csv - bronze reads them from there, so
   the uploaded data is kept exactly as received
3. re-run the whole pipeline; if validation fails, the file is removed again and the warehouse rebuilt
"""

import argparse
import json
from datetime import datetime

import pandas as pd

from src import pipeline as P

POSITIVE_COLUMNS = {"oil_tests": P.GAS_COLUMNS, "costs_budget": ["cost", "annual_budget"]}
NUMERIC_COLUMNS = {"operation": ["load_pct", "top_oil_temp_c", "age_years"], "customers_served": ["customers"]}
DATE_COLUMNS = {"oil_tests": "test_date", "operation": "month"}


def find_bad_rows(df: pd.DataFrame, table: str, known_ids: set) -> pd.Series:
    """One reason per row ("" = the row is fine). Columns are already in Annex B names."""
    reason = pd.Series("", index=df.index)
    blank = df.apply(lambda col: col.astype(str).str.strip() == "")

    if "transformer_id" in df.columns:
        ids = df["transformer_id"].astype(str).str.strip().str.upper()
        reason[blank["transformer_id"]] = "missing transformer_id"
        # operation and oil tests must belong to a transformer we know (the asset list)
        if table in ("operation", "oil_tests") and known_ids:
            reason[(reason == "") & ~ids.isin(known_ids)] = "unknown transformer_id"

    if table in DATE_COLUMNS:
        col = DATE_COLUMNS[table]
        dates = pd.to_datetime(df[col], errors="coerce")
        reason[(reason == "") & dates.isna()] = f"unreadable {col}"
        reason[(reason == "") & (dates > pd.Timestamp.today())] = f"{col} is in the future"

    for col in POSITIVE_COLUMNS.get(table, []):
        values = pd.to_numeric(df[col], errors="coerce")
        reason[(reason == "") & ~(values > 0)] = f"{col} missing, unreadable or not positive"

    for col in NUMERIC_COLUMNS.get(table, []):   # blanks are allowed (gaps), garbage is not
        values = pd.to_numeric(df[col], errors="coerce")
        reason[(reason == "") & values.isna() & ~blank[col]] = f"{col} is not a number"

    if table == "costs_budget":
        reason[(reason == "") & ~df["action"].str.strip().str.lower().isin(["replace", "inspect"])] = "unknown action"
    return reason


def load_csv(table: str, csv_path: str) -> dict:
    """Returns {"rows_loaded", "rows_rejected", "status": "ok" | "error", "detail"} - the /ingest response."""
    if table not in P.REQUIRED_COLUMNS:
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "error",
                "detail": f"unknown table '{table}' - expected one of {sorted(P.REQUIRED_COLUMNS)}"}
    try:
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "error", "detail": f"cannot read CSV: {exc}"}

    checked = raw.rename(columns=P.COLUMN_ALIASES)    # a renamed column (late drop) -> its Annex B name
    missing = [col for col in P.REQUIRED_COLUMNS[table] if col not in checked.columns]
    if missing:
        return {"rows_loaded": 0, "rows_rejected": len(raw), "status": "error",
                "detail": f"missing required column(s) {missing}; add a COLUMN_ALIASES entry in src/pipeline.py "
                          f"if a column was renamed"}

    try:
        with P.get_connection() as conn:
            known_ids = set(pd.read_sql("SELECT transformer_id FROM silver_transformer", conn)["transformer_id"])
    except Exception:
        known_ids = set()                              # no warehouse yet: the pipeline's validation still checks IDs

    reason = find_bad_rows(checked, table, known_ids)
    good = raw[reason == ""]                           # save the ORIGINAL columns: bronze keeps data as received
    rejected = reason[reason != ""]
    detail = "; ".join(f"{n} x {r}" for r, n in rejected.value_counts().items()) or "all rows accepted"
    if good.empty:
        return {"rows_loaded": 0, "rows_rejected": len(rejected), "status": "error", "detail": detail}

    ingested_dir = P.DATA_DIR / "ingested"
    ingested_dir.mkdir(parents=True, exist_ok=True)
    saved = ingested_dir / f"{table}__{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.csv"
    good.to_csv(saved, index=False)

    try:
        P.run_pipeline()
    except AssertionError as exc:                      # the batch broke a validation check: undo it
        saved.unlink()
        P.run_pipeline()
        return {"rows_loaded": 0, "rows_rejected": len(raw), "status": "error",
                "detail": f"rolled back - validation failed after loading: {exc}"}

    return {"rows_loaded": len(good), "rows_rejected": len(rejected), "status": "ok",
            "detail": f"{detail}; saved as {saved.name}; warehouse rebuilt"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a CSV in the Annex B schema and rebuild the warehouse.")
    parser.add_argument("--table", required=True, choices=sorted(P.REQUIRED_COLUMNS))
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    print(json.dumps(load_csv(args.table, args.file), indent=2))
