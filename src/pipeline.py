"""
Transformer Fleet Health - medallion pipeline (Member B).

    python -m src.pipeline                              # data in data/
    DATA_DIR=path/to/other/folder python -m src.pipeline  # same pipeline on another folder of CSVs

    bronze : every CSV exactly as received (+ _ingested_at, _source)
    silver : cleaned, typed, one row per business key, validated
    gold   : star schema + fact_snapshot (step 2)

Every run rebuilds every table from the files on disk, so running it twice gives identical tables.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DB_PATH = ROOT / "warehouse.db"
REPORTS_DIR = ROOT / "reports"

engine = create_engine(f"sqlite:///{DB_PATH}")

# HCL Annex B: the columns each source file MUST have. Extra columns are allowed.
REQUIRED_COLUMNS = {
    "customers_served": ["transformer_id", "customer_type", "customers", "region"],
    "operation": ["transformer_id", "month", "load_pct", "top_oil_temp_c", "age_years"],
    "oil_tests": ["transformer_id", "test_date",
                  "h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm", "co_ppm"],
    "costs_budget": ["action", "rating_class", "cost", "annual_budget"],
}
GAS_COLUMNS = REQUIRED_COLUMNS["oil_tests"][2:]

# The late data drop may rename a column. Map {"new_name_in_file": "annex_b_name"} here.
# This is configuration, not code: HCL 4.6 says a renamed column must load without code changes.
COLUMN_ALIASES: dict[str, str] = {}

CUSTOMER_TYPES = {"hospital", "industry", "urban", "rural"}
DEFAULT_RATING_CLASS = "medium"   # judges' files have no rating_class (it is our extra column)
UNIT_JUMP_RATIO = 200             # a test >= 200x the unit's other tests was logged in ppb, not ppm
IMPLAUSIBLE_TOTAL_PPM = 50_000    # single-test units have no history to compare; no real fault gets near this


def get_connection():
    """A connection to the warehouse - callers use it in a `with` block."""
    return engine.connect()


def write_tables(tables: dict) -> None:
    """Replace a group of tables so that their DATA lands all-or-nothing.

    pandas' to_sql runs CREATE TABLE outside the surrounding transaction (proved on this machine,
    hackathon-kit/PREFLIGHT.md section 2). So step 1 creates every table empty, and step 2 appends
    all the rows inside ONE transaction: if any insert fails, no rows land in any table.
    """
    with engine.begin() as conn:
        for name, df in tables.items():
            df.head(0).to_sql(name, conn, if_exists="replace", index=False)
    with engine.begin() as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="append", index=False)


# ----------------------------------------------------------------------------
# BRONZE - raw, exactly as received, never edited
# ----------------------------------------------------------------------------
def source_files(table: str) -> list[Path]:
    """The base file first, then every file /ingest saved (oldest first - the timestamp is in the name)."""
    base = DATA_DIR / f"{table}.csv"
    assert base.exists(), f"missing source file: {base}"
    ingested = sorted((DATA_DIR / "ingested").glob(f"{table}__*.csv"))
    return [base] + ingested


def load_bronze() -> dict:
    """Read every source file as plain text and stack them, adding where each row came from.

    Bronze is the replay tape: if a later step has a bug we fix it and rebuild from here,
    without needing the original source again.
    """
    bronze = {}
    for table in REQUIRED_COLUMNS:
        frames = []
        for order, path in enumerate(source_files(table)):
            # dtype=str + keep_default_na=False keeps every value exactly as written, blanks included
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df["_ingested_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            df["_source"] = path.name
            df["_load_order"] = order   # later files win when the same key appears twice
            frames.append(df)
        bronze[table] = pd.concat(frames, ignore_index=True)

    write_tables({f"bronze_{table}": df for table, df in bronze.items()})
    for table, df in bronze.items():
        print(f"  bronze       : {len(df):>6,} rows -> bronze_{table}")
    return bronze


# ----------------------------------------------------------------------------
# SILVER - cleaned, typed, one row per business key
# ----------------------------------------------------------------------------
def prepare(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """First steps shared by every table: blanks -> missing, renamed columns -> Annex B names,
    required columns present, transformer IDs tidied ("tr-0045 " -> "TR-0045")."""
    df = raw.replace("", np.nan)
    for file_name, annex_name in COLUMN_ALIASES.items():
        if file_name not in df.columns:
            continue
        if annex_name in df.columns:   # old and new files stacked: fill the gaps from the renamed column
            df[annex_name] = df[annex_name].fillna(df[file_name])
        else:
            df[annex_name] = df[file_name]
        df = df.drop(columns=[file_name])

    missing = [col for col in REQUIRED_COLUMNS[table] if col not in df.columns]
    assert not missing, f"{table}: required column(s) missing: {missing}"

    if "transformer_id" in df.columns:
        df["transformer_id"] = df["transformer_id"].str.strip().str.upper()
    return df


def keep_latest(df: pd.DataFrame, key: list[str]) -> tuple[pd.DataFrame, int]:
    """One row per business key. If a key appears more than once, the row loaded last wins.

    kind="stable" matters: the default sort may reorder rows with equal _load_order,
    which would make "last" arbitrary.
    """
    before = len(df)
    df = df.sort_values("_load_order", kind="stable").drop_duplicates(subset=key, keep="last")
    return df, before - len(df)


def fix_unit_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """Find oil tests logged in ppb instead of ppm (edge case E4) and convert them back to ppm.

    A real fault makes gases climb over several tests. A unit change makes ONE test ~1000x its
    neighbours. So we compare each test's total gas with the median total of the same
    transformer's OTHER tests: 200x or more means a unit jump, and we divide that test by 1000.
    """
    df = df.sort_values(["transformer_id", "test_date"]).reset_index(drop=True)
    total = df[GAS_COLUMNS].sum(axis=1)
    is_jump = pd.Series(False, index=df.index)

    for _, rows in df.groupby("transformer_id").groups.items():
        for i in rows:
            others = total[rows.drop(i)]
            if len(others) > 0:
                is_jump[i] = total[i] >= UNIT_JUMP_RATIO * others.median()
            else:
                is_jump[i] = total[i] >= IMPLAUSIBLE_TOTAL_PPM

    df.loc[is_jump, GAS_COLUMNS] = df.loc[is_jump, GAS_COLUMNS] / 1000
    df["unit_fixed"] = is_jump.astype(int)
    return df


def build_silver(bronze: dict) -> tuple[dict, dict]:
    """Turn the four bronze tables into five clean silver tables. Gaps stay missing (NULL):
    how to fill them is a modelling decision, and a missing oil test is itself information."""
    fixes = {}

    # --- silver_transformer: the asset list, one row per transformer ---
    t = prepare(bronze["customers_served"], "customers_served")
    t["customer_type"] = t["customer_type"].str.strip().str.lower()
    t["customers"] = pd.to_numeric(t["customers"], errors="coerce")
    if "rating_class" not in t.columns:            # extra column - judges' files will not have it
        t["rating_class"] = np.nan
    t["rating_class_assumed"] = t["rating_class"].isna().astype(int)
    t["rating_class"] = t["rating_class"].fillna(DEFAULT_RATING_CLASS).astype(str).str.strip().str.lower()
    t = t[t["transformer_id"].notna()]
    t, fixes["duplicate assets removed"] = keep_latest(t, ["transformer_id"])
    t = t[["transformer_id", "customer_type", "customers", "region", "rating_class", "rating_class_assumed"]]

    # --- silver_operation: one row per transformer per month ---
    op = prepare(bronze["operation"], "operation")
    op["month"] = pd.to_datetime(op["month"], errors="coerce")
    for col in ["load_pct", "top_oil_temp_c", "age_years"]:
        op[col] = pd.to_numeric(op[col], errors="coerce")   # unreadable value -> missing, never a crash
    if "event" not in op.columns:                  # extra column - judges' files will not have it
        op["event"] = "none"
    op["event"] = op["event"].fillna("none").astype(str).str.strip().str.lower()
    op = op[op["transformer_id"].notna() & op["month"].notna()]
    op, fixes["duplicate operation rows removed"] = keep_latest(op, ["transformer_id", "month"])
    op["month"] = op["month"].dt.strftime("%Y-%m-%d")
    op = op.sort_values(["transformer_id", "month"])
    op = op[["transformer_id", "month", "load_pct", "top_oil_temp_c", "age_years", "event"]]

    # --- silver_oil_test: one row per transformer per test date, all in ppm ---
    oil = prepare(bronze["oil_tests"], "oil_tests")
    oil["test_date"] = pd.to_datetime(oil["test_date"], errors="coerce")
    for gas in GAS_COLUMNS:
        oil[gas] = pd.to_numeric(oil[gas], errors="coerce")
    oil = oil[oil["transformer_id"].notna() & oil["test_date"].notna()]
    oil, fixes["duplicate oil tests removed"] = keep_latest(oil, ["transformer_id", "test_date"])
    oil = fix_unit_jumps(oil)
    fixes["oil tests converted ppb -> ppm"] = int(oil["unit_fixed"].sum())
    oil["test_date"] = oil["test_date"].dt.strftime("%Y-%m-%d")
    oil = oil[["transformer_id", "test_date", *GAS_COLUMNS, "unit_fixed"]]

    # --- silver_costs + silver_budget ---
    c = prepare(bronze["costs_budget"], "costs_budget")
    c["action"] = c["action"].str.strip().str.lower()
    c["rating_class"] = c["rating_class"].str.strip().str.lower()
    c["cost"] = pd.to_numeric(c["cost"], errors="coerce")
    c["annual_budget"] = pd.to_numeric(c["annual_budget"], errors="coerce")
    costs, _ = keep_latest(c[c["action"].notna() & c["rating_class"].notna()], ["action", "rating_class"])
    costs = costs[["action", "rating_class", "cost"]]
    # The budget comes from the most recently loaded costs file - the late drop changes it.
    latest_file = c[c["_load_order"] == c["_load_order"].max()]
    budget = pd.DataFrame({"annual_budget": [latest_file["annual_budget"].dropna().iloc[0]]})

    fixes["transformers with rating_class assumed"] = int(t["rating_class_assumed"].sum())
    fixes["missing load / temperature values (left as NULL)"] = int(op[["load_pct", "top_oil_temp_c"]].isna().sum().sum())

    silver = {
        "silver_transformer": t.reset_index(drop=True),
        "silver_operation": op.reset_index(drop=True),
        "silver_oil_test": oil.reset_index(drop=True),
        "silver_costs": costs.reset_index(drop=True),
        "silver_budget": budget,
    }
    return silver, fixes


# ----------------------------------------------------------------------------
# VALIDATE - HCL 4.5 checks, run on silver BEFORE it is written or analysed
# ----------------------------------------------------------------------------
def validate(silver: dict, bronze: dict, fixes: dict) -> None:
    """Run HCL's validation checks, write them to reports/validation_output.txt, and stop the run
    if any fails - bad data must fail loudly here, not quietly corrupt the model."""
    t, op, oil = silver["silver_transformer"], silver["silver_operation"], silver["silver_oil_test"]
    costs, budget = silver["silver_costs"], silver["silver_budget"]
    asset_ids = set(t["transformer_id"])
    today = pd.Timestamp.today().normalize()
    results = []   # (passed, check, detail)

    unknown_op = sorted(set(op["transformer_id"]) - asset_ids)
    results.append((not unknown_op, "every transformer in operation is in the asset list", f"unknown: {unknown_op[:5]}"))
    unknown_oil = sorted(set(oil["transformer_id"]) - asset_ids)
    results.append((not unknown_oil, "every transformer in oil_tests is in the asset list", f"unknown: {unknown_oil[:5]}"))

    non_positive = int((oil[GAS_COLUMNS] <= 0).sum().sum())
    results.append((non_positive == 0, "gas results are positive", f"{non_positive} values <= 0"))
    jumps_left = int(fix_unit_jumps(oil.assign(test_date=pd.to_datetime(oil["test_date"])))["unit_fixed"].sum())
    results.append((jumps_left == 0, "gas results are in ppm (no unit jumps left)", f"{jumps_left} still look like ppb"))

    future_tests = int((pd.to_datetime(oil["test_date"]) > today).sum())
    results.append((future_tests == 0, "test dates are not in the future", f"{future_tests} future tests"))
    future_months = int((pd.to_datetime(op["month"]) > today).sum())
    results.append((future_months == 0, "operation months are not in the future", f"{future_months} future months"))

    bad_types = int((~t["customer_type"].isin(CUSTOMER_TYPES)).sum())
    results.append((bad_types == 0, "every transformer has a valid customer type", f"{bad_types} missing/invalid"))

    costs_ok = bool((costs["cost"] > 0).all() and (budget["annual_budget"] > 0).all())
    results.append((costs_ok, "costs and budget are positive numbers", f"budget = {budget['annual_budget'].iloc[0]:,.0f}"))

    card_path = DATA_DIR / "data_card.json"
    if card_path.exists():
        card_counts = json.loads(card_path.read_text())["row_counts"]
        mismatches = []
        for table, df in bronze.items():
            file_name = f"{table}.csv"
            actual = int((df["_source"] == file_name).sum())   # base file only; ingested files are extra
            if file_name in card_counts and card_counts[file_name] != actual:
                mismatches.append(f"{file_name}: card {card_counts[file_name]} vs file {actual}")
        results.append((not mismatches, "row counts match the data card", "; ".join(mismatches) or "all match"))
    else:
        results.append((True, "row counts match the data card", "SKIPPED - no data_card.json (e.g. judges' data)"))

    lines = [f"Validation run {datetime.now().isoformat(timespec='seconds')} on {DATA_DIR}", "",
             "Cleaning applied before the checks:"]
    lines += [f"  - {name}: {count}" for name, count in fixes.items()]
    lines += ["", "Checks:"]
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {check}  ({detail})" for ok, check, detail in results]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "validation_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [check for ok, check, _ in results if not ok]
    print(f"  validate     : {len(results) - len(failed)}/{len(results)} checks passed "
          f"-> reports/validation_output.txt")
    assert not failed, f"validation failed: {failed}"


# ----------------------------------------------------------------------------
def run_pipeline() -> dict:
    """Rebuild the whole warehouse from the files on disk. Returns row counts per table."""
    print(f"\nPipeline: {DATA_DIR} -> bronze -> silver -> gold\n")
    bronze = load_bronze()
    silver, fixes = build_silver(bronze)
    validate(silver, bronze, fixes)          # checks run BEFORE silver is written
    write_tables(silver)
    for name, df in silver.items():
        print(f"  silver       : {len(df):>6,} rows -> {name}")

    counts = {f"bronze_{table}": len(df) for table, df in bronze.items()}
    counts.update({name: len(df) for name, df in silver.items()})
    return counts


if __name__ == "__main__":
    run_pipeline()
    print("\nDone. Run it again - the tables come out identical.\n")
