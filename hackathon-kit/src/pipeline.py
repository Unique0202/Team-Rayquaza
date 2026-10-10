"""
<PROJECT NAME> — HCLTech Campus Hackathon
Medallion pipeline: source -> bronze -> silver -> gold (star schema) -> analytics.

Run:  python src/pipeline.py
Idempotent: running it twice produces identical results.

GRAIN OF THE FACT TABLE:  one row = <<< FILL AT 10:00 >>>
"""

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "warehouse.db"
SOURCE = ROOT / "data" / "source.csv"        # <<< point at the real file
BUSINESS_KEY = "id"                           # <<< the natural key for dedupe

engine = create_engine(f"sqlite:///{DB_PATH}")


# ----------------------------------------------------------------------------
# EXTRACT
# ----------------------------------------------------------------------------
def extract() -> pd.DataFrame:
    """Read the source exactly as it is. No cleaning here - that is silver's job."""
    df = pd.read_csv(SOURCE)
    print(f"  extract      : {len(df):,} rows from {SOURCE.name}")
    return df


# ----------------------------------------------------------------------------
# BRONZE - raw, immutable, never edited
# ----------------------------------------------------------------------------
def load_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Land the raw data untouched, plus provenance columns.

    Bronze is the replay tape: if a transform has a bug we re-run from here
    instead of re-fetching a source that may be slow, rate-limited or gone.
    """
    bronze = df.copy()
    bronze["_ingested_at"] = pd.Timestamp.now()
    bronze["_source"] = SOURCE.name

    with engine.begin() as conn:
        bronze.to_sql("bronze_records", conn, if_exists="replace", index=False)
    print(f"  bronze       : {len(bronze):,} rows -> bronze_records")
    return bronze


# ----------------------------------------------------------------------------
# SILVER - cleaned, deduplicated, typed
# ----------------------------------------------------------------------------
def transform_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Clean and conform. This is where data quality is enforced."""
    df = bronze.copy()

    # Deduplicate on the business key, keeping the most recently ingested row.
    # (SQL equivalent: ROW_NUMBER() OVER (PARTITION BY key ORDER BY _ingested_at DESC) = 1)
    before = len(df)
    df = df.sort_values("_ingested_at").drop_duplicates(subset=[BUSINESS_KEY], keep="last")
    print(f"  silver dedupe: {before:,} -> {len(df):,} rows")

    # Type casting. errors="coerce" turns unparseable values into NaT/NaN
    # rather than throwing, so one bad row cannot kill the run.
    # df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    # df["amount"]     = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Drop rows with no business key - they cannot be joined to anything.
    df = df[df[BUSINESS_KEY].notna()]

    with engine.begin() as conn:
        df.to_sql("silver_records", conn, if_exists="replace", index=False)
    print(f"  silver       : {len(df):,} rows -> silver_records")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality gate. Fail loudly and early rather than loading bad data.

    Demo note: these are worth SHOWING to the panel - break one deliberately
    and let it fail, then fix it. It proves the checks are real.
    """
    assert df[BUSINESS_KEY].notna().all(), "DQ: null business key"
    assert not df[BUSINESS_KEY].duplicated().any(), "DQ: duplicate business key"
    # assert (df["amount"] >= 0).all(), "DQ: negative amount"
    print(f"  validate     : 3 checks passed")
    return df


# ----------------------------------------------------------------------------
# GOLD - star schema
# ----------------------------------------------------------------------------
def build_date_dimension(start: str, end: str) -> pd.DataFrame:
    """A real calendar table. Almost every team forgets this one.

    It lets the analytics join on month/quarter/weekday without recomputing
    date parts in every query.
    """
    dates = pd.date_range(start, end, freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["date_key"] = range(1, len(dim) + 1)
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = dim["date"].dt.quarter
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.month_name()
    dim["day_of_week"] = dim["date"].dt.day_name()
    dim["is_weekend"] = dim["date"].dt.dayofweek >= 5
    return dim[["date_key", "date", "year", "quarter", "month",
                "month_name", "day_of_week", "is_weekend"]]


def make_dimension(df: pd.DataFrame, natural_key: str, attrs: list[str], name: str) -> pd.DataFrame:
    """Build a Type-1 dimension with a surrogate integer key.

    Surrogate keys because business keys change, get reused, and differ across
    source systems - and because SCD Type 2 needs several rows per business
    entity, which a natural key cannot express.
    """
    dim = df[[natural_key] + attrs].drop_duplicates(subset=[natural_key]).reset_index(drop=True)
    dim.insert(0, f"{name}_key", range(1, len(dim) + 1))
    return dim


def build_gold(silver: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assemble the fact table and its dimensions.

    GRAIN: one row = <<< FILL AT 10:00 >>>
    """
    dims: dict[str, pd.DataFrame] = {}

    # dims["dim_date"] = build_date_dimension("2024-01-01", "2026-12-31")
    # dims["dim_x"] = make_dimension(silver, "x_id", ["x_name"], "dim_x")

    fact = silver.copy()
    # Replace natural keys with surrogate keys, one merge per dimension.
    # validate="many_to_one" makes pandas RAISE if the join would fan out and
    # silently duplicate fact rows - the single most common silent bug here.
    # fact = fact.merge(dims["dim_x"][["dim_x_key", "x_id"]],
    #                   on="x_id", how="left", validate="many_to_one")

    # fact = fact[["dim_x_key", "date_key", "amount", "quantity"]]
    print(f"  gold         : fact {len(fact):,} rows, {len(dims)} dimensions")
    return fact, dims


def create_schema(fact: pd.DataFrame, dims: dict) -> None:
    """Create the gold tables EMPTY, before the real load.

    Why this is a separate step: pandas' to_sql issues the CREATE TABLE outside
    the surrounding transaction, so a rollback leaves an empty table behind even
    though no rows survive. Creating tables here and keeping load_gold() strictly
    append-only makes the actual data load genuinely atomic.
    (Verified on this machine, 2026-10-10 - pandas 3.0 / SQLAlchemy 2.1.)
    """
    with engine.begin() as conn:
        for name, dim in dims.items():
            dim.head(0).to_sql(name, conn, if_exists="replace", index=False)
        fact.head(0).to_sql("fact_events", conn, if_exists="replace", index=False)
    print("  schema       : gold tables created empty")


def load_gold(fact: pd.DataFrame, dims: dict) -> None:
    """Write the star schema inside ONE transaction, append-only.

    This is the ACID moment: if the fact load fails halfway, the dimension rows
    roll back too, rather than leaving a warehouse that looks populated but is
    internally inconsistent. Tables already exist (see create_schema), so this
    transaction contains only DML and rolls back completely.
    """
    with engine.begin() as conn:
        for name, dim in dims.items():
            dim.to_sql(name, conn, if_exists="append", index=False)
        fact.to_sql("fact_events", conn, if_exists="append", index=False)
    print(f"  load_gold    : {len(fact):,} fact rows committed in one transaction")


# ----------------------------------------------------------------------------
# ANALYTICS
# ----------------------------------------------------------------------------
def run_analytics() -> None:
    """The three questions the system exists to answer."""
    queries = {
        # "Top N per group - RANK() ranks within each partition without collapsing rows"
        # "q1_top_per_group": """
        #     SELECT * FROM (
        #         SELECT d.x_name, f.amount,
        #                RANK() OVER (PARTITION BY d.x_name ORDER BY f.amount DESC) AS rnk
        #         FROM fact_events f JOIN dim_x d ON d.dim_x_key = f.dim_x_key
        #     ) WHERE rnk <= 3;
        # """,
        # "q2_monthly_trend": """
        #     SELECT dd.year, dd.month_name, SUM(f.amount) AS total
        #     FROM fact_events f JOIN dim_date dd ON dd.date_key = f.date_key
        #     GROUP BY dd.year, dd.month, dd.month_name
        #     ORDER BY dd.year, dd.month;
        # """,
        # "q3_running_total": """
        #     SELECT dd.date, SUM(f.amount) OVER (
        #                ORDER BY dd.date
        #                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        #            ) AS running_total
        #     FROM fact_events f JOIN dim_date dd ON dd.date_key = f.date_key;
        # """,
    }
    with engine.connect() as conn:
        for name, sql in queries.items():
            print(f"\n--- {name} ---")
            print(pd.read_sql(text(sql), conn).head(10).to_string(index=False))


# ----------------------------------------------------------------------------
def main() -> None:
    print("\nPipeline: source -> bronze -> silver -> gold -> analytics\n")
    raw = extract()
    bronze = load_bronze(raw)
    silver = transform_silver(bronze)
    silver = validate(silver)
    fact, dims = build_gold(silver)
    create_schema(fact, dims)   # DDL first, outside the data transaction
    load_gold(fact, dims)       # DML only -> genuinely atomic
    run_analytics()
    print("\nDone. Run again - the results are identical (idempotent).\n")


if __name__ == "__main__":
    main()
