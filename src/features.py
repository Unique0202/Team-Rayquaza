"""
The shared feature function (Member B).

build_features() is used for TRAINING (pipeline -> fact_snapshot) and for SERVING (the API scores
"now" with it). One function, so the two can never compute an input differently - HCL rule.

THE LEAKAGE RULE: every feature for a snapshot date uses only data dated strictly BEFORE it.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "age_years", "load_mean_12m", "load_p95_12m", "temp_mean_12m", "hot_months_12m",
    "has_oil_test", "n_tests", "months_since_last_test",
    "h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last", "co_last",
    "tdcg_last", "tdcg_delta", "c2h2_delta",
    "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6",
    "months_since_repair",
]
HOT_OIL_C = 80            # a month counts as "running hot" above this top-oil temperature
DAYS_PER_MONTH = 30.44
RATIO_GUARD = 0.1         # added to every ratio's denominator so a 0 ppm reading never divides by zero

# The latest repair or failure before each snapshot - both reset the gas pattern (edge case E5).
RESETS_SQL = """
            SELECT s.transformer_id, s.snapshot_date, MAX(o.month) AS last_reset_month
            FROM snaps s
            JOIN silver_operation o ON o.transformer_id = s.transformer_id
                                   AND o.month < s.snapshot_date
                                   AND o.event IN ('repair', 'failure')
            GROUP BY s.transformer_id, s.snapshot_date"""


def snapshot_cte(snapshot_dates: list[str]) -> str:
    """SQL that lists every (transformer, snapshot date) pair - the grain of fact_snapshot.

    Dates go through pd.Timestamp first, so only well-formed YYYY-MM-DD text can reach the SQL
    (the API passes a user's as_of here)."""
    dates = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in snapshot_dates})
    values = ", ".join(f"('{d}')" for d in dates)
    return f"""
        WITH dates(snapshot_date) AS (VALUES {values}),
        snaps AS (SELECT t.transformer_id, d.snapshot_date
                  FROM silver_transformer t CROSS JOIN dates d)"""


def build_features(conn, snapshot_dates: list[str]) -> pd.DataFrame:
    """One row per transformer per snapshot date: transformer_id, snapshot_date, *FEATURE_COLUMNS.
    Reads silver tables only. Missing stays missing (NaN) - filling gaps is the model's decision."""
    cte = snapshot_cte(snapshot_dates)

    # 1. Loading and temperature over the 12 months before the snapshot, plus age at the latest month.
    operation = pd.read_sql(f"""{cte},
        latest_month AS (
            -- ROW_NUMBER ranks each transformer's months before the snapshot, newest = 1 (its age now)
            SELECT s.transformer_id, s.snapshot_date, o.age_years,
                   ROW_NUMBER() OVER (PARTITION BY s.transformer_id, s.snapshot_date
                                      ORDER BY o.month DESC) AS rn
            FROM snaps s
            JOIN silver_operation o ON o.transformer_id = s.transformer_id
                                   AND o.month < s.snapshot_date)
        SELECT s.transformer_id, s.snapshot_date, lm.age_years,
               AVG(w.load_pct)       AS load_mean_12m,
               AVG(w.top_oil_temp_c) AS temp_mean_12m,
               SUM(CASE WHEN w.top_oil_temp_c > {HOT_OIL_C} THEN 1 ELSE 0 END) AS hot_months_12m
        FROM snaps s
        LEFT JOIN latest_month lm ON lm.transformer_id = s.transformer_id
                                 AND lm.snapshot_date = s.snapshot_date AND lm.rn = 1
        LEFT JOIN silver_operation w ON w.transformer_id = s.transformer_id
                                    AND w.month >= date(s.snapshot_date, '-12 months')
                                    AND w.month <  s.snapshot_date
        GROUP BY s.transformer_id, s.snapshot_date, lm.age_years""", conn)

    # 95th percentile of load in the same 12-month window (SQLite has no percentile function, so pandas)
    window_loads = pd.read_sql(f"""{cte}
        SELECT s.transformer_id, s.snapshot_date, w.load_pct
        FROM snaps s
        JOIN silver_operation w ON w.transformer_id = s.transformer_id
                               AND w.month >= date(s.snapshot_date, '-12 months')
                               AND w.month <  s.snapshot_date""", conn)
    p95 = (window_loads.groupby(["transformer_id", "snapshot_date"])["load_pct"]
           .quantile(0.95).rename("load_p95_12m").reset_index())

    # 2. The latest repair or failure before the snapshot (RESETS_SQL above).
    resets = pd.read_sql(f"{cte} {RESETS_SQL}", conn)

    # 3. The latest USABLE oil test: before the snapshot AND after the month of the latest reset.
    #    We only know an event's month, so a test is trusted only from the next month on.
    tests = pd.read_sql(f"""{cte},
        resets AS ({RESETS_SQL}),
        usable AS (
            SELECT s.transformer_id, s.snapshot_date, x.test_date,
                   x.h2_ppm, x.ch4_ppm, x.c2h2_ppm, x.c2h4_ppm, x.c2h6_ppm, x.co_ppm,
                   x.h2_ppm + x.ch4_ppm + x.c2h2_ppm + x.c2h4_ppm + x.c2h6_ppm + x.co_ppm AS tdcg
            FROM snaps s
            JOIN silver_oil_test x ON x.transformer_id = s.transformer_id
                                  AND x.test_date < s.snapshot_date
            LEFT JOIN resets r ON r.transformer_id = s.transformer_id
                              AND r.snapshot_date = s.snapshot_date
            WHERE r.last_reset_month IS NULL
               OR x.test_date >= date(r.last_reset_month, '+1 month')),
        ranked AS (
            SELECT u.*,
                   -- 1 = the newest usable test before this snapshot
                   ROW_NUMBER() OVER (PARTITION BY transformer_id, snapshot_date ORDER BY test_date DESC) AS rn,
                   -- how many usable tests this transformer had by this snapshot
                   COUNT(*)     OVER (PARTITION BY transformer_id, snapshot_date) AS n_tests,
                   -- LAG looks one test back in time: the previous test's date, total gas and acetylene
                   LAG(test_date) OVER (PARTITION BY transformer_id, snapshot_date ORDER BY test_date) AS prev_date,
                   LAG(tdcg)      OVER (PARTITION BY transformer_id, snapshot_date ORDER BY test_date) AS prev_tdcg,
                   LAG(c2h2_ppm)  OVER (PARTITION BY transformer_id, snapshot_date ORDER BY test_date) AS prev_c2h2
            FROM usable u)
        SELECT * FROM ranked WHERE rn = 1""", conn)

    # 4. Put the pieces together: one row per (transformer, snapshot), even with no tests or no history.
    base = pd.read_sql(f"{cte} SELECT transformer_id, snapshot_date FROM snaps", conn)
    df = (base.merge(operation, on=["transformer_id", "snapshot_date"], how="left")
              .merge(p95, on=["transformer_id", "snapshot_date"], how="left")
              .merge(resets, on=["transformer_id", "snapshot_date"], how="left")
              .merge(tests, on=["transformer_id", "snapshot_date"], how="left"))

    snap = pd.to_datetime(df["snapshot_date"])
    last_test = pd.to_datetime(df["test_date"])
    prev_test = pd.to_datetime(df["prev_date"])

    df["n_tests"] = df["n_tests"].fillna(0).astype(int)
    df["has_oil_test"] = (df["n_tests"] > 0).astype(int)
    df["months_since_last_test"] = (snap - last_test).dt.days / DAYS_PER_MONTH
    for gas in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co"]:
        df[f"{gas}_last"] = df[f"{gas}_ppm"]
    df["tdcg_last"] = df["tdcg"]

    # Rate of change per month between the last two usable tests (NaN with fewer than 2 tests)
    months_between = (last_test - prev_test).dt.days / DAYS_PER_MONTH
    df["tdcg_delta"] = (df["tdcg"] - df["prev_tdcg"]) / months_between
    df["c2h2_delta"] = (df["c2h2_ppm"] - df["prev_c2h2"]) / months_between

    # Gas ratios used in dissolved-gas analysis to tell the fault type (arcing, thermal, discharge)
    df["ratio_c2h2_c2h4"] = df["c2h2_ppm"] / (df["c2h4_ppm"] + RATIO_GUARD)
    df["ratio_ch4_h2"] = df["ch4_ppm"] / (df["h2_ppm"] + RATIO_GUARD)
    df["ratio_c2h4_c2h6"] = df["c2h4_ppm"] / (df["c2h6_ppm"] + RATIO_GUARD)

    df["months_since_repair"] = (snap - pd.to_datetime(df["last_reset_month"])).dt.days / DAYS_PER_MONTH

    out = df[["transformer_id", "snapshot_date", *FEATURE_COLUMNS]]
    return out.sort_values(["snapshot_date", "transformer_id"]).reset_index(drop=True)


def build_labels(conn, snapshot_dates: list[str]) -> pd.DataFrame:
    """failed_next_12m per (transformer, snapshot): 1 if a failure happens in the 12 months starting at
    the snapshot, 0 if not, and missing (NaN) if those 12 months are not fully in the data yet.

    Labels come from our extra `event` column; judges' files do not have it, so this is training-only."""
    cte = snapshot_cte(snapshot_dates)
    labels = pd.read_sql(f"""{cte},
        data_end AS (SELECT date(MAX(month), '+1 month') AS first_unseen_month FROM silver_operation)
        SELECT s.transformer_id, s.snapshot_date,
               CASE
                 -- the 12 months after this snapshot are not all in the data yet: we cannot know the answer
                 WHEN date(s.snapshot_date, '+12 months') > (SELECT first_unseen_month FROM data_end) THEN NULL
                 ELSE MAX(CASE WHEN o.event = 'failure' THEN 1 ELSE 0 END)
               END AS failed_next_12m
        FROM snaps s
        LEFT JOIN silver_operation o ON o.transformer_id = s.transformer_id
                                    AND o.month >= s.snapshot_date
                                    AND o.month <  date(s.snapshot_date, '+12 months')
        GROUP BY s.transformer_id, s.snapshot_date""", conn)
    return labels.sort_values(["snapshot_date", "transformer_id"]).reset_index(drop=True)
