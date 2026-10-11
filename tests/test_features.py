"""Tests for the gold layer and the shared feature function (Member B).

Run: python -m pytest tests/test_features.py -q   (builds the warehouse from data/ first)
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features
from src.pipeline import SNAPSHOT_DATES, get_connection, run_pipeline


@pytest.fixture(scope="module")
def tables():
    """Build the warehouse once, then read what the tests need."""
    run_pipeline()
    with get_connection() as conn:
        return {name: pd.read_sql(f"SELECT * FROM {name}", conn)
                for name in ["fact_snapshot", "silver_operation", "silver_oil_test", "silver_transformer"]}


def latest_usable_test(oil, op, transformer_id, snapshot_date):
    """Plain-pandas version of the SQL in build_features: the newest test before the snapshot
    and after the month of the latest repair/failure. Written separately so the two can disagree."""
    events = op[(op.transformer_id == transformer_id) & (op.month < snapshot_date)
                & op.event.isin(["repair", "failure"])]
    tests = oil[(oil.transformer_id == transformer_id) & (oil.test_date < snapshot_date)]
    if len(events):
        trusted_from = (pd.Timestamp(events.month.max()) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        tests = tests[tests.test_date >= trusted_from]
    return tests.sort_values("test_date")


def test_grain_one_row_per_transformer_per_snapshot(tables):
    snap = tables["fact_snapshot"]
    assert len(snap) == len(tables["silver_transformer"]) * len(SNAPSHOT_DATES)
    assert not snap.duplicated(["transformer_id", "snapshot_date"]).any()


def test_no_leakage_features_match_an_independent_recomputation(tables):
    """HCL leakage check: for a random sample of rows, recompute the oil and load features using only
    data dated before the snapshot. If the SQL had peeked past the snapshot, these would differ."""
    snap, oil, op = tables["fact_snapshot"], tables["silver_oil_test"], tables["silver_operation"]
    sample = snap.sample(150, random_state=42)
    for _, row in sample.iterrows():
        tests = latest_usable_test(oil, op, row.transformer_id, row.snapshot_date)
        assert row.n_tests == len(tests)
        if len(tests):
            assert row.months_since_last_test > 0                      # the last test is BEFORE the snapshot
            assert row.h2_last == pytest.approx(tests.h2_ppm.iloc[-1])
        else:
            assert row.has_oil_test == 0 and np.isnan(row.h2_last)

        start = (pd.Timestamp(row.snapshot_date) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        window = op[(op.transformer_id == row.transformer_id) & (op.month >= start) & (op.month < row.snapshot_date)]
        if window.load_pct.notna().any():
            assert row.load_mean_12m == pytest.approx(window.load_pct.mean())


def test_repair_resets_the_gas_history(tables):
    """Edge case E5: after a repair, only tests from the following month on describe the unit."""
    snap, oil, op = tables["fact_snapshot"], tables["silver_oil_test"], tables["silver_operation"]
    repairs = op[op.event == "repair"]
    assert len(repairs) > 0
    for _, rep in repairs.iterrows():
        row = snap[(snap.transformer_id == rep.transformer_id) & (snap.snapshot_date == SNAPSHOT_DATES[-1])].iloc[0]
        tests_before_snapshot = oil[(oil.transformer_id == rep.transformer_id) & (oil.test_date < SNAPSHOT_DATES[-1])]
        assert row.n_tests < len(tests_before_snapshot)               # pre-repair tests were dropped
        assert row.months_since_repair > 0


def test_labels_only_where_the_next_12_months_are_known(tables):
    snap = tables["fact_snapshot"]
    labelled = snap[snap.snapshot_date <= "2025-10-01"]
    unlabelled = snap[snap.snapshot_date > "2025-10-01"]
    assert labelled.failed_next_12m.notna().all()
    assert unlabelled.failed_next_12m.isna().all()
    assert labelled.failed_next_12m.sum() > 0


def test_the_api_gets_exactly_the_training_features(tables):
    """One shared function: features built on demand (what the API does) equal the stored training rows."""
    with get_connection() as conn:
        live = build_features(conn, ["2025-04-01"])
    stored = tables["fact_snapshot"]
    stored = stored[stored.snapshot_date == "2025-04-01"].sort_values("transformer_id").reset_index(drop=True)
    live = live.sort_values("transformer_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(live[FEATURE_COLUMNS], stored[FEATURE_COLUMNS], check_dtype=False)
