"""Tests for src/loader.py (Member B) - run on a temporary copy of data/, so data/ingested stays clean.

Run: python -m pytest tests/test_loader.py -q
"""

import shutil

import pandas as pd
import pytest

from src import pipeline as P
from src.features import build_features
from src.loader import load_csv


@pytest.fixture()
def data_copy(tmp_path, monkeypatch):
    """Point the pipeline at a copy of data/, build it, and rebuild the real warehouse afterwards."""
    real_dir = P.DATA_DIR
    copy = tmp_path / "data"
    shutil.copytree(real_dir, copy, ignore=shutil.ignore_patterns("ingested"))
    monkeypatch.setattr(P, "DATA_DIR", copy)
    P.run_pipeline()
    yield copy
    monkeypatch.setattr(P, "DATA_DIR", real_dir)
    P.run_pipeline()


def test_late_drop_new_oil_tests_change_the_features(data_copy):
    """The 13:45 rehearsal: 10 never-tested units get their first test -> they must now count as tested."""
    late = data_copy / "late_drop" / "oil_tests_late.csv"
    first_timers = set(pd.read_csv(late).transformer_id) - set(pd.read_csv(data_copy / "oil_tests.csv").transformer_id)

    result = load_csv("oil_tests", str(late))
    assert result["status"] == "ok" and result["rows_loaded"] == 40 and result["rows_rejected"] == 0

    with P.get_connection() as conn:
        now = build_features(conn, [pd.Timestamp.today().strftime("%Y-%m-%d")])
    assert now[now.transformer_id.isin(first_timers)].has_oil_test.eq(1).all()


def test_late_budget_replaces_the_old_one(data_copy):
    result = load_csv("costs_budget", str(data_copy / "late_drop" / "costs_budget_late.csv"))
    assert result["status"] == "ok"
    with P.get_connection() as conn:
        assert pd.read_sql("SELECT annual_budget FROM silver_budget", conn).annual_budget.iloc[0] == 4_000_000


def test_bad_rows_are_rejected_and_counted(data_copy, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({
        "transformer_id": ["TR-0001", "TR-9999", "TR-0002", "TR-0003"],
        "test_date": ["2026-10-02", "2026-10-02", "2031-01-01", "2026-10-02"],
        "h2_ppm": [40, 40, 40, -5], "ch4_ppm": [30] * 4, "c2h2_ppm": [0.5] * 4,
        "c2h4_ppm": [15] * 4, "c2h6_ppm": [20] * 4, "co_ppm": [300] * 4,
    }).to_csv(bad, index=False)
    result = load_csv("oil_tests", str(bad))
    assert result["rows_loaded"] == 1 and result["rows_rejected"] == 3   # unknown id, future date, negative gas
    assert "unknown transformer_id" in result["detail"] and "in the future" in result["detail"]


def test_missing_column_is_refused_with_a_hint(data_copy, tmp_path):
    bad = tmp_path / "no_gas.csv"
    pd.DataFrame({"transformer_id": ["TR-0001"], "test_date": ["2026-10-02"]}).to_csv(bad, index=False)
    result = load_csv("oil_tests", str(bad))
    assert result["status"] == "error" and "COLUMN_ALIASES" in result["detail"]
