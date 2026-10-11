"""Tests for src/model.py (Member C), run on a STAND-IN table (tests/standin.py). They check the CODE: the rules of the
split, the health index of contract section 8c, leakage safeguards, the shape of score(). They say nothing about real
predictive performance. Run:  python -m pytest tests/test_model.py     (delete the stand-in once real data is in)
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import model as M
from src import plan as P
from tests.standin import make_standin

_state = {}


def standin():
    if "data" not in _state:
        _state["data"] = make_standin()
    return _state["data"]


def trained():
    """Train once on the stand-in (small settings) into a temp folder, never into the real models/ and reports/."""
    if "run" not in _state:
        snap, dim, costs, budget = standin()
        out = Path(tempfile.mkdtemp(prefix="model_test_"))
        result = M.train(snap, dim, costs, budget, out_dir=out, n_boot=10, n_repeats=2, verbose=False)
        _state["run"] = (out, result)
    return _state["run"]


def _row(**kw):
    base = {c: np.nan for c in M.FEATURE_COLUMNS}
    base.update({"has_oil_test": 1, "n_tests": 2, "tdcg_delta": 0.0, "load_p95_12m": 50.0, "age_years": 10.0})
    base.update(kw)
    return pd.DataFrame([base])


# ---- contract -----------------------------------------------------------------------------------------
def test_feature_list_matches_the_contract():
    assert M.FEATURE_COLUMNS == [
        "age_years", "load_mean_12m", "load_p95_12m", "temp_mean_12m", "hot_months_12m", "has_oil_test", "n_tests",
        "months_since_last_test", "h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last", "co_last", "tdcg_last",
        "tdcg_delta", "c2h2_delta", "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6", "months_since_repair"]


# ---- health index (contract section 8c) --------------------------------------------------------------------
def test_health_index_scores_each_field_as_in_the_contract():
    """Hand-computed: C2H2 = 20 ppm is condition 3 (30 points); tdcg_delta 20 -> 10; load p95 95 -> 10; age 35 -> 12."""
    row = _row(h2_last=50, ch4_last=50, c2h2_last=20, c2h4_last=10, c2h6_last=10, co_last=100,
               tdcg_delta=20.0, load_p95_12m=95.0, age_years=35.0)
    t = M.health_table(row).iloc[0]
    assert (t["gas_points"], t["trend_points"], t["load_points"], t["age_points"]) == (30, 10, 10, 12)
    assert t["health_score"] == 100 - 62 and t["worst_condition"] == 3


def test_health_index_boundaries_follow_the_table():
    """Edges: the upper limit of a condition belongs to that condition; the next value moves up a band."""
    def points(**kw):
        return M.health_table(_row(h2_last=10, ch4_last=10, c2h2_last=0.5, c2h4_last=5, c2h6_last=5, co_last=50, **kw)).iloc[0]
    assert points(tdcg_delta=10.0)["trend_points"] == 0 and points(tdcg_delta=10.01)["trend_points"] == 10
    assert points(tdcg_delta=30.0)["trend_points"] == 10 and points(tdcg_delta=30.01)["trend_points"] == 20
    assert points(load_p95_12m=90.0)["load_points"] == 0 and points(load_p95_12m=100.0)["load_points"] == 10 and points(load_p95_12m=100.5)["load_points"] == 20
    assert [points(age_years=a)["age_points"] for a in (20, 21, 30, 31, 40, 41)] == [0, 6, 6, 12, 12, 20]
    gas = lambda v: M.health_table(_row(h2_last=v, ch4_last=10, c2h2_last=0.5, c2h4_last=5, c2h6_last=5, co_last=50)).iloc[0]["gas_points"]
    assert [gas(v) for v in (100, 101, 700, 701, 1800, 1801)] == [0, 15, 15, 30, 30, 40]


def test_health_score_is_null_without_a_usable_oil_test_and_clipped_to_0_100():
    no_test = _row(has_oil_test=0, n_tests=0)
    assert np.isnan(M.health_table(no_test).iloc[0]["health_score"])
    terrible = _row(h2_last=5000, ch4_last=5000, c2h2_last=500, c2h4_last=500, c2h6_last=500, co_last=5000,
                    tdcg_delta=99.0, load_p95_12m=130.0, age_years=60.0)
    assert M.health_table(terrible).iloc[0]["health_score"] == 0                      # 100 - (40+20+20+20) clipped at 0


def test_missing_trend_load_or_age_deducts_nothing_instead_of_guessing():
    t = M.health_table(_row(h2_last=10, ch4_last=10, c2h2_last=0.5, c2h4_last=5, c2h6_last=5, co_last=50,
                            tdcg_delta=np.nan, load_p95_12m=np.nan, age_years=np.nan)).iloc[0]
    assert t["trend_points"] == t["load_points"] == t["age_points"] == 0 and t["health_score"] == 100


# ---- split -----------------------------------------------------------------------------------------------
def test_split_follows_annex_a():
    """No transformer in both sets; held-out = latest two labelable snapshots; training label windows end before the test period."""
    split = M.make_split(standin()[0])
    dev, test = split["dev"], split["test"]
    assert set(dev["transformer_id"]).isdisjoint(set(test["transformer_id"]))
    assert sorted(test["snapshot_date"].unique()) == [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01")]
    assert dev["snapshot_date"].max() + pd.DateOffset(months=12) <= test["snapshot_date"].min()
    assert dev["failed_next_12m"].notna().all() and test["failed_next_12m"].notna().all()
    assert abs(len(set(test["transformer_id"])) / 400 - 0.20) < 0.03                   # about 20% of transformers


def test_snapshots_after_2025_10_01_are_never_used_for_training_or_testing():
    split = M.make_split(standin()[0])
    for part in ("dev", "test"):
        assert split[part]["snapshot_date"].max() <= pd.Timestamp("2025-10-01")


def test_split_is_reproducible():
    a, b = M.make_split(standin()[0]), M.make_split(standin()[0])
    assert a["holdout_ids"] == b["holdout_ids"]


# ---- validation and leakage ---------------------------------------------------------------------------------
def test_validation_never_trains_on_a_later_snapshot_or_the_same_transformer():
    """Spy on every fit: training rows are strictly earlier than the validation snapshot and share no transformer with it."""
    dev = M.make_split(standin()[0])["dev"]
    seen = []

    class Spy(M.AgeOnly):
        def fit(self, X, y):
            self.train = X
            return super().fit(X, y)

        def predict_proba(self, X):
            seen.append((self.train, X))
            return super().predict_proba(X)

    M.grouped_time_aware_cv(dev, Spy, repeats=2)
    assert seen
    for tr, va in seen:
        assert tr["snapshot_date"].max() < va["snapshot_date"].min()
        assert set(tr["transformer_id"]).isdisjoint(set(va["transformer_id"]))


def test_validation_is_repeated_with_different_groupings():
    dev = M.make_split(standin()[0])["dev"]
    runs = M.grouped_time_aware_cv(dev, lambda: M.AgeOnly(), repeats=3)
    assert len(runs) == 3 and not np.allclose(runs[0]["p"].to_numpy(), runs[1]["p"].to_numpy())


def test_imputation_is_fitted_on_training_rows_only():
    """The median imputer inside the pipeline learns from the rows it is fitted on, never from later data."""
    pipe = M.make_pipeline("logreg", "median")
    X = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in M.FEATURE_COLUMNS})
    y = np.array([0, 1, 0, 1])
    pipe.fit(X, y)
    before = pipe.named_steps["impute"].statistics_.copy()
    pipe.predict_proba(X.assign(age_years=1e9))                                        # extreme "test" values change nothing
    assert np.array_equal(before, pipe.named_steps["impute"].statistics_)
    assert set(pipe.named_steps) == {"clip", "impute", "scale", "clf"}                   # all preprocessing is a pipeline step


def test_metrics_match_hand_values():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    m = M.metric_bundle(y, p)                                                           # top 10% of 10 rows = 1 row -> 1 of 2 failures
    assert m["recall_top10"] == 0.5 and m["lift_top10"] == 5.0
    assert abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-12


# ---- end to end on the stand-in ----------------------------------------------------------------------------
def test_train_writes_every_output_and_logs_every_configuration():
    out, result = trained()
    for f in ("models/model.joblib", "models/model_card.json", "reports/experiments.csv", "reports/evaluation.md", "reports/reliability.png"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f
    log = pd.read_csv(out / "reports/experiments.csv")
    assert set(log.loc[log["stage"] == "cv", "model"]) == {"age_only", "health_index", "logreg", "hgb"}
    assert set(log.loc[log["stage"] == "cv", "missing_method"]) >= {"median", "median_indicator", "native"}      # three missing-data methods
    assert (log["stage"] == "holdout").sum() == 1                                         # held-out set evaluated exactly once
    assert {"model", "missing_method", "data_window", "params", "seed", "pr_auc_mean", "brier_mean"} <= set(log.columns)


def test_model_card_has_the_annex_c_fields():
    out, _ = trained()
    card = json.loads((out / "models/model_card.json").read_text())["model_card"]
    for key in ("name", "version", "algorithm", "snapshot_design", "validation", "missing_data_methods_compared", "baselines",
                "final", "no_oil_test_units", "budget_test", "known_limits", "random_seed"):
        assert key in card, key
    assert card["random_seed"] == 42 and {"age_only", "rule_health_index"} <= set(card["baselines"])


def test_score_returns_the_contract_columns_and_valid_numbers():
    out, _ = trained()
    bundle = M.load_model(out / "models/model.joblib")
    snap = standin()[0]
    now = snap[snap["snapshot_date"] == pd.Timestamp("2026-10-01")].reset_index(drop=True)
    s = M.score(now, bundle)
    assert len(s) == 400
    assert {"transformer_id", "failure_prob_12m", "prob_low", "prob_high", "status", "health_score", "top_drivers", "data_quality",
            "model_version", "has_oil_test", "months_since_last_test", "age_years"} <= set(s.columns)
    assert s["failure_prob_12m"].between(0, 1).all() and (s["prob_low"] <= s["failure_prob_12m"] + 1e-12).all()
    assert (s["failure_prob_12m"] <= s["prob_high"] + 1e-12).all()
    assert set(s["status"]) <= {"scored", "scored_low_confidence", "insufficient_data"}
    assert all(len(d) <= 3 and all({"feature", "contribution"} <= set(x) for x in d) for d in s["top_drivers"])
    assert s["data_quality"].str.len().min() > 5


def test_units_with_no_oil_test_are_low_confidence_and_have_no_health_score():
    out, _ = trained()
    bundle = M.load_model(out / "models/model.joblib")
    snap = standin()[0]
    s = M.score(snap[snap["snapshot_date"] == pd.Timestamp("2026-10-01")].reset_index(drop=True), bundle)
    none = s[s["has_oil_test"] == 0]
    assert len(none) > 50
    assert (none["status"] == "scored_low_confidence").all() and none["health_score"].isna().all()
    assert none["data_quality"].str.startswith("no oil test").all()


def test_status_rules_stale_test_and_single_test_and_no_data():
    out, _ = trained()
    bundle = M.load_model(out / "models/model.joblib")
    snap = standin()[0]
    base = snap[(snap["snapshot_date"] == pd.Timestamp("2026-10-01")) & (snap["has_oil_test"] == 1) & (snap["n_tests"] >= 2)].head(3).reset_index(drop=True)
    base.loc[0, "months_since_last_test"] = 19.0                                       # stale: more than 18 months
    base.loc[1, ["months_since_last_test", "n_tests"]] = [3.0, 1]                       # only one test
    base.loc[2, ["months_since_last_test", "n_tests"]] = [3.0, 3]                       # fresh and several tests
    s = M.score(base, bundle)
    assert list(s["status"]) == ["scored_low_confidence", "scored_low_confidence", "scored"]
    empty = base.head(1).copy()
    empty[[c for c in M.FEATURE_COLUMNS if c not in ("n_tests",)]] = np.nan
    empty[["has_oil_test", "n_tests"]] = 0
    e = M.score(empty, bundle).iloc[0]
    assert e["status"] == "insufficient_data" and np.isnan(e["failure_prob_12m"])


def test_score_output_feeds_the_plan_without_blind_replacements():
    """Integration with Step 1: score -> compute_risk -> make_plan; a unit with no oil test is only ever inspected."""
    out, _ = trained()
    bundle = M.load_model(out / "models/model.joblib")
    snap, dim, costs, _ = standin()
    s = M.score(snap[snap["snapshot_date"] == pd.Timestamp("2026-10-01")].reset_index(drop=True), bundle)
    risk = P.compute_risk(s, dim)
    plan = P.make_plan(risk, costs, 2_000_000)
    assert plan["total_cost"] <= 2_000_000 and plan["actions"]
    untested = set(risk.loc[risk["has_oil_test"] == 0, "transformer_id"])
    assert all(a["action"] == "inspect" for a in plan["actions"] if a["transformer_id"] in untested)


def test_training_twice_gives_identical_numbers():
    snap, dim, costs, budget = standin()
    runs = []
    for _ in range(2):
        out = Path(tempfile.mkdtemp(prefix="model_repro_"))
        r = M.train(snap, dim, costs, budget, out_dir=out, n_boot=6, n_repeats=2, verbose=False)
        runs.append(r["holdout"]["p_calibrated"]["pr_auc"]["value"])
    assert runs[0] == runs[1]


def test_model_info_reads_the_saved_card():
    out, _ = trained()
    card = json.loads((out / "models/model_card.json").read_text())
    assert card["model_card"]["version"].startswith(("logreg", "hgb"))


def test_units_without_an_oil_test_get_wider_ranges_than_tested_units():
    """Honest uncertainty: a unit we have never looked at can hide a fault, so its 90% range must be wider (brief step 8)."""
    _, result = trained()
    w = result["width_check"]
    assert w["median_width_no_oil_test"] > w["median_width_with_oil_test"], w


def test_the_range_always_brackets_the_probability_and_stays_in_0_1():
    out, _ = trained()
    bundle = M.load_model(out / "models/model.joblib")
    snap = standin()[0]
    s = M.score(snap[snap["snapshot_date"] == pd.Timestamp("2026-10-01")].reset_index(drop=True), bundle)
    ok = s["status"] != "insufficient_data"
    assert ((s.loc[ok, "prob_low"] <= s.loc[ok, "failure_prob_12m"] + 1e-12) & (s.loc[ok, "failure_prob_12m"] <= s.loc[ok, "prob_high"] + 1e-12)).all()
    assert s.loc[ok, "prob_high"].le(1).all() and s.loc[ok, "prob_low"].ge(0).all()


def test_saved_model_points_at_src_model_so_other_modules_can_load_it():
    """Regression: `python -m src.model` once saved the class as __main__.FinalModel and the API could not load it."""
    import subprocess, sys
    out, _ = trained()
    code = ("import joblib, sys; b = joblib.load(sys.argv[1]); "
            "assert type(b['model']).__module__ == 'src.model', type(b['model']).__module__; print('ok')")
    r = subprocess.run([sys.executable, "-c", code, str(out / "models/model.joblib")], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parent.parent)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-500:]


def test_extreme_values_are_clipped_to_the_training_range_so_probabilities_stay_sane():
    """Regression: one unit's acetylene far above training gave a 99.85% probability. Clipping is learned on training rows."""
    clip = M.ClipToTrainingRange().fit(pd.DataFrame({"a": np.arange(1.0, 101.0), "b": np.arange(1.0, 101.0)}))
    out = clip.transform(pd.DataFrame({"a": [1e6, np.nan], "b": [-1e6, 50.0]}))
    assert out.loc[0, "a"] <= 100 and out.loc[0, "b"] >= 1 and np.isnan(out.loc[1, "a"]) and out.loc[1, "b"] == 50.0
