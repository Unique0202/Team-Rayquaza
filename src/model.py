"""Member C: baselines, models, validation, calibration, intervals, explanations, score().
Follows docs/team/SHARED_CONTRACT.md sections 6, 7, 8c and the Member C brief. scikit-learn only.

Plain-English map of what happens in train():
  1. SPLIT      keep a final held-out set (last two labelable snapshots x 20% of transformers) aside, touched ONCE
  2. VALIDATE   on the rest: folds grouped by transformer, training always on EARLIER snapshots, repeated 5 times
  3. COMPARE    two baselines (age only, rule-based health index) against logistic regression (2 missing-data methods)
                and gradient boosting (native missing values); every configuration is logged to reports/experiments.csv
  4. PICK       the simplest model within one spread of the best PR-AUC (rule fixed before looking at results)
  5. CALIBRATE  Platt scaling, so a predicted 0.2 means about 20 percent
  6. INTERVALS  30 bootstrap copies of the final model; 5th to 95th percentile = the 90% range
  7. EVALUATE   the held-out set once; budget plan vs the age-based plan
  8. SAVE       models/model.joblib, models/model_card.json, reports/*.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import plan as P

try:                                                     # B's file is the single source of truth for the feature list
    from src.features import FEATURE_COLUMNS
except ImportError:                                      # until B pushes it: the list from Contract section 6
    FEATURE_COLUMNS = [
        "age_years", "load_mean_12m", "load_p95_12m", "temp_mean_12m", "hot_months_12m",
        "has_oil_test", "n_tests", "months_since_last_test",
        "h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last", "co_last",
        "tdcg_last", "tdcg_delta", "c2h2_delta",
        "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6",
        "months_since_repair",
    ]

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "models/model.joblib"
CARD_PATH = "models/model_card.json"
EXPERIMENTS_PATH = "reports/experiments.csv"
TODAY = "2026-10-11"
HORIZON_MONTHS = 12
N_BOOTSTRAP = 30                  # bootstrap copies for the interval (contract: 20 to 50)
N_REPEATS = 5                     # validation repeats with different seeds
HOLDOUT_FRACTION = 0.20           # share of transformers kept for the final test
N_HOLDOUT_SNAPSHOTS = 2           # the latest labelable snapshots (2025-07-01 and 2025-10-01)
STALE_MONTHS = 18                 # contract: a last test older than this makes a unit low confidence
N_DONORS = 30                     # "what if the missing test had shown what tested units show": scenarios per untested unit
# every feature that exists only because an oil test exists
OIL_COLUMNS = ["has_oil_test", "n_tests", "months_since_last_test", "h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last",
               "co_last", "tdcg_last", "tdcg_delta", "c2h2_delta", "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6"]


# =============================================================================================
# 1. THE RULE-BASED HEALTH INDEX  (contract section 8c)  -  used as health_score, as a baseline, as an explanation
# =============================================================================================
# Upper limit of IEEE C57.104 dissolved-gas conditions 1, 2, 3 (anything above the third limit is condition 4), in ppm.
# TAKEN FROM THE TEAM CONTRACT. NOT VERIFIED AGAINST THE STANDARD: sources disagree on acetylene (an older table gives
# 35/50/80, the contract uses 1/9/35). Check the published C57.104 before quoting these; change them here, nowhere else.
GAS_LIMITS = {"h2_last": (100, 700, 1800), "ch4_last": (120, 400, 1000), "c2h2_last": (1, 9, 35),
              "c2h4_last": (50, 100, 200), "c2h6_last": (65, 100, 150), "co_last": (350, 570, 1400)}
GAS_POINTS = np.array([0, 15, 30, 40])                   # points deducted for condition 1, 2, 3, 4 (the worst gas counts)


def health_table(features: pd.DataFrame) -> pd.DataFrame:
    """Points deducted from 100 for each field, and the resulting health_score (0 worst, 100 best).

    gas score   : worst gas condition among the six latest gases        C1 0 | C2 15 | C3 30 | C4 40
    trend score : tdcg_delta (ppm per month)                            <=10 0 | <=30 10 | above 20
    load score  : load_p95_12m                                          <=90 0 | <=100 10 | above 20
    age score   : age_years                                             <=20 0 | <=30 6 | <=40 12 | above 20
    health_score is NaN (NULL) when there is no usable oil test: the gas and trend scores cannot be computed and an
    untested unit must never look healthy. A missing load, age or trend deducts nothing (unknown, not bad)."""
    f = features
    conditions = []
    for col, (l1, l2, l3) in GAS_LIMITS.items():
        v = f[col].to_numpy(dtype=float)
        cond = 1 + (v > l1).astype(float) + (v > l2) + (v > l3)
        conditions.append(np.where(np.isnan(v), np.nan, cond))
    conditions = np.vstack(conditions)                                            # 6 x n_units
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                           # "all-NaN column" = no gases at all
        worst = np.nanmax(conditions, axis=0)
    no_gas = np.isnan(worst)
    gas_pts = np.where(no_gas, 0, GAS_POINTS[np.nan_to_num(worst, nan=1).astype(int) - 1])

    def stepped(values, cuts, points):
        v = values.to_numpy(dtype=float)
        out = np.full(len(v), float(points[-1]))
        for cut, pts in reversed(list(zip(cuts, points[:-1]))):                   # smallest cut last wins
            out = np.where(v <= cut, pts, out)
        return np.where(np.isnan(v), 0.0, out)

    trend_pts = stepped(f["tdcg_delta"], (10, 30), (0, 10, 20))
    load_pts = stepped(f["load_p95_12m"], (90, 100), (0, 10, 20))
    age_pts = stepped(f["age_years"], (20, 30, 40), (0, 6, 12, 20))
    usable = (f["has_oil_test"].to_numpy(dtype=float) == 1) & ~no_gas
    score = np.clip(100 - (gas_pts + trend_pts + load_pts + age_pts), 0, 100)
    return pd.DataFrame({"gas_points": gas_pts, "trend_points": trend_pts, "load_points": load_pts, "age_points": age_pts,
                         "worst_condition": np.where(no_gas, np.nan, worst),
                         "health_score": np.where(usable, score, np.nan)}, index=f.index)


def health_explanation(row: pd.Series) -> str:
    """One line saying where a unit's health points were lost, e.g. 'gas 30 (condition 3) + trend 10 + load 0 + age 12'."""
    if pd.isna(row["health_score"]):
        return "no usable oil test: health score not available"
    return (f"gas {row['gas_points']:.0f} (condition {row['worst_condition']:.0f}) + trend {row['trend_points']:.0f} + "
            f"load {row['load_points']:.0f} + age {row['age_points']:.0f} deducted from 100")


# =============================================================================================
# 2. MODELS AND MISSING-DATA METHODS  (all preprocessing lives inside a Pipeline, so it is fitted on training rows only)
# =============================================================================================
CONFIGS = [("logreg", "median"), ("logreg", "median_indicator"), ("hgb", "native")]      # simplest first
PARAMS = {"logreg": {"C": 0.1, "class_weight": "balanced"},
          "hgb": {"max_depth": 3, "learning_rate": 0.05, "max_iter": 120, "min_samples_leaf": 30,
                  "l2_regularization": 1.0, "class_weight": "balanced"}}


class KeepObserved(BaseEstimator, TransformerMixin):
    """Drops columns that are completely empty in the training rows (for example months_since_repair when nothing was repaired
    yet). Boosting cannot bin a column with no values at all; there is nothing to learn from it anyway. Fitted on training only."""
    def fit(self, X, y=None):
        self.keep_ = list(X.columns[X.notna().any()])
        return self

    def transform(self, X):
        return X[self.keep_]


def make_pipeline(model: str, method: str) -> Pipeline:
    """(i) logreg + median imputation, (ii) logreg + median + missing-indicator columns, (iii) boosting with native NaN.
    class_weight='balanced' because failures are rare; it distorts raw probabilities, which calibration then repairs."""
    if model == "logreg":
        imputer = SimpleImputer(strategy="median", add_indicator=(method == "median_indicator"), keep_empty_features=True)
        return Pipeline([("impute", imputer), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, **PARAMS["logreg"]))])
    return Pipeline([("keep", KeepObserved()),
                     ("clf", HistGradientBoostingClassifier(random_state=SEED, **PARAMS["hgb"]))])   # NaN handled inside the trees


class AgeOnly:
    """Baseline (a): logistic regression on age alone."""
    def fit(self, X, y):
        self.pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler()), ("clf", LogisticRegression())]).fit(X[["age_years"]], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[["age_years"]])[:, 1]


class HealthIndexBaseline:
    """Baseline (b): the rule-based health index of section 8c turned into a probability by a one-feature logistic fit.
    Points deducted = gas + trend + load + age. A unit with no gas data gets the typical (median) gas and trend points of
    the tested units in the training data, so it is never treated as healthy just because a test is missing."""
    def fit(self, X, y):
        t = health_table(X)
        tested = X["has_oil_test"] == 1
        self.typical_gas = float(t.loc[tested, "gas_points"].median()) if tested.any() else 15.0
        self.typical_trend = float(t.loc[tested, "trend_points"].median()) if tested.any() else 0.0
        self.lr = LogisticRegression().fit(self._deducted(X, t)[:, None], y)
        return self

    def _deducted(self, X, t=None):
        t = health_table(X) if t is None else t
        no_test = (X["has_oil_test"] != 1).to_numpy()
        gas = np.where(no_test, self.typical_gas, t["gas_points"])
        trend = np.where(no_test, self.typical_trend, t["trend_points"])
        return (gas + trend + t["load_points"] + t["age_points"]).to_numpy(dtype=float)

    def predict_proba(self, X):
        return self.lr.predict_proba(self._deducted(X)[:, None])[:, 1]


class SkModel:
    """Gives a Pipeline the same fit / predict_proba(DataFrame) shape as the two baselines."""
    def __init__(self, pipeline, columns=None):
        self.pipeline, self.columns = pipeline, columns or FEATURE_COLUMNS

    def fit(self, X, y):
        self.pipeline = clone(self.pipeline).fit(X[self.columns], y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X[self.columns])[:, 1]


# =============================================================================================
# 3. METRICS FOR RARE EVENTS  (accuracy is useless when 1 unit in 20 fails)
# =============================================================================================
def metric_bundle(y, p) -> dict:
    """PR-AUC (ranking quality for rare events), top-10% recall (share of real failures found in the riskiest 10%),
    lift (that share divided by the 10% random picking would give) and Brier (mean squared error of the probabilities)."""
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": np.nan, "recall_top10": np.nan, "lift_top10": np.nan, "brier": float(brier_score_loss(y, p))}
    k = max(1, int(np.ceil(0.10 * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    recall = float(y[top].sum() / y.sum())
    return {"pr_auc": float(average_precision_score(y, p)), "recall_top10": recall, "lift_top10": recall / (k / len(y)),
            "brier": float(brier_score_loss(y, p))}


def cluster_bootstrap(df: pd.DataFrame, p_col: str, n_boot: int = 400, seed: int = SEED) -> dict:
    """90% interval for each metric by resampling whole TRANSFORMERS (one unit has several correlated snapshot rows)."""
    y, p = df["failed_next_12m"].to_numpy(), df[p_col].to_numpy()
    groups = [np.asarray(ix) for ix in df.groupby("transformer_id").indices.values()]
    out = {k: {"value": v, "lo": np.nan, "hi": np.nan} for k, v in metric_bundle(y, p).items()}
    if y.sum() == 0:
        return {**out, "n_pos": 0, "n_rows": len(y)}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in out}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() in (0, len(idx)):
            continue
        for k, v in metric_bundle(y[idx], p[idx]).items():
            draws[k].append(v)
    for k in out:
        if draws[k]:
            out[k]["lo"], out[k]["hi"] = float(np.nanpercentile(draws[k], 5)), float(np.nanpercentile(draws[k], 95))
    return {**out, "n_pos": int(y.sum()), "n_rows": len(y)}


# =============================================================================================
# 4. SPLIT  (Annex A: training earlier than test, no transformer in both)
# =============================================================================================
def make_split(snap: pd.DataFrame) -> dict:
    """Final held-out set = the latest labelable snapshots x a random 20% of transformers (seed 42, picked separately among
    units that did / did not fail and had / lacked an oil test, so each kind is represented). Development set = the other
    transformers at snapshots whose 12-month label window ENDS before the first held-out snapshot (a plain date cut would
    still leak: a training label that looks 12 months ahead can already contain test-period outcomes)."""
    snap = snap.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    labelled = snap[snap["failed_next_12m"].notna()].copy()
    labelled["failed_next_12m"] = labelled["failed_next_12m"].astype(int)
    dates = sorted(labelled["snapshot_date"].unique())
    test_dates = dates[-N_HOLDOUT_SNAPSHOTS:]
    first_test = pd.Timestamp(test_dates[0])

    per_unit = labelled.groupby("transformer_id").agg(ever_failed=("failed_next_12m", "max"), ever_tested=("has_oil_test", "max"))
    rng = np.random.default_rng(SEED)
    holdout = []
    for (f, t), grp in per_unit.groupby(["ever_failed", "ever_tested"]):
        ids = sorted(grp.index)
        rng.shuffle(ids)
        holdout += ids[: int(round(len(ids) * HOLDOUT_FRACTION))]
    holdout = set(holdout)

    is_hold = labelled["transformer_id"].isin(holdout)
    test = labelled[is_hold & labelled["snapshot_date"].isin(test_dates)].reset_index(drop=True)
    window_end = labelled["snapshot_date"] + pd.DateOffset(months=HORIZON_MONTHS)
    dev = labelled[~is_hold & (window_end <= first_test)].reset_index(drop=True)

    assert set(dev["transformer_id"]).isdisjoint(set(test["transformer_id"])), "a transformer is in both dev and test"
    assert dev["snapshot_date"].max() + pd.DateOffset(months=HORIZON_MONTHS) <= test["snapshot_date"].min(), \
        "training label windows reach into the test period"
    return {"dev": dev, "test": test, "labelled": labelled, "holdout_ids": holdout, "test_dates": test_dates}


# =============================================================================================
# 5. VALIDATION ON THE DEVELOPMENT SET: grouped by transformer AND time-aware, repeated with different seeds
# =============================================================================================
def grouped_time_aware_cv(dev: pd.DataFrame, factory, repeats: int = N_REPEATS):
    """For each repeat (seed 42, 43, ...): shuffle transformers into 5 groups. For each group and each validation snapshot
    (except the earliest) train on the OTHER groups' rows at EARLIER snapshots, predict the group at that snapshot.
    A transformer is therefore never in training and validation together, and training never sees a later date than it
    predicts. Returns one DataFrame of out-of-fold predictions per repeat."""
    snaps = sorted(dev["snapshot_date"].unique())
    ids = np.array(sorted(dev["transformer_id"].unique()))
    runs = []
    for r in range(repeats):
        order = np.random.default_rng(SEED + r).permutation(ids)
        folds = np.array_split(order, 5)
        parts = []
        for fold in folds:
            in_fold = dev["transformer_id"].isin(set(fold))
            for s in snaps[1:]:
                tr = dev[~in_fold & (dev["snapshot_date"] < s)]
                va = dev[in_fold & (dev["snapshot_date"] == s)]
                if tr["failed_next_12m"].nunique() < 2 or va.empty:
                    continue
                model = factory().fit(tr, tr["failed_next_12m"].to_numpy())
                parts.append(va[["transformer_id", "snapshot_date", "failed_next_12m", "has_oil_test"]].assign(p=model.predict_proba(va)))
        runs.append(pd.concat(parts, ignore_index=True))
    return runs


def summarise_runs(runs: list) -> dict:
    """Mean and range (min to max) of every metric over the repeats, overall and for units with no oil test."""
    rows, rows_nt = [], []
    for run in runs:
        rows.append(metric_bundle(run["failed_next_12m"], run["p"]))
        nt = run[run["has_oil_test"] == 0]
        rows_nt.append(metric_bundle(nt["failed_next_12m"], nt["p"]))
    out = {}
    for tag, table in (("", rows), ("no_test_", rows_nt)):
        df = pd.DataFrame(table)
        for col in df:
            out[f"{tag}{col}_mean"] = float(df[col].mean())
            out[f"{tag}{col}_min"] = float(df[col].min())
            out[f"{tag}{col}_max"] = float(df[col].max())
            out[f"{tag}{col}_sd"] = float(df[col].std(ddof=0))
    out["n_rows"], out["n_positives"] = int(len(runs[0])), int(runs[0]["failed_next_12m"].sum())
    return out


EXPERIMENT_COLUMNS = ["timestamp", "stage", "model", "missing_method", "data_window", "n_rows", "n_positives", "params", "seed",
                      "pr_auc_mean", "pr_auc_min", "pr_auc_max", "recall_top10_mean", "recall_top10_min", "recall_top10_max",
                      "lift_top10_mean", "brier_mean", "brier_min", "brier_max", "no_test_pr_auc_mean", "no_test_recall_top10_mean",
                      "no_test_brier_mean", "fit_seconds"]


def log_experiment(row: dict, out_dir: Path) -> None:
    """Append one row (one configuration) to reports/experiments.csv."""
    path = out_dir / EXPERIMENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {c: row.get(c) for c in EXPERIMENT_COLUMNS}
    pd.DataFrame([rec]).to_csv(path, mode="a", header=not path.exists(), index=False)


# =============================================================================================
# 6. THE FINAL MODEL: calibrated, with a bootstrap interval and explanations
# =============================================================================================
class FinalModel:
    """One configuration, fitted, calibrated (Platt / sigmoid), with 30 bootstrap copies for the 90% range."""

    def __init__(self, model: str, method: str):
        self.model, self.method = model, method

    def _calibrated(self, X, y, groups):
        """Platt scaling learned on held-out folds grouped by transformer (stratified so every fold has failures)."""
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
        return CalibratedClassifierCV(make_pipeline(self.model, self.method), method="sigmoid", cv=splits).fit(X, y)

    def fit(self, X: pd.DataFrame, n_boot: int = N_BOOTSTRAP):
        X = X.reset_index(drop=True)
        y, groups = X["failed_next_12m"].to_numpy(), X["transformer_id"].to_numpy()
        feats = X[FEATURE_COLUMNS]
        self.uncalibrated = make_pipeline(self.model, self.method).fit(feats, y)          # raw model (also used for explanations)
        self.calibrated = self._calibrated(feats, y, groups)
        rng = np.random.default_rng(SEED)
        ids = np.unique(groups)
        by_id = {i: np.where(groups == i)[0] for i in ids}
        self.bootstrap = []
        for _ in range(n_boot):                                                           # refit on resampled transformers
            idx = np.concatenate([by_id[i] for i in rng.choice(ids, size=len(ids), replace=True)])
            try:
                self.bootstrap.append(self._calibrated(feats.iloc[idx].reset_index(drop=True), y[idx], groups[idx]))
            except ValueError:                                                            # a resample with a one-class fold: skip it
                continue
        assert len(self.bootstrap) >= max(1, n_boot // 2), f"only {len(self.bootstrap)} of {n_boot} bootstrap copies could be fitted"
        self.medians = feats.median()
        # donors for untested units: real oil-test feature sets of tested units, picked once (seeded) so results repeat
        tested = feats[feats["has_oil_test"] == 1][OIL_COLUMNS].reset_index(drop=True)
        pick = np.random.default_rng(SEED).choice(len(tested), size=min(N_DONORS, len(tested)), replace=False)
        self.donors = tested.iloc[pick].reset_index(drop=True)
        # POLICY FLOOR (Annex A: an untested unit carries a wide interval). We may not claim to be more certain about a unit we have
        # never looked at than about the 90th percentile of tested units. This is a stated rule, not a measured uncertainty.
        self.untested_floor = 0.0
        tested_rows = feats[feats["has_oil_test"] == 1]
        if len(tested_rows):
            d = np.vstack([m.predict_proba(tested_rows)[:, 1] for m in self.bootstrap])
            self.untested_floor = float(np.percentile(np.percentile(d, 95, axis=0) - np.percentile(d, 5, axis=0), 90))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.calibrated.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict_raw(self, X) -> np.ndarray:
        return self.uncalibrated.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def predict_interval(self, X):
        """Calibrated probability and its 90% range (5th to 95th percentile).

        Two sources of doubt are pooled. (1) Model doubt: 30 bootstrap copies refitted on resampled transformers. (2) Missing-test
        doubt, only for units with NO usable oil test: the probability if that test existed and had shown what a randomly chosen
        tested unit shows (30 scenarios). A unit we have never looked at can hide a fault, so its range must be wide; this is also
        exactly the 'hidden risk an inspection could reveal' that plan.py prices (p_high - p)."""
        feats = X[FEATURE_COLUMNS].reset_index(drop=True)
        p = self.predict_proba(feats)
        draws = np.vstack([m.predict_proba(feats)[:, 1] for m in self.bootstrap])          # n_copies x n_units
        untested = (feats["has_oil_test"] != 1).to_numpy()
        if untested.any() and len(self.donors):
            scenarios = []
            for _, donor in self.donors.iterrows():
                what_if = feats.loc[untested].copy()
                for col in OIL_COLUMNS:
                    what_if[col] = donor[col]
                scenarios.append(self.calibrated.predict_proba(what_if)[:, 1])
            extra = np.full((len(scenarios), len(feats)), np.nan)
            extra[:, untested] = np.vstack(scenarios)
            draws = np.vstack([draws, extra])                                                  # tested units: scenario rows stay NaN
        lo, hi = np.nanpercentile(draws, 5, axis=0), np.nanpercentile(draws, 95, axis=0)
        lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        # apply the policy floor to untested units: widen around p (inside 0..1) until the range is at least the floor
        floor = getattr(self, "untested_floor", 0.0)
        short = untested & ((hi - lo) < floor)
        lo[short] = np.maximum(0.0, np.minimum(lo[short], p[short] - floor / 2))
        hi[short] = np.minimum(1.0, np.maximum(hi[short], lo[short] + floor))
        return p, lo, hi

    def contributions(self, X: pd.DataFrame, top: int = 3) -> list:
        """Why each unit scored as it did. Logistic regression: coefficient x scaled feature value (log-odds, exact for the
        raw model). Boosting: occlusion (replace one feature by its training median and see how the probability moves).
        Describes the model, does not prove a cause."""
        feats = X[FEATURE_COLUMNS]
        clf = self.uncalibrated.named_steps["clf"]
        if hasattr(clf, "coef_"):
            names = self.uncalibrated[:-1].get_feature_names_out()
            z = self.uncalibrated[:-1].transform(feats)
            contrib = z * clf.coef_[0]
        else:
            base = self.predict_proba(X)
            contrib = np.column_stack([base - self.predict_proba(X.assign(**{f: self.medians[f]})) for f in FEATURE_COLUMNS])
            names = np.array(FEATURE_COLUMNS)
        out = []
        for r in range(len(feats)):
            order = np.argsort(-np.abs(contrib[r]))[:top]
            out.append([{"feature": str(names[j]), "contribution": round(float(contrib[r, j]), 4)} for j in order])
        return out


# =============================================================================================
# 7. score() and model_info()  (the hand-off to the API, Contract section 7)
# =============================================================================================
_cache = {}


def load_model(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / MODEL_PATH
    if str(p) not in _cache:
        _cache[str(p)] = joblib.load(p)
    return _cache[str(p)]


def score(features: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """In: the output of build_features. Out: one row per transformer with the columns the contract lists, plus
    has_oil_test, months_since_last_test and age_years, which plan.py needs for its replace-not-allowed rule."""
    bundle = bundle or load_model()
    fm = bundle["model"]
    f = features.reset_index(drop=True)
    p, lo, hi = fm.predict_interval(f)
    health = health_table(f)
    no_operation = f["age_years"].isna() & f["load_mean_12m"].isna() & (f["has_oil_test"] != 1)

    months = f["months_since_last_test"]
    low_conf = (f["has_oil_test"] != 1) | (months > STALE_MONTHS) | (f["n_tests"] == 1)
    status = np.where(no_operation, "insufficient_data", np.where(low_conf, "scored_low_confidence", "scored"))

    quality = []
    for i in range(len(f)):
        age = f"age {f.loc[i, 'age_years']:.0f} years" if pd.notna(f.loc[i, "age_years"]) else "age unknown"
        if f.loc[i, "has_oil_test"] != 1:
            quality.append(f"no oil test; {age}")
        else:
            note = "; stale test" if months[i] > STALE_MONTHS else ("; only one test" if f.loc[i, "n_tests"] == 1 else "")
            quality.append(f"{int(f.loc[i, 'n_tests'])} oil test(s), last {months[i]:.0f} months ago{note}")

    out = pd.DataFrame({"transformer_id": f["transformer_id"], "failure_prob_12m": p, "prob_low": lo, "prob_high": hi,
                        "status": status, "health_score": health["health_score"].round(0), "top_drivers": fm.contributions(f),
                        "data_quality": quality, "model_version": bundle["model_version"],
                        "has_oil_test": f["has_oil_test"], "months_since_last_test": months, "age_years": f["age_years"]})
    for col in ("failure_prob_12m", "prob_low", "prob_high"):                  # nothing to say about a unit with no data
        out.loc[no_operation, col] = np.nan
    return out


def model_info() -> dict:
    """The model card (HCL Annex C fields) as saved by train()."""
    return json.loads((ROOT / CARD_PATH).read_text(encoding="utf-8"))


# =============================================================================================
# 8. train(): everything above, in order
# =============================================================================================
def _read_warehouse():
    """Real inputs from B's warehouse: fact_snapshot, dim_transformer, silver_costs, silver_budget."""
    from src.pipeline import get_connection                       # B's file
    conn = get_connection()
    snap = pd.read_sql("SELECT * FROM fact_snapshot", conn)
    dim = pd.read_sql("SELECT * FROM dim_transformer", conn)
    costs = pd.read_sql("SELECT * FROM silver_costs", conn)
    budget = float(pd.read_sql("SELECT annual_budget FROM silver_budget", conn)["annual_budget"].iloc[0])
    return snap, dim, costs, budget


def train(snapshots: pd.DataFrame | None = None, dim_transformer: pd.DataFrame | None = None,
          costs: pd.DataFrame | None = None, budget: float | None = None, out_dir: Path | str = ROOT,
          n_boot: int = N_BOOTSTRAP, n_repeats: int = N_REPEATS, verbose: bool = True) -> dict:
    """Train, evaluate and save. With no arguments it reads B's warehouse. For development on a stand-in table, pass
    snapshots (and optionally dim_transformer, costs, budget) and an out_dir so real outputs are not overwritten."""
    out_dir = Path(out_dir)
    say = print if verbose else (lambda *a, **k: None)
    if snapshots is None:
        snapshots, dim_transformer, costs, budget = _read_warehouse()
    for col in FEATURE_COLUMNS + ["transformer_id", "snapshot_date", "failed_next_12m"]:
        assert col in snapshots.columns, f"fact_snapshot is missing column {col}"

    split = make_split(snapshots)
    dev, test, labelled = split["dev"], split["test"], split["labelled"]
    window = [str(dev["snapshot_date"].min().date()), str(dev["snapshot_date"].max().date())]
    say(f"  split   : dev {len(dev):,} rows / {int(dev['failed_next_12m'].sum())} failures ({window[0]} to {window[1]}); "
        f"held-out {len(test):,} rows / {int(test['failed_next_12m'].sum())} failures, {test['transformer_id'].nunique()} transformers")

    # ---- validation of every configuration on the development set --------------------------------------------
    candidates = [("age_only", "-", lambda: AgeOnly()), ("health_index", "-", lambda: HealthIndexBaseline())] + \
                 [(m, k, (lambda m=m, k=k: SkModel(make_pipeline(m, k)))) for m, k in CONFIGS]
    summary = {}
    for model, method, factory in candidates:
        t0 = time.perf_counter()
        runs = grouped_time_aware_cv(dev, factory, n_repeats)
        s = summarise_runs(runs)
        s["fit_seconds"] = time.perf_counter() - t0
        summary[(model, method)] = s
        log_experiment({"timestamp": datetime.now().isoformat(timespec="seconds"), "stage": "cv", "model": model,
                        "missing_method": method, "data_window": f"{window[0]}..{window[1]}", "params": json.dumps(PARAMS.get(model, {})),
                        "seed": SEED, **s}, out_dir)
        say(f"  cv      : {model:<13}{method:<17} PR-AUC {s['pr_auc_mean']:.3f} (range {s['pr_auc_min']:.3f} to {s['pr_auc_max']:.3f})  "
            f"recall@10% {s['recall_top10_mean']:.2f}  Brier {s['brier_mean']:.4f}")

    # ---- pick the final configuration by a rule fixed beforehand ------------------------------------------
    cand = {c: summary[c] for c in CONFIGS}
    best_c = max(cand, key=lambda c: cand[c]["pr_auc_mean"])
    cutoff = cand[best_c]["pr_auc_mean"] - cand[best_c]["pr_auc_sd"]
    chosen = next(c for c in CONFIGS if cand[c]["pr_auc_mean"] >= cutoff)             # simplest within one spread of the best
    say(f"  pick    : best by PR-AUC is {best_c}; simplest within one spread is {chosen}")

    # ---- fit on dev, calibrate, intervals; then the held-out set, ONCE ---------------------------------------
    t0 = time.perf_counter()
    fm = FinalModel(*chosen).fit(dev, n_boot=n_boot)
    fit_seconds = time.perf_counter() - t0
    baselines = {"age_only": AgeOnly().fit(dev, dev["failed_next_12m"]), "health_index": HealthIndexBaseline().fit(dev, dev["failed_next_12m"])}
    test = test.copy()
    test["p_calibrated"], test["p_uncalibrated"] = fm.predict_proba(test), fm.predict_raw(test)
    test["p_age_only"] = baselines["age_only"].predict_proba(test)
    test["p_health_index"] = baselines["health_index"].predict_proba(test)
    final = {c: cluster_bootstrap(test, c) for c in ["p_calibrated", "p_uncalibrated", "p_age_only", "p_health_index"]}
    no_test_rows = test[test["has_oil_test"] == 0]
    no_test = cluster_bootstrap(no_test_rows, "p_calibrated") if len(no_test_rows) else None
    t0 = time.perf_counter()
    p_i, lo_i, hi_i = fm.predict_interval(test)
    ms_per_unit = 1000 * (time.perf_counter() - t0) / max(len(test), 1)
    widths = pd.Series(hi_i - lo_i)
    width_check = {"median_width_no_oil_test": float(widths[(test["has_oil_test"] == 0).to_numpy()].median()),
                   "median_width_with_oil_test": float(widths[(test["has_oil_test"] == 1).to_numpy()].median())}
    say(f"  holdout : calibrated PR-AUC {final['p_calibrated']['pr_auc']['value']:.3f} vs age-only {final['p_age_only']['pr_auc']['value']:.3f} "
        f"and health index {final['p_health_index']['pr_auc']['value']:.3f}; Brier {final['p_uncalibrated']['brier']['value']:.4f} -> "
        f"{final['p_calibrated']['brier']['value']:.4f} after calibration")
    say(f"  ranges  : median 90% range width, no oil test {width_check['median_width_no_oil_test']:.3f} vs with oil test "
        f"{width_check['median_width_with_oil_test']:.3f}")
    log_experiment({"timestamp": datetime.now().isoformat(timespec="seconds"), "stage": "holdout", "model": chosen[0],
                    "missing_method": chosen[1], "data_window": f"{test['snapshot_date'].min().date()}..{test['snapshot_date'].max().date()}",
                    "n_rows": len(test), "n_positives": int(test["failed_next_12m"].sum()), "params": json.dumps(PARAMS[chosen[0]]), "seed": SEED,
                    "pr_auc_mean": final["p_calibrated"]["pr_auc"]["value"], "recall_top10_mean": final["p_calibrated"]["recall_top10"]["value"],
                    "lift_top10_mean": final["p_calibrated"]["lift_top10"]["value"], "brier_mean": final["p_calibrated"]["brier"]["value"],
                    "no_test_pr_auc_mean": no_test["pr_auc"]["value"] if no_test else np.nan, "fit_seconds": fit_seconds}, out_dir)

    # ---- global importance, reliability curve, budget test ---------------------------------------------------
    perm = permutation_importance(fm.uncalibrated, dev[FEATURE_COLUMNS], dev["failed_next_12m"], scoring="average_precision",
                                  n_repeats=5, random_state=SEED, n_jobs=1)
    importance = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
    plot_reliability(test, out_dir)
    budget_result = budget_test(test, fm, dim_transformer, costs, budget, len(snapshots["transformer_id"].unique()))
    if budget_result:
        say(f"  budget  : model plan removes {budget_result['model_plan']['risk_removed']:,.0f} risk vs age plan "
            f"{budget_result['age_plan']['risk_removed']:,.0f}; real failures in chosen units: "
            f"{budget_result['observed']['model_failures']} vs {budget_result['observed']['age_failures']}")

    # ---- deployed model: same configuration refitted on every labelled row (after the single held-out evaluation) -----
    deployed = FinalModel(*chosen).fit(labelled, n_boot=n_boot)
    version = f"{chosen[0]}-{TODAY}-a"
    card = {"model_card": {
        "name": "Transformer failure within 12 months", "version": version,
        "algorithm": f"{chosen[0]} with {chosen[1]} missing-data handling, Platt (sigmoid) calibration, {len(fm.bootstrap)}-copy bootstrap interval",
        "snapshot_design": f"Quarterly snapshots, label = failure in the next {HORIZON_MONTHS} months; labelable {labelled['snapshot_date'].min().date()} to {labelled['snapshot_date'].max().date()}",
        "validation": f"Grouped by transformer, time-aware (train on earlier snapshots), {n_repeats} repeats (seeds {SEED} to {SEED + n_repeats - 1}); "
                      f"held-out = last {N_HOLDOUT_SNAPSHOTS} labelable snapshots x {int(HOLDOUT_FRACTION * 100)}% of transformers, used once, with a 12-month purge",
        "missing_data_methods_compared": ["median imputation", "median + missing indicators", "native missing support (boosting)"],
        "baselines": {"age_only": _pack(final["p_age_only"]), "rule_health_index": _pack(final["p_health_index"])},
        "final": _pack(final["p_calibrated"]), "final_uncalibrated": _pack(final["p_uncalibrated"]),
        "no_oil_test_units": _pack(no_test) if no_test else "no failures among held-out units without an oil test",
        "budget_test": {"risk_reduced_vs_age_plan_pct": round(budget_result["gain_pct"], 1) if budget_result and budget_result["gain_pct"] is not None else None}
                       if budget_result else "not run: no cost table",
        "interval_width_check": width_check,
        "beats_age_only_in_cv": bool(summary[chosen]["pr_auc_mean"] > summary[("age_only", "-")]["pr_auc_mean"]),
        "deployed_model": "refitted on all labelled rows with the same configuration after the held-out evaluation; reported metrics are for the development fit",
        "known_limits": f"Only {int(dev['failed_next_12m'].sum())} failure rows in development data and {int(test['failed_next_12m'].sum())} in the held-out set; intervals are wide "
                        "and describe model spread, not a guarantee. Synthetic data. Replace value assumes a new unit has a 1% yearly failure chance.",
        "random_seed": SEED}}
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": deployed, "model_version": version, "trained_at": datetime.now().isoformat(timespec="seconds"),
                 "feature_columns": FEATURE_COLUMNS, "config": chosen}, out_dir / MODEL_PATH)
    (out_dir / CARD_PATH).write_text(json.dumps(card, indent=2, default=float), encoding="utf-8")
    _cache.clear()
    write_evaluation(out_dir, split, summary, chosen, final, no_test, width_check, importance, budget_result, ms_per_unit, fit_seconds)
    say(f"  saved   : {out_dir / MODEL_PATH} (version {version}) and reports in {out_dir / 'reports'}")
    return {"model_version": version, "chosen": chosen, "cv": {f"{m}+{k}": v for (m, k), v in summary.items()}, "holdout": final,
            "no_test": no_test, "width_check": width_check, "budget": budget_result, "card": card}


def _pack(m: dict) -> dict:
    return {"pr_auc": round(m["pr_auc"]["value"], 3), "pr_auc_ci90": [round(m["pr_auc"]["lo"], 3), round(m["pr_auc"]["hi"], 3)],
            "top10pct_recall": round(m["recall_top10"]["value"], 3), "lift_top10": round(m["lift_top10"]["value"], 2),
            "brier": round(m["brier"]["value"], 4), "n_pos": m["n_pos"], "n_rows": m["n_rows"]}


def plot_reliability(test: pd.DataFrame, out_dir: Path) -> None:
    """Predicted vs observed failure rate per bin, before and after calibration (the nearer the diagonal the better)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for col, label, color in (("p_uncalibrated", "uncalibrated", "#c77a12"), ("p_calibrated", "calibrated", "#2f5fd0")):
        obs, pred = calibration_curve(test["failed_next_12m"], test[col], n_bins=4, strategy="quantile")
        ax.plot(pred, obs, "o-", color=color, label=label)
    ax.set_xlabel("mean predicted probability"), ax.set_ylabel("observed failure rate")
    ax.set_title("Reliability on the held-out set"), ax.legend()
    top = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, top), ax.set_ylim(0, top)
    fig.tight_layout()
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "reports" / "reliability.png", dpi=140)
    plt.close(fig)


def budget_test(test, fm, dim_transformer, costs, budget, fleet_size):
    """HCL section 7: on the held-out snapshot, our plan vs the age-based plan, by risk removed and real failures caught.
    The fleet budget is scaled to the size of the held-out group."""
    if dim_transformer is None or costs is None or budget is None:
        return None
    last = test["snapshot_date"].max()
    rows = test[test["snapshot_date"] == last].reset_index(drop=True)
    p, lo, hi = fm.predict_interval(rows)
    scores = pd.DataFrame({"transformer_id": rows["transformer_id"], "failure_prob_12m": p, "prob_low": lo, "prob_high": hi,
                           "has_oil_test": rows["has_oil_test"], "months_since_last_test": rows["months_since_last_test"],
                           "age_years": rows["age_years"]})
    risk = P.compute_risk(scores, dim_transformer)
    scaled = float(budget) * len(rows) / fleet_size
    mine, age = P.make_plan(risk, costs, scaled), P.age_plan(risk, costs, scaled)
    truth = rows.set_index("transformer_id")["failed_next_12m"]
    caught = lambda plan: int(sum(truth[a["transformer_id"]] for a in plan["actions"]))
    gain = 100 * (mine["risk_removed"] / age["risk_removed"] - 1) if age["risk_removed"] > 0 else None
    return {"as_of": str(last.date()), "units": len(rows), "budget_scaled": scaled, "model_plan": mine, "age_plan": age,
            "gain_pct": gain, "observed": {"model_failures": caught(mine), "age_failures": caught(age)}}


def write_evaluation(out_dir, split, summary, chosen, final, no_test, width_check, importance, budget, ms_per_unit, fit_seconds):
    """A first draft of reports/evaluation.md straight from the numbers; read it and edit it before you submit."""
    dev, test = split["dev"], split["test"]
    lines = ["# Evaluation report (generated by `python -m src.model`; read it, then edit it)", "",
             "## Split and method",
             f"- Held-out set: snapshots {', '.join(str(pd.Timestamp(d).date()) for d in split['test_dates'])} x {len(split['holdout_ids'])} transformers "
             f"(random {int(HOLDOUT_FRACTION * 100)}%, seed {SEED}, stratified by failed-ever / tested-ever): {len(test)} rows, {int(test['failed_next_12m'].sum())} failure rows. Used once.",
             f"- Development set: the other transformers at snapshots whose 12-month label ends before the first held-out snapshot: {len(dev)} rows, "
             f"{int(dev['failed_next_12m'].sum())} failure rows.",
             f"- Validation: transformers shuffled into 5 groups; train on the other groups at EARLIER snapshots, validate on the group; {N_REPEATS} repeats, seeds {SEED}..{SEED + N_REPEATS - 1}; "
             "metrics are mean (range over repeats).",
             "- All imputation and scaling sit inside a scikit-learn Pipeline, fitted on training rows only.", "",
             "## Metric definitions",
             "PR-AUC: area under precision-recall (the right headline for rare events; no-skill level equals the failure rate). Top-10% recall: share of real failures "
             "in the riskiest 10% of rows. Lift: that share divided by 0.10. Brier: mean squared error of the probabilities (lower is better).", "",
             "## Comparison on development data (time-aware grouped CV)", "",
             "| Model | Missing data | PR-AUC mean (range) | Top-10% recall | Brier | No-oil-test PR-AUC |", "|---|---|---|---|---|---|"]
    for (m, k), s in summary.items():
        lines.append(f"| {m} | {k} | {s['pr_auc_mean']:.3f} ({s['pr_auc_min']:.3f} to {s['pr_auc_max']:.3f}) | {s['recall_top10_mean']:.2f} | "
                     f"{s['brier_mean']:.4f} | {s['no_test_pr_auc_mean']:.3f} |")
    age_cv, chosen_cv = summary[("age_only", "-")]["pr_auc_mean"], summary[chosen]["pr_auc_mean"]
    verdict = ("beats" if chosen_cv > age_cv else "does NOT beat")
    lines += ["", f"Chosen: **{chosen[0]} + {chosen[1]}** (simplest model within one repeat-spread of the best PR-AUC).",
              f"**Verdict against the age-only baseline in CV: the chosen model {verdict} it (mean PR-AUC {chosen_cv:.3f} vs {age_cv:.3f}).** "
              "A complex model must earn its place; say this plainly in the demo whichever way it falls.", "",
              "## Final held-out result (used once)", "", "| Model | PR-AUC [90% CI] | Top-10% recall | Lift | Brier |", "|---|---|---|---|---|"]
    for label, key in (("Age only", "p_age_only"), ("Rule health index", "p_health_index"), ("Final, uncalibrated", "p_uncalibrated"), ("**Final, calibrated**", "p_calibrated")):
        m = final[key]
        lines.append(f"| {label} | {m['pr_auc']['value']:.3f} [{m['pr_auc']['lo']:.3f}, {m['pr_auc']['hi']:.3f}] | {m['recall_top10']['value']:.2f} | "
                     f"{m['lift_top10']['value']:.2f} | {m['brier']['value']:.4f} |")
    lines += ["", f"Failure rows in the held-out set: {final['p_calibrated']['n_pos']} of {final['p_calibrated']['n_rows']}. Intervals come from resampling whole transformers.", "",
              "## Calibration", f"Brier before calibration {final['p_uncalibrated']['brier']['value']:.4f}, after {final['p_calibrated']['brier']['value']:.4f}. "
              "Reliability curve: `reports/reliability.png`. Calibration matters because risk = probability x impact: a systematically wrong probability changes which transformers win the budget.", "",
              "## Units without an oil test"]
    lines.append(f"PR-AUC {no_test['pr_auc']['value']:.3f} [{no_test['pr_auc']['lo']:.3f}, {no_test['pr_auc']['hi']:.3f}] from {no_test['n_pos']} failure rows "
                 f"({no_test['n_rows']} rows). They are always `scored_low_confidence` with no health score." if no_test else "No failures among held-out units without a test, so nothing to report.")
    wider = width_check["median_width_no_oil_test"] > width_check["median_width_with_oil_test"]
    lines += ["", f"Median 90% range width: no oil test {width_check['median_width_no_oil_test']:.3f} vs with oil test {width_check['median_width_with_oil_test']:.3f} "
              f"-> {'wider for untested units, as intended' if wider else '**NOT wider for untested units: investigate before claiming honest uncertainty**'}.", "",
              "## Global feature importance (permutation, development data)", ""]
    lines += [f"- {name}: {value:.4f}" for name, value in importance.head(8).items()]
    lines += ["", "Importance shows what the model uses, not what causes failures.", "", "## Budget test (held-out snapshot)"]
    if budget:
        lines += [f"- Snapshot {budget['as_of']}, {budget['units']} units, budget scaled to {budget['budget_scaled']:,.0f}.",
                  f"- Model plan removes {budget['model_plan']['risk_removed']:,.0f} risk ({budget['model_plan']['risk_reduced_pct']:.1f}% of total) with {len(budget['model_plan']['actions'])} actions costing {budget['model_plan']['total_cost']:,.0f}.",
                  f"- Age plan removes {budget['age_plan']['risk_removed']:,.0f} risk with {len(budget['age_plan']['actions'])} replacements costing {budget['age_plan']['total_cost']:,.0f}.",
                  f"- Real failures among units each plan acted on: model plan {budget['observed']['model_failures']}, age plan {budget['observed']['age_failures']}."]
    else:
        lines.append("- Not run: no cost table was supplied.")
    lines += ["", "## Timing", f"Training the final model: {fit_seconds:.1f} s. Scoring with intervals: {ms_per_unit:.2f} ms per transformer.", "",
              "## Honest limits", "Few failures mean wide intervals; results on synthetic data describe the synthetic world; the held-out set is small. "
              "Read the final table before claiming the model beats the baselines."]
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print("\nModel training and evaluation (reads the warehouse)\n")
    train()
