"""STAND-IN table for developing src/model.py before B's real fact_snapshot exists (Member C brief, Step 2).

It has exactly the columns of Contract section 6 (plus transformer_id, snapshot_date, failed_next_12m), random values
and about 5% positives with a weak learnable signal (age, a developing gas fault, overload). It tests CODE, never claims
anything about real performance. DELETE THIS FILE (and the tests that use it) once the real data is in and tests run on it.
"""
import numpy as np
import pandas as pd

GASES = ["h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last", "co_last"]
SNAPSHOTS = pd.date_range("2024-01-01", "2026-10-01", freq="QS")
LABELABLE_UNTIL = pd.Timestamp("2025-10-01")


def make_standin(n: int = 400, seed: int = 42):
    """Returns (snapshots, dim_transformer, costs, budget)."""
    rng = np.random.default_rng(seed)
    ids = [f"TR-{i:04d}" for i in range(1, n + 1)]
    ctype = rng.choice(["hospital", "industry", "urban", "rural"], n, p=[0.1, 0.2, 0.35, 0.35])
    customers = np.array([int(rng.integers(5000, 20000)) if t == "hospital" else int(rng.integers(200, 3000)) for t in ctype])
    rating = rng.choice(["small", "medium", "large"], n, p=[0.5, 0.35, 0.15])
    dim = pd.DataFrame({"transformer_id": ids, "customer_type": ctype, "customers": customers, "rating_class": rating})

    age0 = rng.uniform(3, 48, n)
    base_load = np.clip(rng.normal(65, 15, n), 25, 115)
    tested = rng.random(n) < 0.62
    first_test = rng.integers(0, 6, n)                                   # snapshot index at which the first test exists
    faulty = rng.random(n) < 0.10
    onset = rng.integers(0, 10, n)
    baseline = {"h2_last": 40, "ch4_last": 30, "c2h2_last": 0.6, "c2h4_last": 15, "c2h6_last": 20, "co_last": 200}

    rows = []
    for si, snap in enumerate(SNAPSHOTS):
        for u in range(n):
            age = age0[u] + si * 0.25 + 0.2
            load_mean = float(np.clip(base_load[u] + rng.normal(0, 4), 20, 120))
            has_test = bool(tested[u] and si >= first_test[u])
            fault_on = bool(faulty[u] and si >= onset[u])
            row = {"transformer_id": ids[u], "snapshot_date": snap, "age_years": age, "load_mean_12m": load_mean,
                   "load_p95_12m": load_mean + rng.uniform(5, 20), "temp_mean_12m": 40 + 0.3 * load_mean + rng.normal(0, 2),
                   "hot_months_12m": int(rng.poisson(1.0)), "has_oil_test": int(has_test),
                   "n_tests": int(min(4, si - first_test[u] + 1)) if has_test else 0,
                   "months_since_last_test": float(rng.uniform(1, 22)) if has_test else np.nan,
                   "months_since_repair": np.nan}
            for g in GASES:
                v = baseline[g] * rng.lognormal(0, 0.3) * (1 + (0.8 * (si - onset[u]) if fault_on and g in ("c2h2_last", "ch4_last", "c2h4_last") else 0))
                row[g] = round(float(v), 1) if has_test else np.nan
            row["tdcg_last"] = float(np.nansum([row[g] for g in GASES])) if has_test else np.nan
            two_tests = has_test and row["n_tests"] >= 2
            row["tdcg_delta"] = float(rng.normal(3 + (25 if fault_on else 0), 5)) if two_tests else np.nan
            row["c2h2_delta"] = float(rng.normal(0.1 + (1.5 if fault_on else 0), 0.2)) if two_tests else np.nan
            row["ratio_c2h2_c2h4"] = row["c2h2_last"] / (row["c2h4_last"] + 0.1) if has_test else np.nan
            row["ratio_ch4_h2"] = row["ch4_last"] / (row["h2_last"] + 0.1) if has_test else np.nan
            row["ratio_c2h4_c2h6"] = row["c2h4_last"] / (row["c2h6_last"] + 0.1) if has_test else np.nan
            logit = -2.9 + 0.045 * (age - 30) + 2.0 * fault_on + 0.03 * (row["load_p95_12m"] - 80)
            fails = rng.random() < 1 / (1 + np.exp(-logit))
            row["failed_next_12m"] = float(fails) if snap <= LABELABLE_UNTIL else np.nan      # NULL when not labelable yet
            rows.append(row)
    snap_df = pd.DataFrame(rows)
    costs = pd.DataFrame({"action": ["replace"] * 3 + ["inspect"] * 3, "rating_class": ["small", "medium", "large"] * 2,
                          "cost": [150000, 350000, 700000, 10000, 15000, 25000]})
    return snap_df, dim, costs, 5_000_000.0
