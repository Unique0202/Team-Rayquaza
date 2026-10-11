"""Generate Member A's synthetic data: python scripts/generate_data.py.

Only the standard library, numpy and pandas are used. Hidden fault state is
kept in memory and in the data card, never in the source CSV feature columns.
"""

import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd


SEED = 42
N_ASSETS = 400
MONTHS = pd.date_range("2023-10-01", periods=36, freq="MS")
TODAY = "2026-10-11"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GASES = ["h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm", "co_ppm"]
ASSET_COLUMNS = ["transformer_id", "customer_type", "customers", "region", "rating_class"]
OP_COLUMNS = ["transformer_id", "month", "load_pct", "top_oil_temp_c", "age_years", "event"]
OIL_COLUMNS = ["transformer_id", "test_date", *GASES]
COST_COLUMNS = ["action", "rating_class", "cost", "annual_budget"]
USER_PROMPT = (
    "read the Team-Rayquaza/docs/team/A_DATA_AND_SERVING.md and also read the "
    "problem statement file properly and nalso the remainng fiels in the team folder, "
    "and start working on part 1.\n"
    "Team-Rayquaza/docs/team/A_DATA_AND_SERVING.md is my part whihc i have to do."
)


def choose(rng, candidates, count):
    """Sort candidates so set/dictionary ordering never changes the seeded draw."""
    return rng.choice(sorted(candidates), size=count, replace=False).tolist()


def make_assets(rng):
    """Create public asset fields and private condition attributes."""
    assets = pd.DataFrame({"transformer_id": [f"TR-{i:04d}" for i in range(1, N_ASSETS + 1)]})
    assets["customer_type"] = rng.permutation(
        ["hospital"] * 40 + ["industry"] * 80 + ["urban"] * 140 + ["rural"] * 140
    )
    ranges = {"hospital": (5000, 20000), "industry": (50, 2000),
              "urban": (2000, 15000), "rural": (200, 3000)}
    assets["customers"] = [int(rng.integers(ranges[t][0], ranges[t][1] + 1))
                           for t in assets.customer_type]
    assets["region"] = rng.choice(["North", "South", "East", "West", "Central"], 400)
    assets["rating_class"] = rng.permutation(["small"] * 200 + ["medium"] * 140 + ["large"] * 60)
    ages = np.concatenate([rng.uniform(2, 10, 100), rng.uniform(10, 30, 160),
                           rng.uniform(30, 50, 140)])
    assets["age_at_start"] = rng.permutation(ages)
    assets["base_load"] = np.clip(rng.normal(65, 15, 400), 20, 120)
    overload = choose(rng, assets.index, 10)
    assets.loc[overload, "base_load"] = rng.uniform(95, 115, 10)
    assets["load_trend"] = rng.permutation([0.2] * 80 + [0.0] * 320)
    less_tested = (assets.customer_type == "rural") | (assets.rating_class == "small")
    probability = np.where(less_tested, 0.50, 0.70)
    assets["is_tested"] = rng.random(400) < probability
    # Adjust to exactly 248 tested, preserving the lower sampling weight for neglected units.
    deficit = 248 - int(assets.is_tested.sum())
    if deficit:
        eligible = assets.index[assets.is_tested == (deficit < 0)].to_numpy()
        weights = probability[eligible] if deficit > 0 else 1 - probability[eligible]
        adjusted = rng.choice(eligible, abs(deficit), replace=False, p=weights / weights.sum())
        assets.loc[adjusted, "is_tested"] = deficit > 0

    old_few = choose(rng, assets.index[assets.is_tested & (assets.age_at_start > 40)], 6)
    schedules = {}
    for i in assets.index[assets.is_tested]:
        months = [int(rng.integers(0, 9))]
        if i not in old_few:
            while months[-1] + 6 <= 35:
                next_month = months[-1] + int(rng.integers(6, 13))
                if next_month > 35:
                    break
                months.append(next_month)
        schedules[i] = months

    # Select visible faults from independently drawn onsets, not from future failure labels.
    assets["fault_onset_month"] = rng.integers(0, 31, 400)
    eligible = [i for i, months in schedules.items()
                if i not in old_few and sum(m > assets.at[i, "fault_onset_month"] for m in months) >= 2]
    faulty = choose(rng, eligible, 32) + choose(rng, assets.index[~assets.is_tested], 8)
    assets["fault"] = "none"
    assets.loc[faulty, "fault"] = rng.permutation(
        ["thermal"] * 20 + ["partial_discharge"] * 12 + ["arcing"] * 8
    )

    eligible = [i for i in schedules if i not in old_few + overload + faulty
                and assets.at[i, "age_at_start"] < 20]
    pair_ids = choose(rng, eligible, 6)
    pairs = [pair_ids[i:i + 2] for i in range(0, 6, 2)]
    copied = ["age_at_start", "base_load", "load_trend", "rating_class", "fault", "fault_onset_month"]
    for first, second in pairs:
        assets.loc[second, copied] = assets.loc[first, copied].to_numpy()
        schedules[second] = schedules[first].copy()
        assets.loc[first, ["customer_type", "customers"]] = ["hospital", int(rng.integers(8000, 15001))]
        assets.loc[second, ["customer_type", "customers"]] = ["rural", int(rng.integers(200, 601))]

    # CO's baseline grows with age later; this stores a persistent per-unit variation factor.
    baseline_factors = rng.lognormal(0, 0.3, (400, 6))
    acetylene = np.full(400, 0.5)
    acetylene[choose(rng, assets.index, 80)] = rng.uniform(1, 2, 80)
    for first, second in pairs:
        baseline_factors[second] = baseline_factors[first]
        acetylene[second] = acetylene[first]
    return assets, schedules, old_few, pairs, overload, baseline_factors, acetylene


def simulate_operation(assets, load_noise, temp_noise, failure_draws, base, repairs):
    """Replay identical weather and failure draws when rescaling the base hazard."""
    rows = []
    active_at_test = np.zeros((400, 36), dtype=bool)
    exposure = {"active_fault_months": 0, "active_fault_failures": 0,
                "no_fault_months": 0, "no_fault_failures": 0}
    for asset in assets.itertuples():
        i = asset.Index
        cleared = False
        for m, date in enumerate(MONTHS):
            event = "none"
            if repairs.get(i) == m:
                event = "repair"
                cleared = True
            active = asset.fault != "none" and m >= asset.fault_onset_month and not cleared
            season = np.sin(2 * np.pi * (date.month - 2) / 12)
            load = float(np.clip(asset.base_load + asset.load_trend * m + 8 * season + load_noise[i, m], 5, 150))
            temp = 25 + 10 * season + 0.45 * load + temp_noise[i, m]
            if active and asset.fault == "thermal":
                temp += 5
            age = asset.age_at_start + m / 12
            age_mult = np.exp((age - 25) / 15)
            load_mult = 1 + max(0, load - 85) / 20
            fault_mult = 1 + 0.6 * min(m - asset.fault_onset_month, 15) if active else 1
            hazard = base * age_mult * load_mult * fault_mult * (1 if asset.is_tested else 1.4)
            assert 0 <= hazard <= 1, "Monthly probability is outside [0, 1]"
            group = "active_fault" if active else "no_fault"
            exposure[group + "_months"] += 1
            if failure_draws[i, m] < hazard:
                assert event != "repair", "Planted repair conflicts with a stochastic failure"
                event = "failure"
                # A repair clears today's fault, not a fault whose onset is still in the future.
                if active:
                    cleared = True
                exposure[group + "_failures"] += 1
            # Events occur at month start; even a test later in the same month sees the reset.
            active_at_test[i, m] = active and not cleared
            rows.append([asset.transformer_id, date.strftime("%Y-%m-%d"),
                         round(load, 1), round(float(np.clip(temp, 15, 120)), 1), round(age, 2), event])
    return pd.DataFrame(rows, columns=OP_COLUMNS), active_at_test, exposure


def gas_values(assets, i, m, active, factors, acetylene, rng):
    """Apply the spec's gas pattern to a persistent baseline, then measurement noise."""
    asset = assets.loc[i]
    age = asset.age_at_start + m / 12
    values = np.array([40, 30, 0.5, 15, 20, 150 + 6 * age]) * factors[i]
    values[2] = acetylene[i]
    if active:
        k = m - asset.fault_onset_month
        if asset.fault == "thermal":
            values[[1, 3, 4]] *= [1 + 0.12 * k, 1 + 0.25 * k, 1 + 0.08 * k]
        elif asset.fault == "partial_discharge":
            values[[0, 1]] *= [1 + 0.30 * k, 1 + 0.05 * k]
        elif asset.fault == "arcing":
            values[2] += 2 * k
            values[[0, 3]] *= [1 + 0.20 * k, 1 + 0.10 * k]
    return np.round(np.maximum(0.5, values * rng.lognormal(0, 0.2, 6)), 1)


def make_oil_tests(assets, schedules, pairs, active, factors, acetylene, rng):
    rows = []
    twins = {second: first for first, second in pairs}
    for i, months in schedules.items():
        if i in twins:
            continue
        for m in months:
            day = int(rng.integers(1, MONTHS[m].days_in_month + 1))
            date = MONTHS[m].replace(day=day).strftime("%Y-%m-%d")
            values = gas_values(assets, i, m, active[i, m], factors, acetylene, rng)
            rows.append([assets.at[i, "transformer_id"], date, *values])
    oil = pd.DataFrame(rows, columns=OIL_COLUMNS)
    for first, second in pairs:
        twin = oil.loc[oil.transformer_id == assets.at[first, "transformer_id"]].copy()
        twin["transformer_id"] = assets.at[second, "transformer_id"]
        twin[GASES] = np.round(np.maximum(0.5, twin[GASES].to_numpy() * rng.lognormal(0, 0.05, (len(twin), 6))), 1)
        oil = pd.concat([oil, twin], ignore_index=True)
    return oil.sort_values(["transformer_id", "test_date"]).reset_index(drop=True)


def make_costs(budget):
    return pd.DataFrame([
        ["replace", "small", 150000, budget], ["replace", "medium", 350000, budget],
        ["replace", "large", 700000, budget], ["inspect", "small", 10000, budget],
        ["inspect", "medium", 15000, budget], ["inspect", "large", 25000, budget],
    ], columns=COST_COLUMNS)


def plant_dirty_rows(assets, operation, oil, excluded, rng):
    """Plant corrupt units, asset duplicates and gaps without erasing event labels."""
    edges = {}
    counts = oil.groupby("transformer_id").size()
    eligible = [i for i in assets.index if i not in excluded
                and assets.at[i, "fault"] == "none"
                and counts.get(assets.at[i, "transformer_id"], 0) >= 3]
    unit_jumps = choose(rng, eligible, 3)
    edges["E4_unit_jump"] = []
    for i in unit_jumps:
        indices = oil.index[oil.transformer_id == assets.at[i, "transformer_id"]]
        row = int(rng.choice(indices[1:]))
        oil.loc[row, GASES] *= 1000
        edges["E4_unit_jump"].append({"transformer_id": oil.at[row, "transformer_id"],
                                      "test_date": oil.at[row, "test_date"]})

    customers = assets[ASSET_COLUMNS].copy()
    duplicates = choose(rng, [i for i in assets.index if i not in excluded + unit_jumps + [44, 45]], 3)
    customers = pd.concat([customers, customers.loc[duplicates]], ignore_index=True)
    messy = assets.loc[[44, 45], ASSET_COLUMNS].copy()
    messy["transformer_id"] = ["tr-0045", "TR-0046 "]
    customers = pd.concat([customers, messy], ignore_index=True)
    edges["E7_duplicate_assets"] = assets.loc[duplicates, "transformer_id"].tolist() + ["TR-0045", "TR-0046"]
    edges["E7_messy_ids"] = ["tr-0045", "TR-0046 "]

    # Gaps represent unavailable telemetry, not an unrecorded failure or repair.
    dropped = choose(rng, operation.index[operation.event == "none"], round(len(operation) * 0.02))
    edges["E8_missing_months"] = operation.loc[dropped, ["transformer_id", "month"]].to_dict("records")
    operation = operation.drop(index=dropped).copy()
    for column in ["load_pct", "top_oil_temp_c"]:
        missing = choose(rng, operation.index, round(len(operation) * 0.01))
        operation.loc[missing, column] = np.nan
        edges["E8_missing_" + column] = operation.loc[missing, ["transformer_id", "month"]].to_dict("records")

    eligible = [i for i in assets.index if i not in excluded + unit_jumps]
    repeated_units = choose(rng, eligible, 2)
    repeated = []
    for i in repeated_units:
        eligible_rows = operation.index[(operation.transformer_id == assets.at[i, "transformer_id"])
                                       & (operation.event == "none")]
        repeated.extend(choose(rng, eligible_rows, 3))
    edges["E7_duplicate_operation"] = operation.loc[repeated, ["transformer_id", "month"]].to_dict("records")
    operation = pd.concat([operation, operation.loc[repeated]], ignore_index=True)
    return customers, operation, oil, edges


def visible_faults(assets, oil, active):
    """Count actual rising marker gases on at least two tests during an active fault."""
    markers = {"thermal": "c2h4_ppm", "partial_discharge": "h2_ppm", "arcing": "c2h2_ppm"}
    visible = []
    for asset in assets.loc[assets.fault != "none"].itertuples():
        tests = oil.loc[oil.transformer_id == asset.transformer_id].sort_values("test_date")
        values = []
        for test in tests.itertuples():
            date = pd.Timestamp(test.test_date)
            m = (date.year - 2023) * 12 + date.month - 10
            if active[asset.Index, m] and m > asset.fault_onset_month:
                values.append(getattr(test, markers[asset.fault]))
        if len(values) >= 2 and any(b > a for a, b in zip(values, values[1:])):
            visible.append(asset.transformer_id)
    return visible


def build_dataset():
    rng = np.random.default_rng(SEED)
    assets, schedules, old_few, pairs, overload, factors, acetylene = make_assets(rng)
    load_noise = rng.normal(0, 4, (400, 36))
    temp_noise = rng.normal(0, 2, (400, 36))
    failure_draws = rng.random((400, 36))
    # Look-alikes share weather and event draws, preserving similar measured histories.
    for first, second in pairs:
        load_noise[second] = load_noise[first]
        temp_noise[second] = temp_noise[first]
        failure_draws[second] = failure_draws[first]
    initial, _, _ = simulate_operation(assets, load_noise, temp_noise, failure_draws, 0.0012, {})
    initial_count = int((initial.event == "failure").sum())
    assert initial_count > 0
    base = 0.0012 * 50 / initial_count
    operation, active, exposure = simulate_operation(assets, load_noise, temp_noise, failure_draws, base, {})

    # Plant planned maintenance on faults still active at the end, so no failure is overwritten.
    eligible = [i for i in schedules if active[i, 35] and assets.at[i, "fault_onset_month"] <= 24
                and sum(active[i, m] for m in schedules[i]) >= 2]
    repair_ids = choose(rng, eligible, 3)
    repairs = {i: int(rng.integers(31, 34)) for i in repair_ids}
    for i, m in repairs.items():
        schedules[i] = sorted(set(schedules[i] + [m - 1, 35]))
    operation, active, exposure = simulate_operation(assets, load_noise, temp_noise, failure_draws, base, repairs)
    oil = make_oil_tests(assets, schedules, pairs, active, factors, acetylene, rng)
    visible = visible_faults(assets, oil, active)

    # October observations continue surviving faults; they do not invent new failure labels.
    untested_faults = assets.index[(~assets.is_tested) & active[:, 35]].tolist()
    first_tests = choose(rng, untested_faults, 2)
    first_tests += choose(rng, assets.index[(~assets.is_tested) & (~assets.index.isin(first_tests))], 8)
    already_tested = choose(rng, assets.index[assets.is_tested], 30)
    late_rows = []
    for i in already_tested + first_tests:
        date = f"2026-10-{int(rng.integers(1, 11)):02d}"
        values = gas_values(assets, i, 36, active[i, 35], factors, acetylene, rng)
        late_rows.append([assets.at[i, "transformer_id"], date, *values])
    late_oil = pd.DataFrame(late_rows, columns=OIL_COLUMNS)
    excluded = old_few + repair_ids + [i for pair in pairs for i in pair]
    customers, operation, oil, edges = plant_dirty_rows(assets, operation, oil, excluded, rng)
    costs = make_costs(5000000)
    frames = {"customers_served.csv": customers, "operation.csv": operation,
              "oil_tests.csv": oil, "costs_budget.csv": costs,
              "late_drop/oil_tests_late.csv": late_oil,
              "late_drop/costs_budget_late.csv": make_costs(4000000)}

    edges["E1_no_oil_test"] = assets.loc[~assets.is_tested, "transformer_id"].tolist()
    edges["E2_old_few_tests"] = assets.loc[old_few, "transformer_id"].tolist()
    edges["E3_lookalike_pairs"] = [[assets.at[a, "transformer_id"], assets.at[b, "transformer_id"]] for a, b in pairs]
    edges["E5_repair_reset"] = [{"transformer_id": assets.at[i, "transformer_id"],
                                "repair_month": MONTHS[m].strftime("%Y-%m-%d")} for i, m in repairs.items()]
    replace_costs = costs.loc[costs.action == "replace"].set_index("rating_class").cost
    faulty_assets = assets.loc[assets.fault != "none"]
    faulty_cost = int(faulty_assets.rating_class.map(replace_costs).sum())
    edges["E6_budget_shortfall"] = {"transformer_ids": faulty_assets.transformer_id.tolist(),
                                   "total_replace_cost": faulty_cost, "annual_budget": 5000000}
    failures = operation.loc[operation.event == "failure"].drop_duplicates(["transformer_id", "month"])
    rates = {"active_fault": exposure["active_fault_failures"] / exposure["active_fault_months"],
             "no_fault": exposure["no_fault_failures"] / exposure["no_fault_months"]}
    card = {
        "name": "transformer_fleet_synthetic", "version": "1.0", "created": TODAY, "seed": SEED,
        "row_counts": {name: len(frame) for name, frame in frames.items() if "/" not in name},
        "late_drop_row_counts": {name: len(frame) for name, frame in frames.items() if "/" in name},
        "date_ranges": {
            "operation.csv": [operation.month.min(), operation.month.max()],
            "oil_tests.csv": [oil.test_date.min(), oil.test_date.max()],
            "late_drop/oil_tests_late.csv": [late_oil.test_date.min(), late_oil.test_date.max()],
        },
        "n_transformers": 400, "n_untested": len(edges["E1_no_oil_test"]),
        "n_failures": len(failures), "n_failures_months_0_23": int((failures.month < "2025-10-01").sum()),
        "n_visible_developing_faults": len(visible), "visible_developing_fault_ids": visible,
        "initial_base_hazard": 0.0012, "initial_failure_count": initial_count, "base_hazard": base,
        "failure_exposure": exposure, "monthly_failure_rates": rates, "edge_cases": edges,
        "overloaded_units": assets.loc[overload, "transformer_id"].tolist(),
        "faulty_units": [{"transformer_id": a.transformer_id, "fault": a.fault,
                          "onset_month": MONTHS[a.fault_onset_month].strftime("%Y-%m-%d")}
                         for a in faulty_assets.itertuples()],
        "late_drop": {"already_tested": assets.loc[already_tested, "transformer_id"].tolist(),
                      "first_tests": assets.loc[first_tests, "transformer_id"].tolist(),
                      "first_test_faults": assets.loc[first_tests[:2], "transformer_id"].tolist(),
                      "as_of_for_rehearsal": TODAY},
        "provenance": {"assistant": "OpenAI Codex (GPT-6 family; exact deployment identifier not exposed)",
                       "user_prompt": USER_PROMPT,
                       "specifications": ["docs/DATA_SPEC.md", "docs/team/A_DATA_AND_SERVING.md", "docs/team/SHARED_CONTRACT.md"],
                       "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "checks": {},
    }
    validate(frames, card, assets, active)
    return frames, card


def validate(frames, card, assets, active):
    """Assert the source contract and every planted recipe before writing any output."""
    customers, op, oil, costs = [frames[name + ".csv"] for name in ["customers_served", "operation", "oil_tests", "costs_budget"]]
    late = frames["late_drop/oil_tests_late.csv"]
    edges = card["edge_cases"]
    ids = set(customers.transformer_id.str.strip().str.upper())
    assert len(ids) == 400 and len(customers) == 405
    assert list(customers.columns) == ASSET_COLUMNS
    assert list(op.columns) == OP_COLUMNS and list(oil.columns) == OIL_COLUMNS
    assert list(late.columns) == OIL_COLUMNS and list(costs.columns) == COST_COLUMNS
    for frame in [op, oil, late]:
        assert set(frame.transformer_id).issubset(ids)
    for frame in [oil, late]:
        assert np.isfinite(frame[GASES].to_numpy()).all() and (frame[GASES] >= 0.5).all().all()
        assert frame.test_date.max() <= TODAY
        assert not frame.duplicated(["transformer_id", "test_date"]).any()
    assert oil.test_date.between("2023-10-01", "2026-09-30").all()
    assert late.test_date.between("2026-10-01", "2026-10-10").all()
    assert op.month.isin(MONTHS.strftime("%Y-%m-%d")).all()
    assert customers.customer_type.isin(["hospital", "industry", "urban", "rural"]).all()
    for frame in [costs, frames["late_drop/costs_budget_late.csv"]]:
        assert len(frame) == 6 and (frame[["cost", "annual_budget"]] > 0).all().all()
        assert not frame.duplicated(["action", "rating_class"]).any()
    assert costs.annual_budget.eq(5000000).all()
    assert frames["late_drop/costs_budget_late.csv"].annual_budget.eq(4000000).all()
    assert card["n_untested"] == 152 and oil.transformer_id.nunique() == 248
    assert set(edges["E1_no_oil_test"]) == ids - set(oil.transformer_id)
    assert card["n_visible_developing_faults"] >= 25, "Need >=25 tested faults with rising post-onset observations"
    assert len(card["faulty_units"]) == 40
    assert 40 <= card["n_failures"] <= 60, f"Failure count: {card['n_failures']}"
    assert card["n_failures_months_0_23"] >= 25, f"Early failures: {card['n_failures_months_0_23']}"
    failures = op.loc[op.event == "failure"]
    assert len(failures) == card["n_failures"]
    assert int((failures.month < "2025-10-01").sum()) == card["n_failures_months_0_23"]
    exposure = card["failure_exposure"]
    assert exposure["active_fault_months"] + exposure["no_fault_months"] == 14400
    assert exposure["active_fault_failures"] + exposure["no_fault_failures"] == len(failures)
    rates = card["monthly_failure_rates"]
    assert rates["active_fault"] >= 2 * rates["no_fault"], "Fault signal must at least double monthly failure rate"
    assert edges["E6_budget_shortfall"]["total_replace_cost"] > 5000000
    indexed_assets = assets.set_index("transformer_id")
    for unit in edges["E2_old_few_tests"]:
        assert indexed_assets.at[unit, "age_at_start"] > 40
        assert (oil.transformer_id == unit).sum() == 1
    assert len(edges["E2_old_few_tests"]) == 6
    assert len(edges["E3_lookalike_pairs"]) == 3
    for first, second in edges["E3_lookalike_pairs"]:
        a, b = indexed_assets.loc[first], indexed_assets.loc[second]
        for column in ["age_at_start", "base_load", "load_trend", "rating_class", "fault", "fault_onset_month"]:
            assert a[column] == b[column]
        assert a.customer_type == "hospital" and b.customer_type == "rural"
        assert 8000 <= a.customers <= 15000 and 200 <= b.customers <= 600
        x = oil.loc[oil.transformer_id == first].reset_index(drop=True)
        y = oil.loc[oil.transformer_id == second].reset_index(drop=True)
        assert x.test_date.equals(y.test_date)
        assert np.allclose(x[GASES], y[GASES], rtol=0.3, atol=0.1)
    for jump in edges["E4_unit_jump"]:
        tests = oil.loc[oil.transformer_id == jump["transformer_id"]].sort_values("test_date")
        assert len(tests) >= 3 and tests.iloc[0].test_date != jump["test_date"]
        bad = tests.loc[tests.test_date == jump["test_date"], GASES].iloc[0]
        assert (bad / tests[GASES].median() >= 200).all(), "Unit jump must be detectable by B's rule"
    assert len(edges["E4_unit_jump"]) == 3
    markers = {"thermal": "c2h4_ppm", "partial_discharge": "h2_ppm", "arcing": "c2h2_ppm"}
    for repair in edges["E5_repair_reset"]:
        unit, date = repair["transformer_id"], repair["repair_month"]
        repair_rows = op.loc[(op.transformer_id == unit) & (op.month == date)]
        assert len(repair_rows) == 1 and repair_rows.iloc[0].event == "repair"
        tests = oil.loc[oil.transformer_id == unit]
        assert (tests.test_date < date).any() and (tests.test_date > date).any()
        marker = markers[indexed_assets.at[unit, "fault"]]
        before = tests.loc[tests.test_date < date, marker].iloc[-1]
        after = tests.loc[tests.test_date > date, marker].iloc[0]
        assert after < before / 2, "Planted repair must visibly reset its fault-related gas"
        i = int(assets.index[assets.transformer_id == unit][0])
        m = MONTHS.get_loc(pd.Timestamp(date))
        assert 31 <= m <= 33 and active[i, m - 1] and not active[i, m:].any()
    assert len(edges["E5_repair_reset"]) == 3
    special_groups = [set(edges["E2_old_few_tests"]),
                      {unit for pair in edges["E3_lookalike_pairs"] for unit in pair},
                      {row["transformer_id"] for row in edges["E4_unit_jump"]},
                      {row["transformer_id"] for row in edges["E5_repair_reset"]}]
    for i, group in enumerate(special_groups):
        for other in special_groups[i + 1:]:
            assert group.isdisjoint(other), "Planted test recipes must not collide"
    assert customers.duplicated().sum() == 3 and op.duplicated().sum() == 6
    assert len(op) == 14400 - 288 + 6
    for column in ["load_pct", "top_oil_temp_c"]:
        assert 0.009 <= op[column].isna().mean() <= 0.011
    assert op.load_pct.dropna().between(5, 150).all()
    assert op.top_oil_temp_c.dropna().between(15, 120).all()
    assert len(late) == 40 and late.transformer_id.nunique() == 40
    assert len(set(late.transformer_id) & set(oil.transformer_id)) == 30
    assert len(set(late.transformer_id) - set(oil.transformer_id)) == 10
    for unit in card["late_drop"]["first_test_faults"]:
        i = int(assets.index[assets.transformer_id == unit][0])
        assert active[i, 35] and unit not in set(oil.transformer_id)
        marker = markers[indexed_assets.at[unit, "fault"]]
        typical_baseline = {"c2h4_ppm": 15, "h2_ppm": 40, "c2h2_ppm": 0.5}[marker]
        value = late.loc[late.transformer_id == unit, marker].iloc[0]
        assert value > 2 * typical_baseline, "First-ever test must show a visible fault pattern"
    for name, count in {**card["row_counts"], **card["late_drop_row_counts"]}.items():
        assert count == len(frames[name])
    card["checks"] = {"source_schema_and_foreign_keys": "passed", "positive_gases_costs_budgets": "passed",
                      "date_bounds": "passed", "hcl_minimums": "passed", "edge_cases_E1_E8": "passed",
                      "failure_count_and_signal": "passed", "row_counts_in_memory": "passed"}


def data_card_markdown(card):
    lines = ["# Transformer fleet synthetic data card", "",
             "transformer_fleet_synthetic v1.0 | 2026-10-11 | seed 42", "",
             "## How it was made", "",
             "Run `python scripts/generate_data.py` from the repository root. The generator uses Python, numpy and pandas, "
             "one seeded random stream, and the formulas in `docs/DATA_SPEC.md`. This is team-generated development "
             "and rehearsal data, not HCL's hidden evaluation data.", "",
             "AI coding assistant: " + card["provenance"]["assistant"] + ". The exact user prompt was:", "",
             "```text", USER_PROMPT, "```", "",
             "The referenced Member A brief directs the assistant to follow DATA_SPEC Part 1 step by step. "
             "No training metrics or held-out outcomes were consulted when constructing this dataset.", "",
             "Environment: Python " + card["provenance"]["python"] + "; numpy " + np.__version__ + "; pandas " + pd.__version__ + ".", "",
             "## Contents", "", "| File | Rows | Dates |", "|---|---:|---|"]
    dates = {"customers_served.csv": "Asset master", "costs_budget.csv": "Reference",
             "late_drop/costs_budget_late.csv": "Reference"}
    for name, bounds in card["date_ranges"].items():
        dates[name] = bounds[0] + " to " + bounds[1]
    for name, count in {**card["row_counts"], **card["late_drop_row_counts"]}.items():
        lines.append(f"| {name} | {count:,} | {dates[name]} |")
    rates = card["monthly_failure_rates"]
    lines += ["", "## Simulation and assumptions", "",
              f"There are 400 units; 248 tested and 152 untested (38%). There are {card['n_failures']} failure events, "
              f"including {card['n_failures_months_0_23']} in months 0–23. Exactly 40 units have a planted developing fault; "
              f"{card['n_visible_developing_faults']} have at least two active post-onset tests with a rising marker gas.", "",
              f"Starting base hazard 0.0012 produced {card['initial_failure_count']} failures. One deterministic "
              f"rescale by 50 / {card['initial_failure_count']} gives {card['base_hazard']:.10f}. The same monthly random draws "
              "are replayed. A failure clears an existing fault immediately; a future scheduled onset is still allowed. "
              "Age continues increasing after a repair.", "",
              f"Active-fault monthly failure rate: {rates['active_fault']:.6f}; no-active-fault rate: {rates['no_fault']:.6f}. "
              "These rates use failures divided by exposed transformer-months, before telemetry gaps. 'Clearly higher' "
              "is checked as at least 2 times. These are synthetic rates, not measured utility statistics.", "",
              "Rural/small units are sampled with lower test probability, then adjusted to 248 tested. Fault onsets are "
              "drawn uniformly in months 0–30; 32 tested faulty units are selected from those with two scheduled tests "
              "after onset, and 8 faults are untested. This visibility constraint intentionally biases the selected onsets "
              "toward earlier months. Six old units have only one test. Look-alike pairs copy hidden condition, weather, "
              "event draws and test dates; gas measurements differ by small noise. Customer and size proportions are "
              "initial quotas, with small changes from the planted pairs.", "",
              "Planned repairs are selected from still-active surviving faults. Additional tests bracket each repair. "
              "Failure and repair events occur on the first of the month, so all subsequent tests see baseline gases. "
              "Random gaps affect only ordinary monthly telemetry, preserving every failure and repair label. "
              "October tests continue September's state; no October failure outcomes are generated.", "",
              "Replacement costs 150,000 / 350,000 / 700,000 and inspection costs 10,000 / 15,000 / 25,000 "
              "for small / medium / large units are team assumptions authorised at the design review. Annual budget "
              "is 5,000,000 and rehearsal budget 4,000,000. Costs are not externally validated estimates.", "",
              "## Traps and edge cases", "",
              "Full row-level coordinates, including all missing months/cells, are in `data_card.json`. "
              "Its `faulty_units` and other hidden state are for validation only: the model must never read them.", ""]
    descriptions = {
        "E1_no_oil_test": "E1: no tests; exercise low-confidence scoring",
        "E2_old_few_tests": "E2: old units with exactly one test",
        "E3_lookalike_pairs": "E3: similar condition, hospital versus rural impact",
        "E4_unit_jump": "E4: all six gases recorded as ppb under ppm headers; divide the flagged rows by 1000 in silver",
        "E5_repair_reset": "E5: planned repair clears the gas pattern",
        "E6_budget_shortfall": "E6: replacing all faulty units exceeds the budget",
        "E7_duplicate_assets": "E7: three exact duplicates plus lowercase TR-0045 and whitespace TR-0046",
        "E7_duplicate_operation": "E7: three repeated ordinary month rows for each of two units",
    }
    for key, description in descriptions.items():
        lines += [description + ":", "", "```json", json.dumps(card["edge_cases"][key], indent=2), "```", ""]
    lines += ["E8: 288 ordinary monthly rows removed; about 1% each of load and temperature cells left empty. "
              "Exact IDs and months are recorded under `E8_missing_*` in the JSON card.", "",
              "## Late-drop rehearsal", "",
              "The oil file adds 30 follow-up tests and 10 first-ever tests, including at least two active faults. "
              "Use `as_of=2026-10-11` when scoring after ingest: the contract default 2026-10-01 excludes October tests "
              "under the strict-before feature rule. The budget file reduces all six annual_budget values by 20%.", "",
              "## Checks", ""]
    for check, result in card["checks"].items():
        lines.append(f"- {check}: {result}")
    lines += ["", "The generator asserts schemas, references, date bounds, positive gas/cost values, all HCL minimums, "
              "E1–E8 recipes, failure totals and exposure-normalised signal. It serializes two independently generated "
              "datasets and compares every output byte. Actual CSV row counts are re-read and compared with the card. "
              "Feature leakage and train/test split checks belong to B/C, once their code exists.", "",
              "## Known limits and privacy", "",
              "The hazard is a simplified formula and gas patterns implement the supplied synthetic recipe, not an "
              "independently validated physical model or certified IEEE interpretation. There is seasonal ambient temperature "
              "but no observed weather, repair costs, network topology or independent field validation. Neglected units "
              "deliberately receive a 1.4x hazard multiplier; this planted association is not real-world evidence. "
              "Look-alike pairs are correlated; C should consider keeping each pair in the same validation group. "
              "Synthetic performance cannot establish performance on HCL or operational data.", "",
              "All transformer IDs, customer counts and regions are synthetic. No real company or personal data is used.", ""]
    return "\n".join(lines)


def write_dataset(directory, frames, card):
    """Use fixed ordering, newline and decimal formatting for portable CSV bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        exported = frame.copy()
        if name == "operation.csv":
            exported["age_years"] = exported.age_years.map(lambda value: f"{value:.2f}")
        exported.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.1f", na_rep="")
    (directory / "data_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    (directory / "DATA_CARD.md").write_text(data_card_markdown(card), encoding="utf-8")


def main():
    print("Generating fleet, operation, oil tests and planted edge cases (seed 42)...")
    frames, card = build_dataset()
    print("Replaying generation and checking byte-for-byte reproducibility...")
    again, again_card = build_dataset()
    with tempfile.TemporaryDirectory(prefix="rayquaza-data-") as tmp:
        first, second = Path(tmp) / "first", Path(tmp) / "second"
        write_dataset(first, frames, card)
        write_dataset(second, again, again_card)
        for path in first.rglob("*"):
            if path.is_file():
                assert path.read_bytes() == (second / path.relative_to(first)).read_bytes(), path.name
        counts = {**card["row_counts"], **card["late_drop_row_counts"]}
        for name, count in counts.items():
            assert len(pd.read_csv(first / name)) == count
    card["checks"]["byte_identical_regeneration"] = "passed (six CSVs and both data cards)"
    card["checks"]["row_counts_from_disk"] = "passed (six CSVs)"
    write_dataset(DATA_DIR, frames, card)
    saved_card = json.loads((DATA_DIR / "data_card.json").read_text())
    for name, count in {**saved_card["row_counts"], **saved_card["late_drop_row_counts"]}.items():
        assert len(pd.read_csv(DATA_DIR / name)) == count
        print(f"{name:36s}: {count:,} rows")
    print(f"oil coverage: 248 tested, 152 untested (38%); visible faults: {card['n_visible_developing_faults']}")
    print(f"failures: {card['n_failures']} total; {card['n_failures_months_0_23']} in months 0–23")
    rates = card["monthly_failure_rates"]
    print(f"fault vs no-fault monthly failure rate: {rates['active_fault']:.6f} vs {rates['no_fault']:.6f}")
    print("edge cases planted: E2=6 E3=3 pairs E4=3 E5=3 E7=5 assets+6 operation rows; E1/E6/E8 checked")
    print("operation.csv MD5:", hashlib.md5((DATA_DIR / "operation.csv").read_bytes()).hexdigest())
    print("All checks passed. Files written to", DATA_DIR)


if __name__ == "__main__":
    main()
