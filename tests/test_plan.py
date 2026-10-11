"""Tests for src/plan.py (Member C). Plain asserts on tiny made-up tables: they check the CODE, not the model.
Run with:  pytest tests/test_plan.py        or        python tests/run_tests.py test_plan
"""
import numpy as np
import pandas as pd

from src.plan import (COST_UNIT, TYPE_WEIGHTS, age_plan, can_replace, compute_risk, inspect_value, make_plan,
                      replace_value)

COSTS = pd.DataFrame({"action": ["replace"] * 3 + ["inspect"] * 3, "rating_class": ["small", "medium", "large"] * 2,
                      "cost": [150000, 350000, 700000, 10000, 15000, 25000]})


def _risk(rows):
    """rows: dicts with id, p, lo, hi, impact and optional tested/months/age/cls. Returns a risk table."""
    out = []
    for r in rows:
        out.append({"transformer_id": r["id"], "failure_prob_12m": r["p"], "prob_low": r.get("lo", r["p"]),
                    "prob_high": r.get("hi", r["p"]), "impact_weight": r["impact"], "risk_score": r["p"] * r["impact"],
                    "has_oil_test": r.get("tested", 1), "months_since_last_test": r.get("months", 3.0 if r.get("tested", 1) else np.nan),
                    "age_years": r.get("age", 30.0), "rating_class": r.get("cls", "small")})
    return pd.DataFrame(out)


def _random_fleet(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.005, 0.6, n)
    hi = np.minimum(p + rng.uniform(0, 0.3, n), 1.0)
    tested = rng.random(n) < 0.7
    return pd.DataFrame({"transformer_id": [f"T{i:03d}" for i in range(n)], "failure_prob_12m": p, "prob_low": p * 0.7, "prob_high": hi,
                         "impact_weight": rng.choice([1, 2, 3, 5], n) * rng.integers(100, 5000, n).astype(float),
                         "has_oil_test": tested.astype(int), "months_since_last_test": np.where(tested, rng.uniform(1, 30, n), np.nan),
                         "age_years": rng.uniform(2, 50, n), "rating_class": rng.choice(["small", "medium", "large"], n)}
                        ).assign(risk_score=lambda d: d["failure_prob_12m"] * d["impact_weight"])


# ---- compute_risk -------------------------------------------------------------------------------
def test_impact_is_type_weight_times_customers_and_risk_is_probability_times_impact():
    scores = pd.DataFrame({"transformer_id": ["A", "B", "C", "D"], "failure_prob_12m": [0.2, 0.2, 0.2, 0.2],
                           "prob_low": [0.1] * 4, "prob_high": [0.3] * 4})
    dim = pd.DataFrame({"transformer_id": ["A", "B", "C", "D"], "customer_type": ["hospital", "industry", "urban", "rural"],
                        "customers": [10, 10, 10, 10], "rating_class": ["small"] * 4})
    r = compute_risk(scores, dim).set_index("transformer_id")
    assert list(r["impact_weight"]) == [50.0, 30.0, 20.0, 10.0]
    assert np.allclose(r["risk_score"], [10.0, 6.0, 4.0, 2.0])
    assert np.allclose(r["risk_low"], [5.0, 3.0, 2.0, 1.0]) and np.allclose(r["risk_high"], [15.0, 9.0, 6.0, 3.0])
    assert TYPE_WEIGHTS == {"hospital": 5, "industry": 3, "urban": 2, "rural": 1}


def test_annex_a_worked_example_hospital_outranks_a_likelier_rural_unit():
    """19% at a hospital (weight 5) beats 25% in a rural area (weight 1), with equal customers."""
    scores = pd.DataFrame({"transformer_id": ["TR-0102", "TR-0117"], "failure_prob_12m": [0.19, 0.25],
                           "prob_low": [0.1, 0.2], "prob_high": [0.3, 0.3]})
    dim = pd.DataFrame({"transformer_id": scores["transformer_id"], "customer_type": ["hospital", "rural"],
                        "customers": [1, 1], "rating_class": ["small", "small"]})
    r = compute_risk(scores, dim).set_index("transformer_id")
    assert r.loc["TR-0102", "risk_score"] > r.loc["TR-0117", "risk_score"]


def test_unknown_transformer_gets_nan_risk_not_zero():
    scores = pd.DataFrame({"transformer_id": ["A", "Z"], "failure_prob_12m": [0.1, 0.1], "prob_low": [0.05] * 2, "prob_high": [0.2] * 2})
    dim = pd.DataFrame({"transformer_id": ["A"], "customer_type": ["urban"], "customers": [5], "rating_class": ["small"]})
    r = compute_risk(scores, dim).set_index("transformer_id")
    assert np.isnan(r.loc["Z", "risk_score"]) and r.loc["A", "risk_score"] == 0.1 * 10


# ---- action values and rules -------------------------------------------------------------------
def test_replace_and_inspect_values_follow_the_contract_formulas():
    row = _risk([{"id": "A", "p": 0.19, "lo": 0.12, "hi": 0.28, "impact": 1000}]).iloc[0]
    assert abs(replace_value(row) - (0.19 - 0.01) * 1000) < 1e-9
    assert abs(inspect_value(row) - (0.28 - 0.19) * 1000) < 1e-9


def test_inspection_is_worthless_when_the_range_is_narrow():
    """We already know the risk (p_high == p), so an inspection reveals nothing: it is never chosen."""
    risk = _risk([{"id": "A", "p": 0.2, "lo": 0.2, "hi": 0.2, "impact": 1000, "tested": 0}])
    assert inspect_value(risk.iloc[0]) == 0
    assert make_plan(risk, COSTS, 1_000_000)["actions"] == []


def test_a_unit_with_no_oil_test_is_never_replaced():
    """No usable test: replacing blind is not allowed, even with an unlimited budget. It may be inspected."""
    risk = _risk([{"id": "NOTEST", "p": 0.4, "lo": 0.05, "hi": 0.9, "impact": 100000, "tested": 0}])
    plan = make_plan(risk, COSTS, 10_000_000)
    assert [a["action"] for a in plan["actions"]] == ["inspect"]


def test_a_stale_test_over_18_months_cannot_be_replaced_but_a_fresh_one_can():
    stale = _risk([{"id": "S", "p": 0.4, "hi": 0.4, "impact": 100000, "months": 19.0}])
    fresh = _risk([{"id": "F", "p": 0.4, "hi": 0.4, "impact": 100000, "months": 18.0}])
    assert not can_replace(stale.iloc[0]) and can_replace(fresh.iloc[0])
    assert make_plan(stale, COSTS, 10_000_000)["actions"] == []                   # replace banned, inspect worth 0
    assert make_plan(fresh, COSTS, 10_000_000)["actions"][0]["action"] == "replace"


# ---- budget rules ----------------------------------------------------------------------------------
def test_total_cost_never_exceeds_budget_and_one_action_per_transformer():
    """PROPERTY over 200 random fleets and budgets (seeded)."""
    rng = np.random.default_rng(42)
    for trial in range(200):
        risk = _random_fleet(int(rng.integers(3, 60)), trial)
        budget = float(rng.choice([0, 4999, 5000, 40000, 300000, 1_500_000, 10_000_000]))
        plan = make_plan(risk, COSTS, budget)
        ids = [a["transformer_id"] for a in plan["actions"]]
        assert plan["total_cost"] <= budget + 1e-6, (trial, plan["total_cost"], budget)
        assert len(ids) == len(set(ids)), trial
        assert all(a["cost"] > 0 and a["value"] > 0 for a in plan["actions"])
        assert plan["risk_reduced_pct"] <= 100 + 1e-9                                 # can never remove more than exists
        for a in plan["actions"]:                                                  # the no-blind-replacement rule, everywhere
            if a["action"] == "replace":
                row = risk[risk["transformer_id"] == a["transformer_id"]].iloc[0]
                assert can_replace(row), (trial, a)


def test_with_a_huge_budget_every_unit_gets_its_best_positive_action():
    risk = _random_fleet(40, 7)
    plan = make_plan(risk, COSTS, 1e12)
    chosen = {a["transformer_id"]: a for a in plan["actions"]}
    for _, row in risk.iterrows():
        values = {"inspect": inspect_value(row)}
        if can_replace(row):
            values["replace"] = replace_value(row)
        best_action = max(values, key=values.get)
        if values[best_action] > 0:
            assert chosen[row["transformer_id"]]["action"] == best_action
            assert abs(chosen[row["transformer_id"]]["value"] - values[best_action]) < 1e-9
        else:
            assert row["transformer_id"] not in chosen


def test_knapsack_removes_at_least_as_much_risk_as_the_age_plan():
    """On units that may all be replaced, the exact plan can never do worse than 'oldest first' on what it maximises
    (the age plan is one of the plans the knapsack could have picked), and the replace-only optimum never removes less
    actual risk than the age plan."""
    for trial in range(60):
        risk = _random_fleet(30, 100 + trial)
        risk["has_oil_test"], risk["months_since_last_test"] = 1, 4.0
        budget = 1_000_000.0
        age = age_plan(risk, COSTS, budget)
        assert make_plan(risk, COSTS, budget)["value"] >= age["value"] - 1e-9, trial
        assert make_plan(risk, COSTS, budget, allow_inspect=False)["risk_removed"] >= age["risk_removed"] - 1e-9, trial


def test_budget_edges_zero_tiny_and_exact():
    risk = _risk([{"id": "A", "p": 0.3, "hi": 0.4, "impact": 1000, "cls": "small"}])
    assert make_plan(risk, COSTS, 0)["actions"] == []
    assert make_plan(risk, COSTS, 4999)["actions"] == []                           # cannot afford even an inspection (10,000)
    exact = make_plan(risk, COSTS, 150000)                                         # exactly the replace price
    assert [a["action"] for a in exact["actions"]] == ["replace"] and exact["total_cost"] == 150000
    just_short = make_plan(risk, COSTS, 149999)                                    # replace no longer fits: inspect instead
    assert [a["action"] for a in just_short["actions"]] == ["inspect"]


def test_ties_give_the_same_plan_every_time():
    risk = _random_fleet(25, 3)
    risk[["failure_prob_12m", "prob_high", "impact_weight"]] = [0.2, 0.3, 1000.0]
    risk["risk_score"] = 200.0
    first = [(a["transformer_id"], a["action"]) for a in make_plan(risk, COSTS, 400000)["actions"]]
    for _ in range(4):
        assert [(a["transformer_id"], a["action"]) for a in make_plan(risk.sample(frac=1, random_state=1), COSTS, 400000)["actions"]] == first


def test_bad_budgets_and_missing_columns_are_refused_clearly():
    risk = _random_fleet(5, 1)
    for bad in (-1, float("nan"), None, True):
        try:
            make_plan(risk, COSTS, bad)
        except ValueError:
            continue
        raise AssertionError(f"budget {bad!r} should be refused")
    try:
        make_plan(risk.drop(columns=["has_oil_test"]), COSTS, 1000)
    except ValueError as e:
        assert "has_oil_test" in str(e)
    else:
        raise AssertionError("expected a clear error for a missing column")


def test_missing_rating_class_uses_medium_pricing():
    risk = _risk([{"id": "A", "p": 0.3, "hi": 0.3, "impact": 100000, "cls": None}])
    plan = make_plan(risk, COSTS, 1_000_000)
    assert plan["actions"][0]["cost"] == 350000


# ---- output shape and the age plan ----------------------------------------------------------------
def test_plan_has_the_contract_shape_and_a_rationale_for_every_action():
    plan = make_plan(_random_fleet(30, 5), COSTS, 800000)
    assert {"actions", "total_cost", "budget", "risk_before", "risk_removed", "risk_reduced_pct"} <= set(plan)
    assert plan["actions"] and 0 <= plan["risk_reduced_pct"] <= 100
    for a in plan["actions"]:
        assert {"transformer_id", "action", "cost", "risk_before", "risk_removed", "risk_revealed", "value", "rationale"} <= set(a)
        assert a["action"] in ("inspect", "replace") and len(a["rationale"]) > 20 and a["requires_approval"] is True


def test_age_plan_replaces_oldest_first_and_respects_the_budget():
    risk = _risk([{"id": "OLD", "p": 0.1, "impact": 100, "age": 50.0}, {"id": "MID", "p": 0.1, "impact": 100, "age": 30.0},
                  {"id": "NEW", "p": 0.1, "impact": 100, "age": 5.0}])
    plan = age_plan(risk, COSTS, 300000)                                           # two small replacements fit (150,000 each)
    assert [a["transformer_id"] for a in plan["actions"]] == ["OLD", "MID"] and plan["total_cost"] <= 300000
    assert all(a["action"] == "replace" for a in plan["actions"])


def test_cost_unit_is_5000_and_costs_are_rounded_up_never_down():
    """A cost that is not a multiple of 5,000 is rounded up in the knapsack, so the budget can never be overspent."""
    odd = COSTS.copy()
    odd.loc[odd["action"] == "inspect", "cost"] = 12001.0
    risk = _risk([{"id": "A", "p": 0.2, "hi": 0.5, "impact": 100000, "tested": 0}])
    assert make_plan(risk, odd, 14999)["actions"] == [] or make_plan(risk, odd, 14999)["total_cost"] <= 14999
    assert make_plan(risk, odd, 15000)["total_cost"] <= 15000 and COST_UNIT == 5000


# ---- honest reporting: removed vs revealed ------------------------------------------------------------------
def test_inspections_reveal_risk_but_remove_none_and_replacements_never_remove_more_than_the_unit_has():
    """Regression: inspections used to count as removed risk, so the real-data plan claimed 108% of the fleet's risk."""
    risk = _random_fleet(80, 11)
    plan = make_plan(risk, COSTS, 2_000_000)
    assert plan["n_inspect"] > 0 and plan["n_replace"] > 0
    for a in plan["actions"]:
        if a["action"] == "inspect":
            assert a["risk_removed"] == 0 and a["risk_revealed"] > 0
        else:
            assert a["risk_revealed"] == 0 and 0 < a["risk_removed"] <= a["risk_before"]
    assert abs(plan["value"] - (plan["risk_removed"] + plan["risk_revealed"])) < 1e-6
    assert 0 <= plan["risk_reduced_pct"] <= 100


def test_replace_only_plan_has_no_inspections():
    plan = make_plan(_random_fleet(40, 12), COSTS, 1_000_000, allow_inspect=False)
    assert plan["n_inspect"] == 0 and all(a["action"] == "replace" for a in plan["actions"])
