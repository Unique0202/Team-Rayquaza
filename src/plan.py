"""Risk and budget plan (Member C). Follows docs/team/SHARED_CONTRACT.md sections 7 and 8.

No machine learning here: every function takes numbers in and gives numbers out, so each one can be
tested on a tiny made-up table. Three ideas, kept apart on purpose:

  1. compute_risk : risk = failure probability x impact weight          (what could go wrong, and how bad)
  2. make_plan    : pick none / inspect / replace per transformer       (what to do about it, within a budget)
  3. age_plan     : the simple rule we must beat (replace the oldest first)

What an action is worth (contract section 8, from the mentor's answer that an inspection reveals how severe a
transformer really is):
  replace value = (p - 0.01) x impact      a brand-new unit is assumed to have a 1% yearly failure chance
  inspect value = (p_high - p) x impact    the hidden risk an inspection could reveal; a wide range means we
                                           know little (inspection is valuable), a narrow one means ~0
  A transformer with no usable oil test, or a last test older than 18 months, cannot be replaced blind:
  it can only be inspected.
"""

import numpy as np
import pandas as pd

TYPE_WEIGHTS = {"hospital": 5, "industry": 3, "urban": 2, "rural": 1}
NEW_UNIT_RISK = 0.01              # assumed yearly failure chance of a freshly replaced unit
STALE_MONTHS = 18                 # a last test older than this does not count as "usable" for replacing
COST_UNIT = 5000                  # all costs are multiples of 5,000, so the knapsack works in these steps
DEFAULT_RATING_CLASS = "medium"   # used when a transformer has no rating class (matches B's silver rule)

# columns that must be present in the table handed to make_plan / age_plan
NEEDS_FOR_PLAN = ["transformer_id", "failure_prob_12m", "prob_low", "prob_high", "impact_weight", "risk_score",
                  "has_oil_test", "months_since_last_test"]


def compute_risk(scores: pd.DataFrame, dim_transformer: pd.DataFrame) -> pd.DataFrame:
    """Add impact and risk columns to the model's scores.

    scores          : one row per transformer: transformer_id, failure_prob_12m, prob_low, prob_high, plus
                      has_oil_test, months_since_last_test, age_years (score() passes these through)
    dim_transformer : transformer_id, customer_type, customers, rating_class
    Adds: customer_type, customers, rating_class, impact_weight, risk_score, risk_low, risk_high.
    A transformer missing from dim_transformer gets NaN impact and risk (it cannot be ranked), not zero."""
    dim = dim_transformer[["transformer_id", "customer_type", "customers", "rating_class"]].drop_duplicates("transformer_id")
    out = scores.merge(dim, on="transformer_id", how="left", validate="many_to_one")
    weight = out["customer_type"].map(TYPE_WEIGHTS).astype(float)                # hospital 5 ... rural 1
    out["impact_weight"] = weight * out["customers"].astype(float)               # type weight x customers served
    out["risk_score"] = out["failure_prob_12m"] * out["impact_weight"]           # risk = probability x impact
    out["risk_low"] = out["prob_low"] * out["impact_weight"]
    out["risk_high"] = out["prob_high"] * out["impact_weight"]
    return out


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _cost_lookup(costs: pd.DataFrame) -> dict:
    """{action: {rating_class: cost}} from the costs table (columns action, rating_class, cost)."""
    lookup = {}
    for action, grp in costs.groupby("action"):
        lookup[action] = {r.rating_class: float(r.cost) for r in grp.itertuples()}
    return lookup


def _cost_of(lookup: dict, action: str, rating_class) -> float:
    """Cost of an action for a rating class. Unknown or missing class -> 'medium'; if even that is absent, the
    median cost of that action. NaN means the action has no usable price and will be skipped."""
    prices = lookup.get(action, {})
    if not prices:
        return float("nan")
    if rating_class in prices:
        return prices[rating_class]
    if DEFAULT_RATING_CLASS in prices:
        return prices[DEFAULT_RATING_CLASS]
    return float(np.median(list(prices.values())))


def _check_columns(risk: pd.DataFrame, needed: list) -> None:
    missing = [c for c in needed if c not in risk.columns]
    if missing:
        raise ValueError(f"the risk table is missing column(s) {missing}; "
                         "run it through compute_risk() and make sure score() passes them through")


def _check_budget(budget) -> float:
    if isinstance(budget, bool) or budget is None or not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a non-negative number")
    return float(budget)


def can_replace(row) -> bool:
    """Replacing needs a usable oil test no older than 18 months (contract section 8)."""
    return bool(row["has_oil_test"] == 1 and pd.notna(row["months_since_last_test"])
                and row["months_since_last_test"] <= STALE_MONTHS)


def replace_value(row) -> float:
    """Risk removed by replacing: (p - 0.01) x impact, never negative."""
    return max(float(row["failure_prob_12m"]) - NEW_UNIT_RISK, 0.0) * float(row["impact_weight"])


def inspect_value(row) -> float:
    """Risk an inspection could reveal: (p_high - p) x impact, never negative."""
    return max(float(row["prob_high"]) - float(row["failure_prob_12m"]), 0.0) * float(row["impact_weight"])


def _rationale(row, action: str, cost: float, removed: float) -> str:
    """One plain-English line built from the numbers behind the choice."""
    p, lo, hi = row["failure_prob_12m"], row["prob_low"], row["prob_high"]
    impact = row["impact_weight"]
    if action == "replace":
        return (f"replace: failure chance {p:.1%} (90% range {lo:.1%} to {hi:.1%}) x impact {impact:,.0f}; "
                f"a new unit is assumed {NEW_UNIT_RISK:.0%}, so this removes about {removed:,.0f} risk for {cost:,.0f}")
    reason = "no usable oil test" if not can_replace(row) else "range is wide"
    return (f"inspect: {reason}; the range {lo:.1%} to {hi:.1%} is {hi - p:.1%} above the estimate {p:.1%}, and an "
            f"inspection could reveal up to {removed:,.0f} risk (impact {impact:,.0f}) for {cost:,.0f}")


def _result(actions: list, risk: pd.DataFrame, budget: float) -> dict:
    """Package a list of chosen actions in the shape the contract asks for, and re-check the two hard rules."""
    total_cost = float(sum(a["cost"] for a in actions))
    ids = [a["transformer_id"] for a in actions]
    assert total_cost <= budget + 1e-6, "plan exceeds budget"
    assert len(ids) == len(set(ids)), "a transformer got two actions"
    risk_before = float(risk["risk_score"].sum(skipna=True))
    removed = float(sum(a["risk_removed"] for a in actions))
    return {"actions": actions, "total_cost": total_cost, "budget": budget, "risk_before": risk_before,
            "risk_removed": removed, "risk_reduced_pct": 100 * removed / risk_before if risk_before > 0 else 0.0}


# ----------------------------------------------------------------------------
# the plan
# ----------------------------------------------------------------------------
def make_plan(risk: pd.DataFrame, costs: pd.DataFrame, budget: float) -> dict:
    """Choose at most one action per transformer to remove the most risk for the money.

    For every transformer the options are none / inspect / replace (replace only if allowed). This is a
    multiple-choice 0/1 knapsack, solved exactly by dynamic programming: best[b] is the most risk that can be
    removed with b cost units (5,000 each) spent so far. Costs are rounded UP to whole units, so the plan can
    never overspend."""
    budget = _check_budget(budget)
    _check_columns(risk, NEEDS_FOR_PLAN)
    risk = risk[risk["risk_score"].notna()].sort_values("transformer_id").reset_index(drop=True)   # fixed order = repeatable ties
    lookup = _cost_lookup(costs)
    cap = int(budget // COST_UNIT)                                   # whole cost units that fit in the budget

    # options[i] = list of (cost_in_units, value, action, cost) for transformer i; only actions worth something
    options = []
    for _, row in risk.iterrows():
        opts = []
        if can_replace(row) and replace_value(row) > 0:
            c = _cost_of(lookup, "replace", row.get("rating_class"))
            if pd.notna(c) and c > 0:
                opts.append((int(np.ceil(c / COST_UNIT)), replace_value(row), "replace", c))
        if inspect_value(row) > 0:
            c = _cost_of(lookup, "inspect", row.get("rating_class"))
            if pd.notna(c) and c > 0:
                opts.append((int(np.ceil(c / COST_UNIT)), inspect_value(row), "inspect", c))
        options.append(opts)

    # A huge budget must not build a huge table: nobody can spend more than every unit's priciest option, so
    # the table never needs more columns than that (this also keeps /plan safe against a silly budget like 1e12).
    cap = min(cap, sum(max((o[0] for o in opts), default=0) for opts in options))

    best = np.zeros(cap + 1)                                         # best[b]: most risk removed using at most b units
    took = np.full((len(risk), cap + 1), -1, dtype=np.int16)         # which option transformer i took at budget b
    for i, opts in enumerate(options):
        new = best.copy()
        for k, (units, value, _, _) in enumerate(opts):
            if units > cap:
                continue
            candidate = np.full(cap + 1, -np.inf)
            candidate[units:] = best[: cap + 1 - units] + value      # spend `units` on this option, rest on earlier units
            better = candidate > new + 1e-12                         # strictly better only, so ties keep the cheaper choice
            new = np.where(better, candidate, new)
            took[i, better] = k
        best = new

    # walk backwards to read off which option each transformer took
    actions, b = [], int(np.argmax(best))
    for i in range(len(risk) - 1, -1, -1):
        k = took[i, b]
        if k >= 0:
            units, value, action, cost = options[i][k]
            row = risk.iloc[i]
            actions.append({"transformer_id": row["transformer_id"], "action": action, "cost": cost,
                            "risk_before": float(row["risk_score"]), "risk_removed": float(value),
                            "rationale": _rationale(row, action, cost, value), "requires_approval": True})
            b -= units
    actions.reverse()
    return _result(actions, risk, budget)


def age_plan(risk: pd.DataFrame, costs: pd.DataFrame, budget: float) -> dict:
    """The rule we must beat: replace the oldest transformers first until the money runs out.
    Risk removed is counted with the same replace-value formula so the comparison is fair."""
    budget = _check_budget(budget)
    _check_columns(risk, NEEDS_FOR_PLAN + ["age_years"])
    risk = risk[risk["risk_score"].notna()].reset_index(drop=True)
    lookup = _cost_lookup(costs)
    actions, spent = [], 0.0
    for _, row in risk.sort_values(["age_years", "transformer_id"], ascending=[False, True]).iterrows():
        c = _cost_of(lookup, "replace", row.get("rating_class"))
        if pd.isna(c) or c <= 0 or spent + c > budget:
            continue                                                 # cannot afford this one, try the next oldest
        spent += c
        value = replace_value(row)
        actions.append({"transformer_id": row["transformer_id"], "action": "replace", "cost": c,
                        "risk_before": float(row["risk_score"]), "risk_removed": float(value),
                        "rationale": f"replace: age {row['age_years']:.0f} years (oldest first)", "requires_approval": True})
    return _result(actions, risk, budget)
