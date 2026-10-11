"""
Transformer Fleet Health - the asset-planning engineer's interface (Member B).

    streamlit run ui/app.py

Five screens, following the engineer's job: where is the risk -> why -> what do we do within the budget
-> what changes when new inspection results arrive -> can we trust the model and the data.
It calls the SAME Python functions as the API (build_features -> score -> compute_risk -> make_plan),
so the screen and the API can never disagree.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # Streamlit runs this file directly, so make `src` importable

from src import pipeline as P
from src.features import build_features
from src.loader import load_csv
from src.model import health_explanation, health_table, load_model, model_info, score
from src.plan import age_plan, compute_risk, make_plan

TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")
GASES = ["h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm", "co_ppm"]

st.set_page_config(page_title="Transformer Fleet Health", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------
# DATA - read the warehouse, score the fleet (cached until the warehouse is rebuilt)
# ----------------------------------------------------------------------------
def warehouse_version() -> float:
    """Changes every time the pipeline rebuilds the warehouse; part of the cache key below."""
    return P.DB_PATH.stat().st_mtime


def read_table(sql: str) -> pd.DataFrame:
    with P.get_connection() as conn:
        return pd.read_sql(sql, conn)


@st.cache_data(show_spinner="Scoring the fleet...")
def scored_fleet(version: float, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every transformer scored as of a date: probability, range, risk, health points. Returns (risk, features)."""
    with P.get_connection() as conn:
        features = build_features(conn, [as_of])
        dim = pd.read_sql("SELECT * FROM dim_transformer", conn)
    risk = compute_risk(score(features), dim)
    health = health_table(features).assign(transformer_id=features["transformer_id"])
    risk = (risk.drop(columns=["health_score"])
                .merge(health, on="transformer_id", how="left")
                .merge(dim[["transformer_id", "region"]], on="transformer_id", how="left"))
    return risk.sort_values("risk_score", ascending=False, na_position="last").reset_index(drop=True), features


def save_score_run(risk: pd.DataFrame, as_of: str, note: str) -> str:
    """Store one scoring run (dim_run + fact_risk) so later runs can be compared with it."""
    run_id = f"score-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
    cols = ["transformer_id", "failure_prob_12m", "prob_low", "prob_high", "status", "health_score",
            "impact_weight", "risk_score", "risk_low", "risk_high", "model_version"]
    rows = risk[cols].assign(run_id=run_id)
    run = pd.DataFrame([{"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                         "kind": f"score: {note}", "as_of": as_of, "budget": None,
                         "model_version": risk["model_version"].iloc[0]}])
    with P.engine.begin() as conn:               # both tables or neither
        run.to_sql("dim_run", conn, if_exists="append", index=False)
        rows.to_sql("fact_risk", conn, if_exists="append", index=False)
    return run_id


def pct(x) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("⚡ Transformer Fleet Health")
page = st.sidebar.radio("Screen", ["1 · Fleet risk ranking", "2 · Transformer detail", "3 · Budget plan",
                                   "4 · New data & what changed", "5 · Model & data quality"])
as_of = st.sidebar.date_input("Score as of", value=pd.Timestamp(TODAY), max_value=pd.Timestamp(TODAY)).strftime("%Y-%m-%d")
st.sidebar.caption("Only data dated before this day is used: the model never sees the future.")

if not P.DB_PATH.exists():
    st.error("No warehouse yet. Run `python -m src.pipeline` first.")
    st.stop()

model_ready = True
try:
    load_model()
except FileNotFoundError:
    model_ready = False

if not model_ready and not page.startswith("5"):
    st.warning("The model is not trained yet. Run `python -m src.model`, then refresh this page.")
    st.stop()

if model_ready:
    risk, features = scored_fleet(warehouse_version(), as_of)
    st.sidebar.caption(f"Model `{risk['model_version'].iloc[0]}` · {len(risk)} transformers")


# ----------------------------------------------------------------------------
# 1. FLEET RISK RANKING - where is the risk?
# ----------------------------------------------------------------------------
if page.startswith("1"):
    st.title("Fleet risk ranking")
    st.caption("Risk = chance of failing in the next 12 months × impact (hospital 5, industry 3, urban 2, rural 1, × customers served).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transformers", len(risk))
    c2.metric("Never oil-tested", f"{(risk['has_oil_test'] != 1).mean():.0%}")
    c3.metric("Low-confidence scores", int((risk["status"] != "scored").sum()))
    top10 = risk["risk_score"].head(len(risk) // 10).sum() / risk["risk_score"].sum()
    c4.metric("Risk held by the top 10%", f"{top10:.0%}")

    f1, f2, f3, f4 = st.columns(4)
    types = f1.multiselect("Customer type", sorted(risk["customer_type"].dropna().unique()))
    regions = f2.multiselect("Region", sorted(risk["region"].dropna().unique()))
    statuses = f3.multiselect("Status", sorted(risk["status"].unique()))
    search = f4.text_input("Search ID", placeholder="TR-0102")

    view = risk.copy()
    view.insert(0, "rank", range(1, len(view) + 1))
    if types:
        view = view[view["customer_type"].isin(types)]
    if regions:
        view = view[view["region"].isin(regions)]
    if statuses:
        view = view[view["status"].isin(statuses)]
    if search:
        view = view[view["transformer_id"].str.contains(search.strip().upper())]
    view["range"] = [f"{pct(lo)}–{pct(hi)}" for lo, hi in zip(view["prob_low"], view["prob_high"])]

    st.dataframe(
        view[["rank", "transformer_id", "customer_type", "customers", "failure_prob_12m", "range",
              "health_score", "risk_score", "status", "data_quality"]],
        hide_index=True, width="stretch", height=520,
        column_config={
            "failure_prob_12m": st.column_config.ProgressColumn("Failure chance (12 m)", min_value=0, max_value=1, format="%.2f"),
            "range": "Likely range",
            "health_score": st.column_config.NumberColumn("Health (0–100)", format="%d"),
            "risk_score": st.column_config.NumberColumn("Risk", format="%.0f"),
        })
    st.caption("Health is blank when there is no usable oil test: an untested unit is never shown as healthy.")


# ----------------------------------------------------------------------------
# 2. TRANSFORMER DETAIL - why is this one where it is?
# ----------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("Transformer detail")
    tid = st.selectbox("Transformer (sorted by risk)", risk["transformer_id"])
    row = risk[risk["transformer_id"] == tid].iloc[0]

    if row["status"] != "scored":
        st.warning(f"**{row['status'].replace('_', ' ')}**: {row['data_quality']}. Treat the number with care; inspection recommended.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure chance, next 12 months", pct(row["failure_prob_12m"]), help="Calibrated probability")
    c2.metric("Likely range", f"{pct(row['prob_low'])} – {pct(row['prob_high'])}")
    c3.metric("Risk", f"{row['risk_score']:.0f}", help=f"impact weight {row['impact_weight']:,.0f}")
    c4.metric("Health score", "–" if pd.isna(row["health_score"]) else f"{row['health_score']:.0f} / 100")
    st.caption(f"{row['customer_type']} · {row['customers']:,.0f} customers · {row['region']} · size {row['rating_class']} · {row['data_quality']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Why: top factors")
        drivers = pd.DataFrame(row["top_drivers"])
        if len(drivers):
            st.dataframe(drivers, hide_index=True, width="stretch")
        st.subheader("Health points lost, per field")
        st.write(health_explanation(row))
        st.bar_chart(pd.Series({"gas": row["gas_points"], "trend": row["trend_points"],
                                "load": row["load_points"], "age": row["age_points"]}, name="points lost"))
    with right:
        st.subheader("Oil-gas history (ppm, log scale)")
        tests = read_table(f"SELECT * FROM silver_oil_test WHERE transformer_id = '{tid}' ORDER BY test_date")
        events = read_table(f"SELECT month, event FROM silver_operation WHERE transformer_id = '{tid}' AND event != 'none'")
        if tests.empty:
            st.info("No oil test on record: the score relies on age and loading only.")
        else:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            dates = pd.to_datetime(tests["test_date"])
            for gas in GASES:
                ax.plot(dates, tests[gas], marker="o", label=gas.replace("_ppm", ""))
            fixed = tests["unit_fixed"] == 1
            if fixed.any():
                ax.scatter(dates[fixed], tests.loc[fixed, "co_ppm"], s=160, facecolors="none", edgecolors="red",
                           label="corrected ppb→ppm")
            for _, ev in events.iterrows():
                ax.axvline(pd.Timestamp(ev["month"]), color="grey", linestyle="--")
                ax.text(pd.Timestamp(ev["month"]), ax.get_ylim()[1], ev["event"], fontsize=8, rotation=90, va="top")
            ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=3)
            st.pyplot(fig)
        st.subheader("Loading and oil temperature")
        op = read_table(f"SELECT month, load_pct, top_oil_temp_c FROM silver_operation WHERE transformer_id = '{tid}' ORDER BY month")
        st.line_chart(op.set_index("month"))


# ----------------------------------------------------------------------------
# 3. BUDGET PLAN - what do we do with the money?
# ----------------------------------------------------------------------------
elif page.startswith("3"):
    st.title("Budget plan")
    costs = read_table("SELECT * FROM silver_costs")
    default_budget = float(read_table("SELECT annual_budget FROM silver_budget")["annual_budget"].iloc[0])
    budget = st.number_input("Budget", min_value=0.0, value=default_budget, step=250_000.0, format="%.0f")

    plan = make_plan(risk, costs, budget)
    baseline = age_plan(risk, costs, budget)
    actions = pd.DataFrame(plan["actions"])

    # Two different kinds of value, reported separately by plan.py - never add them up:
    #   risk_removed  = a replacement makes the risk go away;  risk_revealed = an inspection may uncover hidden risk
    def count(p, action):
        return sum(1 for a in p["actions"] if a["action"] == action)

    removed, n_replace = plan["risk_removed"], count(plan, "replace")
    checked, n_inspect = plan["risk_revealed"], count(plan, "inspect")
    age_removed, age_n = baseline["risk_removed"], count(baseline, "replace")
    total_risk = plan["risk_before"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spent", f"{plan['total_cost']:,.0f}", f"of {budget:,.0f}", delta_color="off")
    c2.metric(f"Risk removed by {n_replace} replacements", f"{removed / total_risk:.1%}",
              f"{(removed - age_removed) / total_risk:+.1%} vs replace-oldest")
    c3.metric(f"Replace-oldest plan ({age_n} units)", f"{age_removed / total_risk:.1%}")
    c4.metric(f"Inspections ({n_inspect} units)", f"{checked:,.0f}", "hidden risk checked", delta_color="off")

    st.bar_chart(pd.DataFrame({"fleet risk removed by replacements": [removed, age_removed]},
                              index=["our plan (knapsack)", "replace oldest first"]), horizontal=True)
    st.caption("Inspections are cheap and go where we know least: their value is the risk they might reveal, "
               "so it is shown separately and never added to the risk removed.")

    st.subheader("Proposed actions: every one needs an engineer's approval")
    if len(actions):
        actions["requires_approval"] = True
        st.dataframe(actions, hide_index=True, width="stretch",
                     column_config={"cost": st.column_config.NumberColumn(format="%.0f"),
                                    "risk_before": st.column_config.NumberColumn(format="%.0f"),
                                    "risk_removed": st.column_config.NumberColumn(format="%.0f")})
        if st.button("✅ Approve this plan"):
            run_id = f"plan-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
            dim_action = read_table("SELECT action_key, action, rating_class FROM dim_action")
            rows = (actions.merge(risk[["transformer_id", "rating_class"]], on="transformer_id", how="left")
                           .merge(dim_action, on=["action", "rating_class"], how="left"))
            run = pd.DataFrame([{"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                                 "kind": "plan: approved", "as_of": as_of, "budget": budget,
                                 "model_version": risk["model_version"].iloc[0]}])
            with P.engine.begin() as conn:
                run.to_sql("dim_run", conn, if_exists="append", index=False)
                rows.assign(run_id=run_id)[["run_id", "transformer_id", "action_key", "action", "cost",
                                            "risk_before", "risk_removed"]].to_sql("fact_plan", conn, if_exists="append", index=False)
            st.success(f"Plan approved and recorded as `{run_id}` ({len(rows)} actions).")
    else:
        st.info("The budget does not cover any action.")


# ----------------------------------------------------------------------------
# 4. NEW DATA & WHAT CHANGED - the inspection feedback loop
# ----------------------------------------------------------------------------
elif page.startswith("4"):
    st.title("New data & what changed")
    st.caption("An inspection is a new oil test. Load it, and the pipeline re-runs, the model re-scores and the plan updates. "
               "If a unit turns out worse than predicted, its risk goes up.")

    table = st.selectbox("Which table is the file for?", ["oil_tests", "operation", "customers_served", "costs_budget"])
    uploaded = st.file_uploader("CSV in the HCL Annex B format", type="csv")
    rehearsal = ROOT / "data" / "late_drop" / ("oil_tests_late.csv" if table == "oil_tests" else "costs_budget_late.csv")
    use_rehearsal = rehearsal.exists() and table in ("oil_tests", "costs_budget") and st.checkbox(f"Use the rehearsal file {rehearsal.name}")

    if st.button("Load and re-score", disabled=not (uploaded or use_rehearsal)):
        costs = read_table("SELECT * FROM silver_costs")
        budget_before = float(read_table("SELECT annual_budget FROM silver_budget")["annual_budget"].iloc[0])
        plan_before = make_plan(risk, costs, budget_before)
        save_score_run(risk, as_of, "before load")

        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
            path = tmp.name
        else:
            path = str(rehearsal)
        with st.spinner("Checking the file, rebuilding the warehouse, re-scoring..."):
            result = load_csv(table, path)
        (st.success if result["status"] == "ok" else st.error)(
            f"{result['status']}: {result['rows_loaded']} rows loaded, {result['rows_rejected']} rejected. {result['detail']}")

        if result["status"] == "ok":
            new_risk, _ = scored_fleet(warehouse_version(), as_of)
            save_score_run(new_risk, as_of, "after load")
            budget_after = float(read_table("SELECT annual_budget FROM silver_budget")["annual_budget"].iloc[0])
            plan_after = make_plan(new_risk, read_table("SELECT * FROM silver_costs"), budget_after)

            diff = risk[["transformer_id", "failure_prob_12m", "risk_score", "status"]].merge(
                new_risk[["transformer_id", "failure_prob_12m", "risk_score", "status"]],
                on="transformer_id", suffixes=(" before", " after"))
            diff["risk change"] = diff["risk_score after"] - diff["risk_score before"]
            changed = diff[diff["risk change"].abs() > 1e-9].sort_values("risk change", key=abs, ascending=False)
            st.subheader(f"{len(changed)} transformers changed")
            st.dataframe(changed, hide_index=True, width="stretch")

            before = {a["transformer_id"]: a["action"] for a in plan_before["actions"]}
            after = {a["transformer_id"]: a["action"] for a in plan_after["actions"]}
            moves = [{"transformer_id": t, "before": before.get(t, "—"), "after": after.get(t, "—")}
                     for t in sorted(set(before) | set(after)) if before.get(t) != after.get(t)]
            st.subheader(f"Plan changes (budget {budget_before:,.0f} → {budget_after:,.0f})")
            st.dataframe(pd.DataFrame(moves), hide_index=True, width="stretch")

    st.divider()
    ingested = sorted((P.DATA_DIR / "ingested").glob("*.csv")) if (P.DATA_DIR / "ingested").exists() else []
    st.caption(f"Files loaded so far: {', '.join(f.name for f in ingested) or 'none'}")
    if ingested and st.checkbox("I want to remove every loaded file and go back to the original data (demo reset)"):
        if st.button("Reset to the original data"):
            for f in ingested:
                f.unlink()
            P.run_pipeline()
            st.cache_data.clear()
            st.success("Back to the original data.")


# ----------------------------------------------------------------------------
# 5. MODEL & DATA QUALITY - can we trust it?
# ----------------------------------------------------------------------------
else:
    st.title("Model & data quality")
    if model_ready:
        card = model_info()
        st.subheader("Model card")
        st.json(card, expanded=False)
        reliability = ROOT / "reports" / "reliability.png"
        if reliability.exists():
            st.image(str(reliability), caption="Reliability: does a predicted 20% really mean about 20%?")
        experiments = ROOT / "reports" / "experiments.csv"
        if experiments.exists():
            st.subheader("Every model we tried")
            st.dataframe(pd.read_csv(experiments), hide_index=True, width="stretch")
    else:
        st.warning("Model not trained yet: run `python -m src.model`.")

    st.subheader("Data validation (runs before every analysis)")
    report = ROOT / "reports" / "validation_output.txt"
    st.code(report.read_text(encoding="utf-8") if report.exists() else "No validation report yet.")
