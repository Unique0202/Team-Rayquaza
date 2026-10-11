# Member C — Model & Plan

**You own:** the ML model (baselines → models → honest validation → calibration → intervals → explanations) and
the risk + budget plan.
**Hand this whole PDF to your Claude session and say:** *"Read this document, then work through it step by step.
Explain every ML step to me in plain words — what it does and why — before moving on."*

> The **Shared Contract** is attached at the end of this PDF (also in `docs/team/SHARED_CONTRACT.md`). Your input
> is B's `fact_snapshot` table (Contract §5–6); your outputs are the functions in Contract §7.
> Your files: `src/model.py`, `src/plan.py`, `tests/test_model.py`, `tests/test_plan.py`, `models/**`,
> `reports/experiments.csv`, `reports/evaluation.md`. **scikit-learn only** — no xgboost/shap/mlflow.

---

## Step 1 — The plan first (13:30 → 14:15): it needs no data

`src/plan.py` — Contract §7 signatures, Contract §8 logic:

1. `compute_risk(scores, dim_transformer)`: `impact_weight = TYPE_WEIGHTS[customer_type] × customers`;
   `risk_score = p × impact`; `risk_low/high = p_low/high × impact`.
2. `make_plan(risk, costs, budget)`: per transformer three options — none / inspect / replace.
   - replace value = `(p − 0.01) × impact`, cost from `costs` by `rating_class`
   - inspect value = `(p_high − p) × impact`
   - **no usable oil test, or last test > 18 months → replace not allowed** (inspect only)
   - **multiple-choice 0/1 knapsack by dynamic programming**, costs in units of 5,000; at most one action per unit;
     total cost ≤ budget. Each chosen action gets a one-line `rationale` built from the numbers.
3. `age_plan(...)`: replace oldest first until the budget runs out — the baseline the guide says we must beat.
4. `tests/test_plan.py` (pytest, toy data): total cost never exceeds budget · at most one action per transformer ·
   a no-test unit is never replaced · with a huge budget every positive-value action is chosen · knapsack ≥ age plan
   on risk removed.

**Commit suggestion:** `git commit -m "add risk calculation and knapsack plan"`.

---

## Step 2 — Model code on a stand-in table (14:15 → 15:00)

B's real `fact_snapshot` lands ~15:00. Until then, build a **stand-in** DataFrame with the exact Contract §6
columns (random values, ~5% positives) so your code runs end to end. **Delete the stand-in when real data arrives.**

`src/model.py`, in this order — explain each in a code comment:

1. **Split (Annex A rules):** the **final held-out set** = latest labelable snapshots (2025-07-01, 2025-10-01)
   **and** a random 20% of transformers (seed 42) that never appear in training. Touched **once**, at the end.
2. **Validation on the rest:** grouped by transformer (`GroupKFold`) **and** time-aware (train folds use earlier
   snapshots than their validation fold), **repeated 5×** with different seeds → mean ± range for every metric.
3. **Baselines:** (a) age-only logistic regression; (b) rule-based health index (points for high `c2h2_last`,
   rising `tdcg_delta`, `load_p95_12m > 90`, old age) — scaled 0–100 this is also our **`health_score`**
   (NULL when no oil test).
4. **Models × missing-data methods** (HCL R5 — compare at least 3): logistic regression with (i) median
   imputation, (ii) median + missing-indicator (`SimpleImputer(add_indicator=True)`); and
   `HistGradientBoostingClassifier` with (iii) native NaN handling. **All preprocessing inside a sklearn `Pipeline`**
   so it is fitted on training folds only (Annex A "Folds" rule). `class_weight="balanced"` for the rarity.
5. **Metrics** (HCL §7): PR-AUC with interval, top-10% recall and lift, Brier score — overall **and separately for
   no-oil-test units**. One row per configuration appended to `reports/experiments.csv`
   (model, missing method, data window, params, seed, metrics).
6. **Pick the final model by the numbers** (expect logistic regression to win — say why if it does).
7. **Calibration:** `CalibratedClassifierCV(method="sigmoid")` (Platt — isotonic overfits with few positives).
   Show calibrated vs uncalibrated Brier + a reliability curve (`sklearn.calibration.calibration_curve`, saved as
   `reports/reliability.png` with matplotlib).
8. **Intervals:** train 20–50 bootstrap copies of the final pipeline; `prob_low/high` = 5th/95th percentile of their
   predictions. Units with no oil test must come out wider — check it.
9. **Explanations:** global = `permutation_importance`; per transformer = logistic-regression coefficient × scaled
   feature value → top 3 `top_drivers`.
10. `score()` + `model_info()` (Contract §7); save `models/model.joblib` and `models/model_card.json` (HCL Annex C fields).
11. **Budget test (HCL §7):** on the held-out snapshots, your plan vs `age_plan` — risk removed and real failures
    caught. Put it in `reports/evaluation.md`.

---

## Step 3 — Real data, evaluation report (15:00 → 15:40)

Pull B's `fact_snapshot`, run `python -m src.model`, read the numbers honestly, write `reports/evaluation.md`
(split, folds, seeds, metric definitions, comparison table, calibration, no-test-unit results, error analysis of
missed failures, budget test). **15:40 — push `models/` and `reports/`.** Send your README section to B.

**Never:** tune anything on the held-out set · use `event`/failure information from after a snapshot ·
fit imputation/scaling outside the pipeline. Each of these is a **disqualification** in HCL §10.

---

## What the others are building

- **A — Data & Serving:** the data (~14:15), then the API that calls your `score()`, `model_info()`,
  `compute_risk()`, `make_plan()`, `age_plan()`.
- **B (Unique) — Architecture:** `fact_snapshot` (~15:00), `build_features()` (the same function the API uses
  to score — never re-compute features yourself), and the `run_log` / `fact_risk` / `fact_plan` tables you write into.

## You must be able to explain (Q&A is individual)
Why grouped *and* time-aware validation · what PR-AUC and Brier measure and why not accuracy · why Platt over
isotonic · what the interval means and why no-test units are wider · the three missing-data methods and which won ·
how the knapsack works and why inspect value = `(p_high − p) × impact`.
