# Transformer Fleet Health & Replacement Planning

HCLTech Future Ready AI Engineer Hackathon · Data Science & ML edition · 11 Oct 2026 · Team Rayquaza

**One line:** for an electricity utility that can afford to fix only a few transformers a year, we predict each
transformer's chance of failing in the next 12 months, weigh it by who would lose power, and choose which units to
replace or inspect within the budget, with a reason for every choice.

---

## The problem

A utility owns ~400 power transformers. A failure can cut power to a hospital or a village and costs far more than a
planned replacement. Oil tests (gases dissolved in the oil, the best health signal) exist for only ~60% of units, and
failures are rare.

**User:** the asset-planning engineer. **Decision:** with this year's budget, which transformers to replace, which to
inspect, and which to leave, backed by probabilities they can trust.

| The engineer asks | The system answers |
|---|---|
| What is TR-0102's chance of failing in 12 months? | A **calibrated probability with a 90% range** (wider when we know less) |
| What is the health of TR-0241? | A 0–100 **health score with points lost per field** (gas condition, gas trend, load, age). Blank, never "healthy", without an oil test |
| Which transformers carry the most risk? | **Risk = probability × impact**, ranked, with the top factors for each |
| What do we do with 5 million? | A **knapsack plan** of replace/inspect actions within budget, compared with a replace-oldest plan. Every action needs approval |
| Score TR-0399, which has never been tested | `scored_low_confidence`, a wide range, and an inspection recommendation |
| New oil tests arrived: what changed? | Load them → the warehouse rebuilds → re-score → re-plan → before/after comparison |

---

## Quick start (runs locally, Python 3.13)

```bash
pip install -r requirements.txt          # exact versions we tested with
python -m src.pipeline                    # bronze -> silver -> validate -> gold (about 3 s)
python -m src.model                       # train, evaluate, save model + reports (about 1 min)
uvicorn src.api:app                       # API on http://127.0.0.1:8000  (docs at /docs)
streamlit run ui/app.py                   # interface on http://localhost:8501
python -m pytest -q                       # 101 tests
```

Load new data (also what `POST /ingest` and the UI do):
```bash
python -m src.loader --table oil_tests --file data/late_drop/oil_tests_late.csv
```

### Sample API calls (on Windows PowerShell use `curl.exe`)
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"transformer_ids": ["TR-0102", "TR-0399"], "as_of": "2026-10-11"}'
curl -X POST http://127.0.0.1:8000/plan -H "Content-Type: application/json" -d '{"budget": 5000000}'
curl -X POST http://127.0.0.1:8000/ingest -F "file=@data/late_drop/oil_tests_late.csv" -F 'table={"table": "oil_tests"}'
curl http://127.0.0.1:8000/model/info
curl http://127.0.0.1:8000/explain/TR-0102
curl http://127.0.0.1:8000/health
```

---

## Architecture

```
 4 CSVs ──► BRONZE ─────────► SILVER ──────────────► GOLD (star schema) ──► MODEL + PLAN ──► API (FastAPI)
(Annex B)   raw, as received   cleaned, typed,        4 shared dimensions    score, risk,      UI (Streamlit)
            + _ingested_at     one row per key,       5 fact tables          knapsack          (same functions)
            + _source          9 checks BEFORE write  fact_snapshot = grain
```

| Layer | What happens | Why |
|---|---|---|
| **Bronze** | every CSV stored exactly as received (as text) + where and when it arrived; rebuilt from the files on disk, including every file `/ingest` saved | the **replay tape**: a bug in a later step is fixed and re-run without the source; audit trail |
| **Silver** | IDs tidied (`"tr-0045 "` → `TR-0045`), duplicates removed (last loaded wins), gas readings logged in ppb converted to ppm, types cast; **gaps stay NULL**; then **9 validation checks run before anything is written** | the one place data quality is enforced; bad data stops the run instead of reaching the model |
| **Gold** | star schema (below) | shaped for reading; one leakage-safe table for the model **and** the API |

**Star schema: shared (conformed) dimensions used by several fact tables**

| Fact ↓ / Dimension → | `dim_transformer` | `dim_date` | `dim_run` | `dim_action` |
|---|---|---|---|---|
| `fact_oil_test`: one row per test | ✔ | ✔ | | |
| `fact_operation`: one row per transformer-month | ✔ | ✔ | | |
| **`fact_snapshot`: one row per transformer per quarter** | ✔ | ✔ | | |
| `fact_risk`: one row per scoring run per transformer | ✔ | ✔ | ✔ | |
| `fact_plan`: one row per plan per chosen action | ✔ | ✔ | ✔ | ✔ |

**Grain of the core table:** one row of `fact_snapshot` = one transformer at one quarterly snapshot date
(2024-01-01 … 2026-10-01). Every feature uses only data dated **before** the snapshot; the label is "failed in the
12 months starting at the snapshot", and is **NULL (not 0)** when those 12 months are not all in the data yet.
The "latest usable oil test" is found with `ROW_NUMBER()`, the previous test (rate of gas rise) with `LAG()`.
**One function, `build_features()`, builds features for training and for the API**, so the two can never differ.

---

## Data

HCL provided no data, so `scripts/generate_data.py` (seed 42) generates it, with every HCL edge case planted on
purpose. Details, planted IDs and the AI prompt used are in `data/DATA_CARD.md` / `data/data_card.json`.
400 transformers · 36 months (Oct 2023 – Sep 2026) · 1,016 oil tests · 38% of units never tested · 48 failures.
Failures follow a documented rule: older, overloaded units with a developing gas fault fail more often, plus randomness.

---

## Model

**Target:** failure within 12 months of the snapshot. 3,200 labelled rows, **138 positive (4.3%)**.

| Step | What we did | Why |
|---|---|---|
| Baselines | age only; a rule-based health index (points per field, IEEE C57.104 gas condition levels) | a complex model must beat simple rules to earn its place |
| Models × missing data | logistic regression + median fill · logistic regression + median + missing flags · gradient boosting with native NaN handling | HCL asks for ≥ 2 model families and ≥ 3 missing-data methods |
| Validation | grouped by transformer **and** time-aware (train on earlier quarters), 5 repeats; final held-out set = last 2 labelled quarters × 20% of transformers, **used once** | no transformer in both train and test; never predict from the future |
| Imbalance | `class_weight="balanced"` inside the training pipeline | failures are rare |
| Calibration | Platt (sigmoid); Brier **0.182 → 0.053** | risk = probability × impact multiplies the probability directly |
| Uncertainty | 30 bootstrap copies → 90% range; untested units get wider ranges (median width 0.121 vs 0.071) | honest about what we don't know |
| Explanations | top 3 drivers per transformer; permutation importance globally | every score has a reason |

**Results, development data (grouped time-aware CV, mean and range over 5 repeats):**

| Model | PR-AUC | Top-10% recall | Brier |
|---|---|---|---|
| Age only | **0.105** (0.088–0.123) | 0.23 | 0.039 |
| Health index rule | 0.060 (0.053–0.066) | 0.13 | 0.041 |
| **Logistic regression + median (chosen)** | 0.082 (0.060–0.103) | 0.20 | 0.172 → calibrated |
| Logistic regression + missing flags | 0.079 (0.059–0.100) | 0.19 | 0.168 |
| Gradient boosting, native NaN | 0.072 (0.048–0.086) | 0.16 | 0.103 |

**Held-out (used once, 160 rows, 9 failures):** final calibrated model PR-AUC **0.104** [90% CI 0.038–0.224];
age only 0.152 [0.052–0.406]; health-index rule 0.356 [0.054–0.711].

**Honest verdict:** on our synthetic data, **the model does not beat the age-only baseline**. The failure-generating
rule is driven mostly by age, and the gas-fault signal is faint (faulty units' ethylene median 18.1 vs 14.7 ppm).
With only 9 held-out failures, all intervals overlap heavily. We keep the multi-feature model in service because the
decision process needs it: an inspection result (a new oil test) must be able to change a transformer's risk, and
an age-only model would ignore it. In a follow-up experiment, a 4-input model (age + the health index's gas/trend/load
points) scored 0.117 PR-AUC in the same CV and beat age-only. It is the next step (see *Known limitations*).
Full report: `reports/evaluation.md`; every configuration tried: `reports/experiments.csv`; model card: `GET /model/info`.

---

## From prediction to decision

- **Impact** = customer-type weight (hospital 5, industry 3, urban 2, rural 1) × customers served.
  **Risk** = calibrated probability × impact. A hospital unit at 19% outranks a rural one at 25%.
- **Replace** removes risk: `(p − 0.01) × impact` (a new unit is assumed to have a 1% yearly failure chance).
- **Inspect** reveals hidden risk: `(p_high − p) × impact`, worth most where we know least. Inspection removes nothing
  by itself; its result comes back through `/ingest` and the **next** plan acts on it.
- **Never replace blind:** a unit with no usable oil test, or a last test older than 18 months, can only be inspected.
- **Knapsack** (dynamic programming over cost in steps of 5,000): at most one action per transformer, cost ≤ budget.
- **Result (5M budget, today):** 7 replacements remove **20.9%** of fleet risk, vs **5.6%** for replacing the oldest 15
  units; 226 inspections cover 377,943 of hidden risk. Held-out budget test: our plan removes 10.0% vs 2.3% for the age
  plan, and acted on 4 real failures vs 2.
- **Every plan is a proposal** (`requires_approval: true`); approving it in the UI records it in `dim_run` / `fact_plan`.

---

## Edge cases and how each is handled

| Edge case (HCL §4.4) | Handling | Where |
|---|---|---|
| No oil test at all | `has_oil_test` feature; `scored_low_confidence`; wider range; health score blank, never "healthy"; inspect before any replacement | features, model, plan |
| Old units with few measurements | low confidence (one test); wide range | model |
| Look-alike pair, hospital vs rural | same probability, very different impact → the hospital ranks far higher | `dim_transformer`, plan |
| Gas values jump from a unit change | a test ≥ 200× the median of the same unit's **other** tests → ÷1000 and flagged (found exactly the 3 planted) | `pipeline.fix_unit_jumps` |
| Recent repair resets the pattern | gas features use only tests from the month **after** the latest repair or failure | `features.RESETS_SQL` |
| Budget cannot cover all high-risk units | the knapsack takes the best risk per unit of money; the rest wait | plan |
| Duplicate assets/rows, messy IDs | `strip().upper()`, one row per key, last loaded wins (stable sort) | `pipeline.keep_latest` |
| Missing months or values | left NULL in silver; the model's pipeline fills them (inside training folds only) | silver, model |
| Judges' files without our extra columns | no `rating_class` → `medium` (flagged); no `event` → `none` | silver |
| Renamed column in a late file | `COLUMN_ALIASES` in `src/pipeline.py`: configuration, not code | silver, loader |
| Bad rows in an upload | counted with a reason and left out; if validation then fails, the load is rolled back | `src/loader.py` |

---

## Validation and tests

- **Validation script output:** `reports/validation_output.txt`, 9/9 HCL checks (asset list, positive gases in ppm, no
  future dates, every unit has a customer type, positive costs and budget, row counts match the data card).
- **101 tests** (`python -m pytest -q`). Highlights: the **leakage test** recomputes 150 random snapshot rows in plain
  pandas from data before the snapshot only. We planted a leak on purpose (tests up to 3 months in the future) and the
  test failed, so it really guards leakage. Also: grain, repair reset, labels, API features = training features,
  loader (late drop, bad rows), budget never exceeded, one action per unit, API contract shapes.
- **Re-runnable:** silver and gold are rebuilt in full; two runs give identical tables. Gold is written in one
  transaction (tables created empty first, because pandas' `CREATE TABLE` escapes the transaction). Run history
  (`dim_run`, `fact_risk`, `fact_plan`) is never wiped, so before/after comparisons survive a rebuild.

**Reproduce the headline numbers:** `python -m src.pipeline && python -m src.model` → `reports/evaluation.md`
(seed 42 everywhere; data from `scripts/generate_data.py`, seed 42).

---

## Trade-offs: what we chose and what we rejected

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Pipeline | medallion (bronze/silver/gold) | one script raw → model | raw data kept for replay and audit; quality enforced in one place |
| Gold layer | star schema, conformed dimensions | 3NF / snowflake | read-heavy; fewer joins; type weight kept inside `dim_transformer` |
| Grain | quarterly snapshots | monthly / yearly | tests come every 6–12 months: monthly = near-duplicate rows, yearly = too few |
| Refresh | full rebuild (~3 s) | incremental + watermark | at this size incremental only adds failure modes |
| Model | logistic regression | boosting by default | few failures; readable weights; boosting did not beat it |
| Calibration | Platt (sigmoid) | isotonic | isotonic overfits with ~40 failures |
| Inspection value | uncertainty `(p_high − p) × impact` | a fixed % of risk | inspection is ~23× cheaper; a fixed % would inspect everything |
| UI | calls the same Python functions as the API | UI → HTTP → API | one set of functions, no second copy of the logic |
| Storage | SQLite | PostgreSQL server | no server to fail; window functions cover our queries |
| Packaging | local run | Docker | the panel confirmed Docker is optional; removed to focus on the system |

---

## Assumptions

- Costs (the panel told us to assume them): replace 150k / 350k / 700k, inspect 10k / 15k / 25k (small / medium / large);
  annual budget 5,000,000 (late-drop rehearsal: 4,000,000).
- A replaced unit has a 1% yearly failure chance. A test is "stale" after 18 months.
- A repair or failure resets the gas pattern; a test counts from the month after it.
- Impact weights from HCL Annex A. Judges' files without `rating_class` are priced as `medium`.

## Known limitations

- **The ML model does not beat age-only on this data** (see *Model*); the 4-input points model is the next step.
- Synthetic data: results describe our generator's world; HCL's hidden data is the real test.
- Only 138 positive rows (9 in the held-out set): every metric has a wide interval.
- `dim_transformer` is SCD Type 1 (overwrite): history of customer changes is not kept.
- The API's default scoring date is 2026-10-01; pass `"as_of"` (e.g. today) so the latest oil tests count.

---

## Feedback from the design review (12:00) and what we changed

| Panel said | What we did |
|---|---|
| "If an inspection shows the transformer is worse than predicted, its risk should go up." | An inspection **is** a new oil test: load it (`/ingest` or UI screen 4) → rebuild → re-score → risk rises, range narrows → the next plan replaces it. The rehearsal file re-scores 40 transformers live. |
| "Assume the costs of replacement and inspection, and the budget." | Assumptions above, used by `dim_action` and the plan. |
| Asked which libraries we use for data, models and everything. | Listed in the design brief: pandas/numpy, SQLite/SQLAlchemy, scikit-learn/joblib, plain-Python knapsack, matplotlib, FastAPI/Uvicorn/Pydantic, Streamlit, pytest. |
| "Docker can be removed." | Removed; `requirements.txt` pinned to the exact versions we tested with. |

---

## How we used AI, and how we validated it

**AI coding assistants** (Claude Code, OpenAI Codex) generated most of the implementation: the data generator,
pipeline, feature SQL, model and plan code, API, UI and tests. **We specified** the problem framing, the grain, the data
model, the architecture, the decision rules and the scope, in a written contract (`docs/team/SHARED_CONTRACT.md`)
that every assistant session followed. **Every commit was made by a team member, by hand.**

**How we validated it:** an independent 36-check audit of the generated data against its spec; 101 tests; a leakage
test proven by planting a leak; row counts before/after every join (`validate="many_to_one"`); a headless run of every
UI screen against the real model; and reading every reported number for plausibility.

**Mistakes we caught:**
1. The plan reported **"108.9% of risk removed"**: inspections' *revealed* risk was being added to replacements'
   *removed* risk. Found by testing the UI; the plan now reports the two separately.
2. A test "passed" for the wrong reason (it failed on a missing column, not the future date it was meant to catch). Fixed.
3. Scoring "as of 2026-10-01" would silently ignore the late-drop oil tests (dated Oct 1–10). Caught before the drop.
4. `requirements.txt` pinned pandas 2.2.3, while the code was built and tested on pandas 3.0.0 (different defaults). Re-pinned.
5. A push added a nested repository pointer instead of the data files. Diagnosed and removed.

---

## Team

| Member | GitHub | Owned |
|---|---|---|
| Unique (B) | `Unique0202` | architecture: bronze/silver/gold pipeline, validation, shared feature function, loader, Streamlit UI, README, design brief, team contract |
| A | `sreerajmahesh` | synthetic data generator and data card, FastAPI service and API tests |
| C | `abinavhub` | baselines, models, validation, calibration, intervals, explanations, risk and knapsack plan, evaluation report |
