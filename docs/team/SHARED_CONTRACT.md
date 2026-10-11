# Shared Contract — read by every team member and every Claude session

> This is the single source of truth that lets three people build in parallel. **If your code and
> this file disagree, this file wins.** If you need to change something here (a column, a function
> signature), **tell the other two humans first** — never change it silently.

---

## 1. The problem in 4 lines

A utility has ~400 transformers and money to fix only a few. We **predict each transformer's chance of
failing in the next 12 months** (calibrated, with an uncertainty range), compute **risk = probability ×
impact** (hospital 5, industry 3, urban 2, rural 1, × customers served), and choose **replace / inspect /
nothing** within a budget with a **knapsack**, beating an age-based plan. Served through FastAPI + Streamlit +
Docker. Full HCL guide: `Problem_Statements/Transformer_Fleet_Health_and_Replacement_Planning_Participant_Guide_DS_ML.docx`.

**Mentor answer (12:00, recorded):** an inspection tells us how severe the transformer really is; if it is
worse than the model predicted, its risk should go up. → See §8 for how that drives the plan.

---

## 2. Rules for every Claude session (non-negotiable)

1. **Never run a git command that writes** (`add`, `commit`, `push`, `pull`, `checkout`, …). Suggest it; the
   human runs it. No `Co-Authored-By` / tool trailers in suggested messages.
2. **Never write inside `hackathon-kit/`** (read-only reference). Root `CLAUDE.md` applies.
3. **The HCL guide overrides `CLAUDE.md`'s stack list.** Docker, FastAPI, scikit-learn model training and
   pytest tests are **required** by HCL — use them.
4. **Only edit the files your member owns** (§3). Need something from another member's file? Ask the human.
   This is what keeps `git pull` free of conflicts.
5. **Readable over clever.** Every line must be explainable by a student under questioning. Comment every
   non-obvious query and every ML step with one plain-English line.
6. **Libraries:** pandas, numpy, scikit-learn, scipy, fastapi, uvicorn, pydantic, joblib, streamlit,
   python-multipart, pytest. **Nothing else** without asking.
7. **Python package style:** `src/__init__.py` exists; import as `from src.features import build_features`;
   run modules as `python -m src.pipeline` from the repo root.
8. **Seeds:** `SEED = 42` everywhere randomness appears.

---

## 3. Repo layout and ownership

| Path | Owner | What |
|---|---|---|
| `scripts/generate_data.py` | **A** | synthetic data generator (spec: `docs/DATA_SPEC.md`) |
| `data/*.csv`, `data/late_drop/`, `data/data_card.json`, `data/DATA_CARD.md` | **A** | generated data + data card |
| `src/__init__.py` | **B** | empty file |
| `src/pipeline.py` | **B** | bronze → silver → gold, validation. `python -m src.pipeline` |
| `src/features.py` | **B** | the **shared feature function** (training *and* API use it) + labels |
| `src/loader.py` | **B** | CLI + function to load a judge CSV (`/ingest` calls it) |
| `src/model.py` | **C** | baselines, models, validation, calibration, intervals, `score()` |
| `src/plan.py` | **C** | risk, knapsack plan, age-based plan |
| `src/api.py` | **A** | FastAPI app — thin, calls B's and C's functions |
| `ui/app.py` | **A** | Streamlit UI (3 pages) |
| `tests/test_features.py` | **B** | leakage + validation tests |
| `tests/test_plan.py`, `tests/test_model.py` | **C** | budget never exceeded, one action per unit, … |
| `tests/test_api.py` | **A** | endpoints respond in the contract shape |
| `models/` | **C** (generated) | `model.joblib`, `model_card.json` |
| `reports/experiments.csv`, `reports/evaluation.md` | **C** (generated) | experiment log, evaluation report |
| `reports/validation_output.txt` | **B** (generated) | output of the validation step |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | **A** | packaging |
| `README.md` | **B** | graded; A and C send their sections to B |
| `warehouse.db` | generated | SQLite warehouse — **gitignored**, rebuilt by the pipeline |
| `NOTES.local.md` | each person, own laptop | private, gitignored |

---

## 4. Source files (produced by A, consumed by B)

Full detail in `docs/DATA_SPEC.md`. Summary: **Annex B columns are mandatory.** The *extra* ones are ours and
may be missing in judges' files, so **B's pipeline must cope without them.**

| File | Columns |
|---|---|
| `customers_served.csv` | `transformer_id, customer_type, customers, region` + *extra* `rating_class` |
| `operation.csv` | `transformer_id, month, load_pct, top_oil_temp_c, age_years` + *extra* `event` (`none`/`failure`/`repair`) |
| `oil_tests.csv` | `transformer_id, test_date, h2_ppm, ch4_ppm, c2h2_ppm, c2h4_ppm, c2h6_ppm, co_ppm` |
| `costs_budget.csv` | `action, rating_class, cost, annual_budget` |

Data period: monthly 2023-10-01 → 2026-09-01. "Today" = 2026-10-11.

---

## 5. Warehouse tables (`warehouse.db`, SQLite) — built by B

### Bronze — raw, never edited
Rebuilt on every pipeline run from **files on disk**: `data/<table>.csv` plus every file in
`data/ingested/<table>__*.csv` (what `/ingest` saved). The files are the immutable replay tape.

| Table | Columns |
|---|---|
| `bronze_customers_served`, `bronze_operation`, `bronze_oil_tests`, `bronze_costs_budget` | all source columns **as text, exactly as received** + `_ingested_at` (TEXT ISO) + `_source` (file name) |

### Silver — validated, cleaned, typed, one row per business key

| Table | Key | Columns |
|---|---|---|
| `silver_transformer` | `transformer_id` | `customer_type, customers INT, region, rating_class, rating_class_assumed INT` (1 if extra column missing → `medium` used) |
| `silver_operation` | `transformer_id, month` | `load_pct REAL, top_oil_temp_c REAL, age_years REAL, event TEXT` (`'none'` if column absent) |
| `silver_oil_test` | `transformer_id, test_date` | `h2_ppm … co_ppm REAL` (all ppm), `unit_fixed INT` (1 if a ppb→ppm ×1000 jump was corrected) |
| `silver_costs` | `action, rating_class` | `cost REAL` |
| `silver_budget` | — | `annual_budget REAL` (one row; latest file wins) |

Cleaning rules: IDs `strip().upper()`; exact duplicates dropped; duplicate keys → keep the latest
`_ingested_at`; gaps stay **NULL** (never filled in silver).

### Gold — star schema, shaped for reading

| Table | Grain (one row = …) | Columns |
|---|---|---|
| `dim_transformer` | one transformer | `transformer_key INT` (surrogate), `transformer_id, customer_type, customers, type_weight INT, impact_weight REAL` (= type_weight × customers), `region, rating_class, rating_class_assumed` |
| `dim_date` | one calendar day | `date_key INT` (YYYYMMDD), `date, year, quarter, month, is_snapshot_date INT` |
| `fact_oil_test` | one transformer × one test | `transformer_key, date_key`, six gases, `tdcg` (sum of the six) |
| `fact_operation` | one transformer × one month | `transformer_key, date_key, load_pct, top_oil_temp_c, age_years, is_failure INT, is_repair INT` |
| **`fact_snapshot`** | **one transformer × one quarterly snapshot date** | `transformer_key, transformer_id, snapshot_date`, **all `FEATURE_COLUMNS` (§6)**, `failed_next_12m INT` (NULL = not labelable yet) |
| `run_log` | one scoring / plan run | `run_id TEXT, created_at, kind ('score'/'plan'), as_of, budget, model_version` |
| `fact_risk` | one run × one transformer | `run_id, transformer_id, failure_prob_12m, prob_low, prob_high, status, health_score, impact_weight, risk_score, risk_low, risk_high, model_version` |
| `fact_plan` | one run × one chosen action | `run_id, transformer_id, action, cost, risk_before, risk_removed` |

`run_log`, `fact_risk`, `fact_plan` are **written by C's code** (via `src/plan.py` / `src/model.py`),
table definitions created by B. Storing every run lets us show *what changed* after the late drop.

**Snapshot dates:** first day of each quarter, **2024-01-01 → 2026-10-01**.
**Labelable** only if `snapshot_date + 12 months ≤ 2026-10-01` → snapshots 2024-01-01 … 2025-10-01 have labels;
later ones have `failed_next_12m = NULL` (used for scoring only). 2026-10-01 is "now".

---

## 6. The feature list — defined ONCE in `src/features.py` (owner B)

```python
FEATURE_COLUMNS = [
    "age_years", "load_mean_12m", "load_p95_12m", "temp_mean_12m", "hot_months_12m",
    "has_oil_test", "n_tests", "months_since_last_test",
    "h2_last", "ch4_last", "c2h2_last", "c2h4_last", "c2h6_last", "co_last",
    "tdcg_last", "tdcg_delta", "c2h2_delta",
    "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6",
    "months_since_repair",
]
```

Every feature uses **only data dated strictly before `snapshot_date`** (this is the leakage rule).

| Feature | Definition |
|---|---|
| `age_years` | from the latest `silver_operation` month before the snapshot |
| `load_mean_12m`, `load_p95_12m` | mean / 95th percentile of `load_pct` over months in [snapshot − 12 months, snapshot) |
| `temp_mean_12m` | mean `top_oil_temp_c`, same window |
| `hot_months_12m` | count of months in the window with `top_oil_temp_c > 80` |
| `has_oil_test` | 1 if any usable test before the snapshot, else 0 |
| `n_tests` | number of usable tests before the snapshot |
| `months_since_last_test` | (snapshot − last test date) in days / 30.44; NULL if no test |
| `*_last` | the six gases of the latest usable test; NULL if none |
| `tdcg_last` | sum of the six gases (total dissolved combustible gas) of the latest test |
| `tdcg_delta`, `c2h2_delta` | (last − previous) ÷ months between them; NULL if < 2 usable tests |
| `ratio_c2h2_c2h4`, `ratio_ch4_h2`, `ratio_c2h4_c2h6` | gas ratios of the latest test, denominator + 0.1 to avoid ÷0; NULL if no test |
| `months_since_repair` | months since the latest `repair` **or** `failure` event before the snapshot; NULL if none |

**"Usable test"** = a test **after** the most recent repair/failure event before the snapshot (a repair resets the
gas pattern — edge case E5). **Missing stays NULL** — handling missing values is the model's job (C).

**Label:** `failed_next_12m = 1` if `silver_operation` has `event = 'failure'` in a month **m** with
`snapshot_date ≤ m < snapshot_date + 12 months`; `0` if not; `NULL` if not labelable (§5).

---

## 7. Function interfaces — the hand-off points

```python
# src/features.py  (B)
FEATURE_COLUMNS: list[str]
def build_features(conn, snapshot_dates: list[str]) -> pd.DataFrame:
    """Columns: transformer_id, snapshot_date, *FEATURE_COLUMNS. Reads silver tables only."""
def build_labels(conn, snapshot_dates: list[str]) -> pd.DataFrame:
    """Columns: transformer_id, snapshot_date, failed_next_12m."""

# src/pipeline.py  (B)
DB_PATH = "warehouse.db"
def get_connection(): ...                     # sqlite3 / SQLAlchemy connection to DB_PATH
def run_pipeline() -> dict:                   # rebuild bronze → silver → gold; returns row counts per table

# src/loader.py  (B)
def load_csv(table: str, csv_path: str) -> dict:
    """table ∈ {oil_tests, operation, customers_served, costs_budget}.
    Checks required Annex B columns, copies the file to data/ingested/<table>__<timestamp>.csv,
    re-runs the pipeline. Returns {"rows_loaded": int, "rows_rejected": int, "status": "ok"|"error", "detail": str}."""

# src/model.py  (C)
MODEL_PATH = "models/model.joblib"; CARD_PATH = "models/model_card.json"
def train() -> dict:                          # reads fact_snapshot, runs the full evaluation, saves model + card + reports
def score(features: pd.DataFrame) -> pd.DataFrame:
    """In: output of build_features. Out per transformer: transformer_id, failure_prob_12m, prob_low, prob_high,
    status, health_score, top_drivers (list of {"feature","contribution"}), data_quality, model_version."""
def model_info() -> dict:                     # contents of model_card.json

# src/plan.py  (C)
TYPE_WEIGHTS = {"hospital": 5, "industry": 3, "urban": 2, "rural": 1}
def compute_risk(scores: pd.DataFrame, dim_transformer: pd.DataFrame) -> pd.DataFrame:  # adds impact_weight, risk_score, risk_low, risk_high
def make_plan(risk: pd.DataFrame, costs: pd.DataFrame, budget: float) -> dict:
    """{"actions": [{"transformer_id","action","cost","risk_before","risk_removed","rationale"}],
        "total_cost", "budget", "risk_before", "risk_removed", "risk_reduced_pct"}"""
def age_plan(risk: pd.DataFrame, costs: pd.DataFrame, budget: float) -> dict:          # same shape, replaces oldest first
```

**Status values** (HCL §6.2): `scored` · `scored_low_confidence` (no oil test, last test > 18 months old, or only 1 test)
· `insufficient_data` (no operation rows) · `invalid_input` (API only) · `plan` (API `/plan` only).

---

## 8. Decision logic — how replace vs inspect is decided (owner C)

1. **Model** gives each transformer `p` (calibrated) and a range `[p_low, p_high]`. Few/old/no tests → wide range.
2. **Value of each action** (risk removed):
   - **Replace:** `(p − 0.01) × impact` — a new unit is assumed to have a 1% yearly failure chance.
   - **Inspect:** `(p_high − p) × impact` — *the hidden risk an inspection could reveal.* Wide range (we know
     little) → inspection is valuable. Narrow range (we already know) → inspection is worth ~0.
   - **Rule:** a transformer with **no usable oil test or a last test > 18 months old cannot be replaced blind** —
     only inspected (don't spend ~350k on something we have not looked at).
3. **Knapsack** picks at most one action per transformer, maximising total risk removed, total cost ≤ budget
   (dynamic programming over cost in units of 5,000 — all costs are multiples of 5,000).
4. **Inspection feedback loop (mentor's point):** an inspection *is* a new oil test. Its result arrives through
   `/ingest` (`oil_tests`) → pipeline rebuilds → model re-scores → if severe, `p` goes **up** and the range
   **narrows** → replace value rises, inspect value falls → **the next plan replaces it** (if budget allows).
   If the test is clean, `p` goes down and it drops off the list. The late-drop file (new oil tests) demonstrates
   exactly this. **A human engineer approves every plan** (`requires_approval: true`).

---

## 9. API (owner A) — which function each endpoint calls

| Endpoint | Calls |
|---|---|
| `POST /predict` (optional `transformer_ids`, `as_of`) | `build_features(conn, [as_of or "2026-10-01"])` → `score()` → `compute_risk()`; writes `fact_risk` |
| `POST /plan` (`budget`, optional costs) | latest `fact_risk` → `make_plan()` + `age_plan()`; writes `fact_plan` |
| `POST /ingest` (multipart CSV + `{"table": ...}`) | `load_csv()` |
| `GET /model/info` | `model_info()` |
| `GET /explain/{transformer_id}` | `score()` for that unit → `top_drivers` + its latest tests/operation rows |
| `GET /health` | DB reachable? model file loaded? |

Response shapes: HCL guide §6.1 (`predictions[]` with `failure_prob_12m`, `prob_interval_90`, `health_score`,
`impact_weight`, `risk_score`, `top_drivers`, `data_quality`, `status`, `model_version`).

---

## 10. Timeline and sync points (code freeze 16:15)

| Time | A — Data & Serving | B — Architecture (Unique) | C — Model & Plan |
|---|---|---|---|
| 13:15–13:30 | read docs, set up | read docs, set up | read docs, set up |
| 13:30–14:15 | **generator + data card** | bronze + silver on a 5-row hand-made sample | `plan.py` + its tests (toy data) |
| **14:15 sync** | **push `data/`** | pull data, run on real files | — |
| 14:15–15:00 | `api.py` with stub responses in contract shape | gold + `features.py` + labels + leakage test | `model.py` on a stand-in `fact_snapshot` |
| **15:00 sync** | — | **push `fact_snapshot` code** | pull, train on real data |
| 15:00–15:40 | wire API to real functions; Streamlit; Dockerfile | `loader.py`, validation output, README | evaluation, calibration, model card, experiment log |
| **15:40 sync** | `docker compose up` works | README sections merged | **push `models/` + `reports/`** |
| 15:40–16:10 | rehearse demo end-to-end, fix, tag `final` | | |

**Cut list if behind (in this order):** Streamlit to one page → bootstrap intervals from 20 not 50 models →
drop the `/explain` sensor detail. **Never cut:** time-based grouped validation, calibration, the knapsack, the API.

---

## 11. Git workflow (the humans run these, not Claude)

- Everyone works on `main`, touching only their own files (§3).
- Before pushing: `git pull --rebase` then `git push`. Commit **at least every 30–45 minutes**, small messages.
- Commit `data/` and `models/` (judges and Docker need them). Never commit `warehouse.db`, `.env`, `NOTES.local.md`.
