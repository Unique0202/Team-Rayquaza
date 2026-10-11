# Member B (Unique) — Architecture: Bronze → Silver → Gold

**You own:** the medallion pipeline, the shared feature function, the loader, validation and the README.
**Hand this whole PDF to your Claude session and say:** *"Read this document, then work through it step by step.
Explain each step to me in 2–3 plain lines before moving on, and keep NOTES.local.md updated."*

> The **Shared Contract** is attached at the end of this PDF (also in `docs/team/SHARED_CONTRACT.md`). It defines
> every table, column and function signature you must produce — **follow it exactly**, C and A code against it.
> Your files: `src/__init__.py`, `src/pipeline.py`, `src/features.py`, `src/loader.py`, `tests/test_features.py`,
> `reports/validation_output.txt`, `README.md`. Reference only: `hackathon-kit/src/pipeline.py` (copy ideas, never edit it).

---

## Step 1 — Bronze + silver on a tiny sample (13:30 → 14:15)

A's real data arrives ~14:15. Don't wait: hand-make `data/_sample/` with 5 transformers (a few rows per file, in
the Annex B columns + extras, include one duplicate and one lowercase ID) and build against it. **Delete
`data/_sample/` once real data lands.**

1. `src/pipeline.py` — sections in this order, one status line printed per stage:
   - `load_bronze()` — read `data/<table>.csv` + `data/ingested/<table>__*.csv` **as text** (`dtype=str`), add
     `_ingested_at`, `_source`, write `bronze_*` (Contract §5).
   - `validate()` — HCL §4.5 checks as `assert`s, results written to `reports/validation_output.txt`:
     every transformer in operation is in the asset list (`customers_served`) · gases > 0 and in ppm (after the
     unit fix) · no test date in the future · every transformer has a customer type · costs and budget > 0 ·
     row counts match `data/data_card.json` (skip with a printed note if the card is absent — judges' data).
   - `build_silver()` — IDs `strip().upper()`, drop exact duplicates, duplicate keys → keep latest `_ingested_at`,
     cast types, **unit-jump fix** (a test whose gases are ≥ 200× that transformer's median → divide by 1000,
     set `unit_fixed = 1`), missing `rating_class` → `'medium'` + `rating_class_assumed = 1`, missing `event`
     column → `'none'`. **Gaps stay NULL.**
2. **Re-runnable:** silver and gold are rebuilt in full every run (`if_exists="replace"`), so running twice gives
   identical tables. **Atomic:** write gold tables inside one `engine.begin()` transaction — and keep the kit's
   lesson: create the empty tables first, then append (see `hackathon-kit/PREFLIGHT.md` §2).

**Commit suggestion:** `git commit -m "add bronze and silver layers with validation"`.

---

## Step 2 — Gold + the shared feature function (14:15 → 15:00) — the most important step

1. `src/features.py`: `FEATURE_COLUMNS`, `build_features(conn, snapshot_dates)`, `build_labels(conn, snapshot_dates)`
   — **exactly** as Contract §6–7. Use SQL with window functions where it is clearest:
   - latest usable test before a snapshot → `ROW_NUMBER() OVER (PARTITION BY transformer_id ORDER BY test_date DESC)`
     filtered to `test_date < snapshot_date` and after the last repair/failure
   - previous test for the deltas → `LAG()` over the same partition
   - 12-month load / temperature → `AVG`, `COUNT(CASE WHEN top_oil_temp_c > 80 …)`, p95 in pandas (`quantile(0.95)`)
   Comment each query with one line saying what it returns.
2. Gold tables (Contract §5): `dim_transformer` (surrogate key, `impact_weight = type_weight × customers`),
   `dim_date`, `fact_oil_test`, `fact_operation`, **`fact_snapshot`** (features + label for snapshots
   2024-01-01 … 2026-10-01), plus `dim_action` (from `silver_costs`, surrogate `action_key`) and **empty** `dim_run`, `fact_risk`, `fact_plan` for C and A to write into. Show the panel the **conformed-dimension matrix** (Contract §5): `dim_transformer` and `dim_date` are shared by every fact table.
3. `tests/test_features.py` (pytest):
   - **Leakage test:** for every `fact_snapshot` row, the latest test date / operation month used is `< snapshot_date`
     (HCL validation check).
   - A repaired transformer's gas features ignore tests from before the repair.
   - `failed_next_12m` is NULL for snapshots after 2025-10-01.
4. **15:00 — push.** Tell C: *"`fact_snapshot` is on main."* Suggested:
   `git commit -m "add gold star schema and snapshot features"`.

**Done when:** `python -m src.pipeline` prints row counts per table, `fact_snapshot` has ~4,800 rows
(400 × 12 snapshots), and the tests pass.

---

## Step 3 — Loader, validation output, README (15:00 → 15:40)

1. `src/loader.py`: `load_csv(table, csv_path)` (Contract §7) + CLI
   `python -m src.loader --table oil_tests --file path.csv`. Reject rows missing required Annex B columns or with
   an unknown transformer id; count them in `rows_rejected`. Copy accepted file to `data/ingested/`, re-run the
   pipeline. A's `/ingest` calls this.
   **Late-drop rule (HCL §4.6):** a renamed column must load without code changes → keep a small
   `COLUMN_ALIASES = {"old_name": "annex_b_name"}` dict at the top of `loader.py` (configuration, not code).
2. `README.md` (graded): architecture diagram (the medallion strip), setup and run steps, sample `curl` commands
   (from A), assumptions (incl. the mentor's inspection answer and our 1% new-unit risk), limitations,
   edge cases and how each is handled, AI-usage disclosure, team contribution table. Collect sections from A and C.

---

## What the others are building

- **A — Data & Serving:** `data/*.csv` at ~14:15 (spec `docs/DATA_SPEC.md`), then the API calling your
  `build_features()` / `load_csv()`, Streamlit, Docker.
- **C — Model & Plan:** reads your `fact_snapshot`, trains, writes `models/`; `src/plan.py` writes into your
  `dim_run` / `fact_risk` / `fact_plan`.

## You must be able to explain (Q&A is individual)
Each medallion layer and why it exists · the grain of `fact_snapshot` and how leakage is prevented · the
window-function queries · the unit-jump fix and the repair reset · why star over 3NF, full refresh over
incremental · why the transaction is split into create-then-append.
