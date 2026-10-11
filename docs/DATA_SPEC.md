# Synthetic Data Spec — Transformer Fleet Health (HCL DS/ML edition)

> **Read this whole file before writing any code.** It is a complete, self-contained brief for
> generating the four CSV files the team needs. HCL does **not** provide data — we generate all of it.
> If anything here is unclear or impossible, **stop and ask the human** rather than guessing.

---

## 0. Ground rules for the agent doing this

- **Never run a git command that writes** (`add`, `commit`, `push`, etc.). Suggest the command; the
  human runs it. No `Co-Authored-By` or tool trailers in suggested commit messages. (See `CLAUDE.md`.)
- **Never write inside `hackathon-kit/`** — it is read-only reference.
- **Libraries: Python stdlib + `numpy` + `pandas` only.** Nothing else is needed. Do not pip install.
- **Deterministic:** one fixed seed (`SEED = 42`) via `numpy.random.default_rng(SEED)`. Running the
  script twice must produce byte-identical CSVs.
- **Readable code** — the human must explain every line to a judging panel. One file, clear
  functions, a comment on anything non-obvious. No classes, no config systems, no clever one-liners.
- **No real company or personal data** anywhere. IDs and regions are invented.

### What to produce

| Path | What |
|---|---|
| `scripts/generate_data.py` | The generator. `python scripts/generate_data.py` writes everything below. |
| `data/customers_served.csv` | Asset master (one row per transformer, plus planted duplicates) |
| `data/operation.csv` | Monthly loading / temperature / age, plus failure & repair events |
| `data/oil_tests.csv` | Sparse dissolved-gas test results |
| `data/costs_budget.csv` | Action costs per size class + annual budget |
| `data/late_drop/oil_tests_late.csv` | Rehearsal copy of the 13:45 late data drop (new tests) |
| `data/late_drop/costs_budget_late.csv` | Rehearsal copy of the late drop (budget cut 20%) |
| `data/data_card.json` | Machine-readable: row counts per file + IDs of every planted edge case |
| `data/DATA_CARD.md` | Human-readable data card (HCL Annex D — template in §8) |

---

## 1. Why this data exists (context the generator must respect)

The team builds an ML model that predicts **the chance each transformer fails in the next 12 months**,
then picks which to replace/inspect within a budget. **On judging day HCL loads *their own* hidden
data (same schema) through our loader and scores our model against real outcomes.** So:

1. **Values must be physically realistic** (typical dissolved-gas levels, loading, temperatures).
   A model trained on silly data will score badly on theirs.
2. **Failures must be learnable but noisy** — driven by age, loading and developing gas faults,
   with randomness. Not a perfect rule, not pure noise.
3. **Every edge case HCL lists must be planted on purpose** and its IDs recorded in the data card,
   so the team can show each trap being handled in the demo.

---

## 2. Global settings

| Setting | Value |
|---|---|
| Seed | `42` |
| Number of transformers | **400** — IDs `TR-0001` … `TR-0400` (4-digit zero-padded) |
| Operation period | **36 months: 2023-10-01 to 2026-09-01** (first day of each month) |
| Oil-test dates | between 2023-10-01 and 2026-09-30 |
| Late-drop test dates | between 2026-10-01 and 2026-10-10 |
| "Today" | 2026-10-11 — **no date anywhere may be later than this** |
| CSV format | UTF-8, comma-separated, header row, **no index column**, `.` decimal, dates `YYYY-MM-DD` |
| Missing values | empty cell (pandas writes NaN as empty — that is correct) |

---

## 3. The four files — exact schema

**Annex B columns are mandatory and must be spelled exactly as below.** We add a few **extra columns**
(Annex B allows extras) — marked *EXTRA*. The team's pipeline must still work without them, because
the judges' files will only contain Annex B columns.

### 3.1 `customers_served.csv` — the asset master

| Column | Type | Rule |
|---|---|---|
| `transformer_id` | string | `TR-0001` … `TR-0400` |
| `customer_type` | string | `hospital` 10% · `industry` 20% · `urban` 35% · `rural` 35% |
| `customers` | int | by type — hospital 5,000–20,000 · industry 50–2,000 · urban 2,000–15,000 · rural 200–3,000 (uniform int) |
| `region` | string | one of `North`, `South`, `East`, `West`, `Central` (uniform) |
| `rating_class` *EXTRA* | string | `small` 50% · `medium` 35% · `large` 15% — needed to price actions |

Rows: 400 + planted duplicates (§6). **This file is the "asset list"** referred to in HCL's validation checks.

### 3.2 `operation.csv` — monthly operation

| Column | Type | Rule |
|---|---|---|
| `transformer_id` | string | |
| `month` | date | first of month, 2023-10-01 … 2026-09-01 |
| `load_pct` | float | average load as % of rating, 1 decimal (see §4.2) |
| `top_oil_temp_c` | float | °C, 1 decimal (see §4.2) |
| `age_years` | float | age at that month, 2 decimals; increases by 1/12 each month |
| `event` *EXTRA* | string | `none`, `failure` or `repair` — the **training labels** come from this |

Rows: 400 × 36 = 14,400, minus dropped gap rows, plus planted duplicate rows (§6).

### 3.3 `oil_tests.csv` — dissolved gas analysis (sparse)

| Column | Type | Rule |
|---|---|---|
| `transformer_id` | string | only transformers that are "tested" (§4.3) |
| `test_date` | date | |
| `h2_ppm` | float | Hydrogen |
| `ch4_ppm` | float | Methane |
| `c2h2_ppm` | float | Acetylene |
| `c2h4_ppm` | float | Ethylene |
| `c2h6_ppm` | float | Ethane |
| `co_ppm` | float | Carbon monoxide |

All gases: **strictly positive** (floor at 0.5), 1 decimal, unit **ppm** (except the planted unit-jump rows, §6).

### 3.4 `costs_budget.csv` — small reference table (exactly 6 rows)

| action | rating_class | cost | annual_budget |
|---|---|---|---|
| replace | small | 150000 | 5000000 |
| replace | medium | 350000 | 5000000 |
| replace | large | 700000 | 5000000 |
| inspect | small | 10000 | 5000000 |
| inspect | medium | 15000 | 5000000 |
| inspect | large | 25000 | 5000000 |

(`annual_budget` repeats on every row — that is how Annex B lays it out.) **These costs and the budget are our
assumptions. At the 12:00 design review the panel told us to assume them.** Record that in `DATA_CARD.md`.

---

## 4. How to generate it — the hidden "true" story

Generate in this order. Keep the hidden per-transformer attributes in a pandas DataFrame
(`assets`) while generating; **do not write hidden attributes to the CSVs** except where stated
(they go into `data_card.json` for the team's own checking).

### 4.1 Per-transformer hidden attributes

| Attribute | Rule |
|---|---|
| `age_at_start` (on 2023-10-01) | 25% uniform 2–10 · 40% uniform 10–30 · 35% uniform 30–50 years |
| `base_load` | normal(65, 15), clipped to 20–120. **Plus 10 overloaded units** with base_load uniform 95–115 |
| `load_trend` | 20% of units grow +0.2 load-points per month; others 0 |
| `is_tested` | has any oil test. **Target ~62% tested, ~38% untested** (HCL minimum: ≥30% untested). Make rural and small units **less likely** to be tested (e.g. tested-probability 0.50 for rural/small, 0.70 otherwise, then adjust to hit ~62%) |
| `fault` | **40 units** get a developing fault (~10%). Type: `thermal` 50% · `partial_discharge` 30% · `arcing` 20%. **At least 25 of them must be tested units with ≥2 tests after onset** (HCL minimum: ≥10 clear developing faults visible in the data) |
| `fault_onset_month` | uniform integer 0–30 (index into the 36 months) |

### 4.2 Monthly operation (for every transformer, every month m = 0…35)

- **Season** (Northern hemisphere, hot May–Aug): `season = sin(2π · (calendar_month − 2) / 12)`
- `load_pct = base_load + load_trend·m + 8·season + normal(0, 4)`, clipped 5–150
- `ambient = 25 + 10·season`
- `top_oil_temp_c = ambient + 0.45·load_pct + normal(0, 2)`, **+5 if a thermal fault is active**, clipped 15–120
- `age_years = age_at_start + m/12`

### 4.3 Failure process (this creates the labels — get it right)

For each transformer, walk month by month and draw a failure with monthly probability `h`:

```
h = base · age_mult · load_mult · fault_mult · untested_mult
base          = 0.0012                         # tune, see below
age_mult      = exp((age_years − 25) / 15)     # 50-year-old ≈ 5×, 10-year-old ≈ 0.4×
load_mult     = 1 + max(0, load_pct − 85) / 20 # overload raises risk
fault_mult    = 1 if no active fault, else 1 + 0.6 · min(months_since_onset, 15)
untested_mult = 1.4 if not is_tested else 1    # PLANTED: neglected units fail more — disclose it
```

- If a failure is drawn: set that month's `event = "failure"`. The unit is **repaired the same
  month**: its active fault is cleared and its gases return to baseline from the next test on.
  It stays in the data under the same ID (it can, rarely, fail again).
- **Tune `base`** so total failures over 36 months land in **40–60** (~3–4% per year — realistic and
  rare). Do it deterministically: run once, then rescale `base` by `50 / actual` and regenerate once.
  Print the final count. **Assert** at least 25 failures land in months 0–23 (the team needs
  12 months of future to label a snapshot, so only the first 24 months are labelable).
- **Assert** that units with an active fault have a clearly higher failure rate than units without
  (the signal must be learnable). Print both rates.

### 4.4 Oil tests (tested units only)

- First test in month uniform 0–8; then every 6–12 months (uniform int) until month 35 →
  typically 3–6 tests per unit. `test_date` = a random day within that month.
- **Baseline gases** per transformer (draw once, lognormal around these medians, σ = 0.3):

| Gas | Baseline median (ppm) | Notes |
|---|---|---|
| h2 | 40 | |
| ch4 | 30 | |
| c2h2 | 0.5 | 80% of units sit at the 0.5 floor; 20% at 1–2 |
| c2h4 | 15 | |
| c2h6 | 20 | |
| co | 150 + 6 · age_years | CO rises with paper insulation ageing |

- **Developing fault growth**, `k = months since onset` while the fault is active:

| Fault | Gas multipliers on baseline |
|---|---|
| `thermal` | ch4 × (1 + 0.12k) · c2h4 × (1 + 0.25k) · c2h6 × (1 + 0.08k) |
| `partial_discharge` | h2 × (1 + 0.30k) · ch4 × (1 + 0.05k) |
| `arcing` | c2h2 + 2.0·k (additive) · h2 × (1 + 0.20k) · c2h4 × (1 + 0.10k) |

- Each test value = (baseline × multipliers) × lognormal noise (σ = 0.2), floored at 0.5, rounded to 1 decimal.

These ratios are deliberately the textbook patterns (thermal → ethylene/methane, partial discharge →
hydrogen, arcing → acetylene), so a documented gas rule (IEEE C57.104 / Duval-style) works on them.

---

## 5. HCL minimums — the generator must assert all of these

- [ ] ≥ 300 transformers (we use 400)
- [ ] ≥ 30% of transformers have **no** oil test
- [ ] ≥ 10 transformers with a **clear developing fault** visible in their tests (≥2 rising tests after onset)
- [ ] ≥ 2 look-alike pairs serving very different customers (we plant 3, §6)

---

## 6. Planted edge cases (HCL §4.4) — exact recipe, record every ID in `data_card.json`

| # | Edge case | How to plant it |
|---|---|---|
| E1 | **No oil test at all** | the ~38% untested units (§4.1) |
| E2 | **Old transformers with few measurements** | ≥ 6 tested units with `age_at_start` > 40 that get **exactly 1** oil test |
| E3 | **Look-alike pairs, different impact** | 3 pairs. Second unit of each pair copies the first's `age_at_start`, `base_load`, `load_trend`, `rating_class`, `fault` and test schedule; its gas values = first unit's × lognormal(σ 0.05). One unit is `hospital` with 8,000–15,000 customers, the other `rural` with 200–600 |
| E4 | **Gas values jump because of a changed unit** | 3 tested units (≥3 tests each): pick one **non-first** test and multiply **all six gases by 1000** (recorded in ppb under the ppm column). No unit column — the jump must be detectable from the data |
| E5 | **Recent repair resets the pattern** | 3 tested units with a developing fault: in a month 2–4 months before 2026-09 set `event = "repair"`, clear the fault from then on (gases back to baseline at later tests). Make sure each has ≥1 test before and ≥1 test after the repair |
| E6 | **Budget cannot cover all high-risk units** | already true by design: 40 faulty units × replace cost ≫ 5,000,000. **Assert** total replace cost of all faulty units > `annual_budget` |
| E7 | **Duplicated assets** | `customers_served.csv`: 3 exact duplicate rows, plus 2 near-duplicates of existing IDs with messy formatting (`"tr-0045"` lowercase, `"TR-0046 "` trailing space). `operation.csv`: for 2 units, duplicate 3 month-rows exactly |
| E8 | **Gaps** | drop ~2% of `operation.csv` rows at random (missing months); set ~1% of `load_pct` and ~1% of `top_oil_temp_c` to empty |

Make sure planted units do not collide in a way that breaks a recipe (e.g. an E2 unit should not also
be an E4 unit). Pick them with the seeded RNG from eligible units and record them.

---

## 7. Late-drop rehearsal files (`data/late_drop/`)

HCL drops a late file at 13:45: *"new oil test results for some transformers, and a changed budget."*
Generate a rehearsal version so the team can practise:

- `oil_tests_late.csv` — same schema as `oil_tests.csv`, dates 2026-10-01 … 2026-10-10:
  new tests for 30 already-tested units (continue their gas story) **plus 10 previously-untested
  units** getting their first-ever test (at least 2 of these 10 should show a fault pattern).
- `costs_budget_late.csv` — same 6 rows, `annual_budget = 4000000` (a 20% cut).

---

## 8. Data card — `data/data_card.json` and `data/DATA_CARD.md`

**`data_card.json`** — the team's validation script reads this to check "row counts match the data card":

```json
{
  "name": "transformer_fleet_synthetic", "version": "1.0", "created": "2026-10-11", "seed": 42,
  "row_counts": {"customers_served.csv": 0, "operation.csv": 0, "oil_tests.csv": 0, "costs_budget.csv": 6},
  "n_transformers": 400, "n_untested": 0, "n_failures": 0, "n_failures_months_0_23": 0,
  "edge_cases": {
    "E2_old_few_tests": [], "E3_lookalike_pairs": [["TR-....", "TR-...."]],
    "E4_unit_jump": [{"transformer_id": "TR-....", "test_date": "YYYY-MM-DD"}],
    "E5_repair_reset": [{"transformer_id": "TR-....", "repair_month": "YYYY-MM-01"}],
    "E7_duplicate_assets": [], "E7_duplicate_operation": []
  },
  "faulty_units": [{"transformer_id": "TR-....", "fault": "thermal", "onset_month": "YYYY-MM-01"}]
}
```
(fill real values; `faulty_units` is our own answer key — the **model must never read it**; it is
only for the team's sanity checks and the demo narrative.)

**`DATA_CARD.md`** — HCL Annex D fields, in plain English:

| Field | What to write |
|---|---|
| Name and version | `transformer_fleet_synthetic` v1.0, 2026-10-11 |
| How it was made | Script `scripts/generate_data.py`, seed 42, numpy + pandas. **Paste the prompt(s) given to the AI assistant** and name the model used (HCL requires this) |
| What it contains | each file, row count, date range |
| Traps and edge cases | E1–E8 with the IDs and what each one tests |
| Checks run | the asserts from §5 and §9 and their results |
| Known limits | failure process is a simple hazard formula; gases follow textbook patterns only; no weather, no repair costs, untested units planted to fail 1.4× more often |
| Privacy | confirm no real company or personal data |

---

## 9. Final self-checks — the script must `assert` these and print a summary

1. Every `transformer_id` in `operation.csv` and `oil_tests.csv` exists in `customers_served.csv` (after stripping/uppercasing the messy duplicates)
2. All gas values > 0
3. No date later than 2026-10-11; all main oil-test dates ≤ 2026-09-30
4. Every transformer has a `customer_type`
5. All costs and budgets > 0
6. All §5 minimums hold
7. Failure total in 40–60; ≥25 in months 0–23; faulty units fail at a clearly higher rate
8. Row counts written to `data_card.json` equal the actual CSV row counts
9. Running the script twice gives identical files (the human can check with `certutil -hashfile data\operation.csv MD5` on Windows)

The summary printed at the end should look like:

```
customers_served.csv : 405 rows (400 unique + 5 planted duplicates)
operation.csv        : 14,1xx rows
oil_tests.csv        : ~1,000 rows, 248 tested units, 152 untested (38%)
failures             : 52 total, 34 in months 0-23
fault vs no-fault monthly failure rate: 0.0xx vs 0.00x
edge cases planted   : E2=6  E3=3 pairs  E4=3  E5=3  E7=5+2
All checks passed.
```

---

## 10. When done — hand back to the human

Tell the human, in 5–6 plain lines: what the script does, the final counts, where the edge cases
are, and anything you were unsure about. Then suggest (do not run):

```
git add scripts/generate_data.py data/ docs/DATA_SPEC.md
git commit -m "add synthetic data generator and data card"
```
