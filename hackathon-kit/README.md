# <PROJECT NAME>

HCLTech Campus Hackathon · 11 Oct 2026 · Squad <N>

**One-line pitch:** _<what this does, for whom, in plain language>_

---

## The problem

_<Restated in one sentence, in the user's language — not technical.>_

**Who uses this, and the decision they make with it:**
_<...>_

## User stories

1. As a _<role>_, I want _<capability>_ so that _<outcome>_.
2. As a _<role>_, I want _<capability>_ so that _<outcome>_.
3. As a _<role>_, I want _<capability>_ so that _<outcome>_.

**Implemented:** 1, 2  ·  **Not implemented:** 3 — see *What we cut*.

## Architecture

```
SOURCE -> BRONZE (raw, immutable) -> SILVER (clean, deduped, validated)
       -> GOLD (star schema) -> ANALYTICS (SQL + chart)
```

| Layer | What's in it | Why |
|---|---|---|
| **Bronze** | Raw as ingested, `_ingested_at`, `_source`. Never edited. | The replay tape — a transform bug is re-runnable without re-fetching the source |
| **Silver** | Deduped on business key, typed, null-handled, 3 DQ asserts | One place where data quality is enforced |
| **Gold** | Star schema: `fact_<event>` + dimensions | Shaped for reading, not for storing |

## Data model

**GRAIN: one row of `fact_<event>` = one _<...>_.**

Every measure in the fact table is true at that grain.

**Fact:** `fact_<event>` — _<FKs>_ + measures _<name (additive / semi-additive / non-additive)>_

**Dimensions:** `dim_date` (full calendar), `dim_<...>`, `dim_<...>` — each with a surrogate
integer key, because business keys change, get reused, and differ across source systems.

**SCD:** dimensions are Type 1 (overwrite). `dim_<...>` should be **Type 2** — see *Known limitations*.

## Trade-offs — what we chose and what we rejected

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Schema | **Star** | Snowflake | Fewer joins, faster reads; accepted redundancy in dimensions |
| Pipeline | **ELT / medallion** | ETL | Land raw first so a transform bug is re-runnable |
| History | **SCD Type 1** | Type 2 | Scope. We know what we gave up: history on `dim_<...>` |
| Load | **Full refresh** | Incremental | At this data size, incremental needs a watermark and adds failure modes we can't test in four hours |
| Cadence | **Batch** | Streaming | The decision this supports is daily, not per-second |
| Code layout | **Single module** | Package | At this size, import friction costs more than it buys. We'd split the moment a second source arrives |

## What we built vs what we cut

**Built:** _<...>_

**Cut, deliberately:** _<...>_ — we prioritised one slice that runs end to end over several
half-finished features.

## How we used AI, and how we validated it

**What the AI generated:** _<...>_

**What we specified ourselves:** the problem framing, the grain, the data model, the architecture,
and the scope decisions.

**How we validated it:** _<specific checks — row counts before/after joins, `validate="many_to_one"`
on merges, the three DQ asserts, running the pipeline twice to confirm idempotency>_

**A mistake we caught:** _<the specific thing the AI got wrong and how we found it>_

## Known limitations

- `dim_<...>` is Type 1, so we lose history on changes.
- No freshness checks on ingest.
- _<...>_

## Feedback from the design review

| Panel said | What we changed |
|---|---|
| _<verbatim>_ | _<what we did about it>_ |

## How to run

```bash
pip install -r requirements.txt
python src/pipeline.py        # runs end to end, prints a status per stage
python src/pipeline.py        # run it twice — identical result (idempotent)
```

## Team

| Name | Owned |
|---|---|
| Unique (`Unique0202`) | Data model, silver→gold transforms, README |
| _<...>_ | Extraction, bronze, data-quality checks |
| _<...>_ | Analytics queries, chart, demo |
