# HACKATHON BRIEF — read this first

> ## 🔴 ABSOLUTE RULE — GIT IS MINE. YOU NEVER TOUCH IT.
>
> **You must never run any git command that writes.** Not `git add`, `git commit`, `git push`,
> `git checkout`, `git branch`, `git merge`, `git rebase`, `git reset`, `git stash`, `git rm`, or
> anything equivalent. **Not even when I say "commit this"** — if I say that, give me the command
> and I will run it myself.
>
> Read-only git is fine and useful: `git status`, `git diff`, `git log`. Use those freely.
>
> **Why:** every commit in this repository is made by me, by hand. The git history is the
> evaluator's evidence of *individual* contribution, and it has to record a person deciding to
> commit, not a tool emitting commits.
>
> ### Commit messages: no attribution trailers, ever
>
> When you suggest a message, **never include** `Co-Authored-By:`, `Generated with`, a 🤖 emoji,
> a `Claude-Session:` line, or any other tool marker. Plain message, nothing appended.
>
> Write them **the way a student writes them**: short, lowercase-ish, imperative, one line.
> *"add bronze load with ingest timestamp"* — not a release note, no bullet lists, no `feat:`
> prefix unless I use one first, no multi-paragraph body.
>
> ### Do this instead — suggest, don't act
>
> At natural checkpoints (a stage works, a decision is made, before something risky), stop and say:
>
> ```
> Good point to commit. Suggested:
>     git add src/pipeline.py
>     git commit -m "add silver dedupe on order_id"
> ```
>
> Then wait. **Prompt me roughly every 20–30 minutes** if I have not committed — I want frequent
> small commits across the day, because a history of one giant commit at 15:55 is worth nothing as
> evidence of contribution. Nudging me is helpful; doing it for me is not.


One-day campus hackathon. **About four hours of build time.** A panel reviews the result twice and
**scores each person individually**. Everything below is the plan agreed before coding started.

**Fill the `>>> FILL AT 10:00 <<<` blanks before writing any code. If they are blank, ask — do not
guess the domain or the architecture.**

---

## 0. Constraints that override everything

- **~4 hours of build.** Scope is the binding constraint, not capability.
- **Every line must be explainable by a human under questioning.** A panel points at arbitrary code
  and asks *"why is this here?"* **Code the team cannot explain is worse than no code.**
- **Working end-to-end beats feature-complete.** One slice that runs top to bottom.
- The README is a **graded artifact**. `NOTES.local.md` (§7) is **private and never committed**.

### How you must work — not optional

1. **Explain as you go.** After each meaningful chunk, give 2–3 lines of plain English: what it
   does, why this approach, anything non-obvious. The humans are scored on *understanding your
   output* — an unexplained clever solution actively costs marks.
2. **Keep `NOTES.local.md` current** as you build. See §7. This is a standing instruction.
3. **Boring over clever.** Two ways, one more readable → take the readable one even if longer. No
   metaprogramming, no dense comprehensions, no one-liners that need decoding at 16:30.
4. **Never run a git command that writes** — see the rule at the top. Suggest the command; I run
   it. No attribution trailers in any message you suggest.
5. **No new dependencies** beyond the agreed stack without asking. A failed `pip install` on
   conference wifi costs twenty minutes nobody has.
6. **No speculative structure.** No abstract base classes, no config systems, no "we might need
   this later." There is no later.
7. **Flag uncertainty** instead of producing confident wrong code.
8. **Comment every non-obvious query** — especially window functions — with one line saying what it
   returns, in words.

---

## 1. The problem

> **>>> FILL AT 10:00 <<<**
>
> **Problem statement (verbatim):** _..._
>
> **Restated in one sentence, in the user's language:** _..._
>
> **Who the user is and the decision they make with this:** _..._
>
> **User stories:**
> 1. As a _..._, I want _..._ so that _..._
> 2. As a _..._, I want _..._ so that _..._
> 3. As a _..._, I want _..._ so that _..._

---

## 2. What the human knows — build only within this

**This is the hard boundary.** Propose nothing outside it. If the ideal solution needs something
not on this list, say so, then propose the best thing that *is* on it — the team is scored on
defending what they built, not on using impressive tools.

### Solid — can defend to depth
- **SQL** — joins, `GROUP BY`/`HAVING`, subqueries, **CTEs (incl. recursive)**, **window functions**
  (`ROW_NUMBER`/`RANK`/`DENSE_RANK`, `LAG`/`LEAD`, running totals, frame clauses), `CASE`, set ops,
  NULL semantics
- **Relational design** — ER modelling, keys (primary/candidate/composite/foreign), constraints,
  triggers, `CHECK`, cascading FKs, **normalization 1NF→BCNF and the anomalies each removes**.
  Has personally designed a **12-entity normalized PostgreSQL schema**.
- **Dimensional modelling** — star vs snowflake, **grain**, fact vs dimension, additive /
  semi-additive / non-additive measures, surrogate keys, **SCD Types 0–3**
- **Medallion** (bronze/silver/gold), OLTP vs OLAP, warehouse / lake / lakehouse, columnar (Parquet)
- **ACID** — all four, **isolation levels and the anomaly each prevents**, locking, deadlock, WAL
- **Indexing** — B+ tree, clustered vs non-clustered, composite/leftmost-prefix, covering,
  `EXPLAIN`, **and when an index hurts**
- **ETL/ELT in Python** — pandas, SQLAlchemy, idempotency, incremental vs full refresh, watermarks,
  upsert, staging, data-quality checks
- **Python**, **C++**, **JavaScript**
- **Backend** — Node.js, Express, **REST API design**, PostgreSQL, Prisma
- **Deployment** — Linux, Nginx, PM2 (has shipped a system onto a real server)
- **LLM integration** — Groq API, Llama 3.1, prompt design, prompt-level guardrails
- **Product/design** — three years of formal design education: requirements, user stories,
  wireframing, information design, **accessibility / WCAG**
- **CS fundamentals** — DSA, OOP, OS, DBMS, Network Security, Statistical ML (coursework)
- React / TypeScript / Tailwind — **has shipped, but is not a front-end specialist.** Keep UI simple.

### Does NOT have — do not propose, do not import
Cloud of any kind (AWS / Azure / GCP) · **PySpark** · **Airflow** or any orchestrator · Kafka /
Kinesis / Pub-Sub · Delta Lake / Iceberg · dbt · Docker · Kubernetes · CI/CD · automated-test
frameworks (pytest etc. — use inline `assert`) · Power BI / Tableau / Looker · LangChain /
LlamaIndex / vector DBs / RAG · Django · Java beyond coursework · ML model training.

---

## 3. Architecture — **choose it from the problem. Nothing here is mandatory.**

> **The medallion + star schema below is a prepared default for a data/analytics-shaped problem.
> It is NOT a template to apply regardless.** Forcing a pipeline architecture onto a problem that
> is not pipeline-shaped looks like someone who learned one pattern. **Criterion 1 scores *"why you
> chose your approach and its trade-offs"* — so the choice itself is being marked.**

**At 10:00, read the problem statement and pick from these. Say out loud why.**

| If the problem is… | Build | Core of it |
|---|---|---|
| **A. Analytics / insights** — *"analyse", "report on", "find patterns in", "forecast"* | **Medallion + star schema** | bronze → silver → gold, fact + dimensions, window-function SQL |
| **B. Transactional system** — *"build a system to manage / track / book / issue"* | **Normalized relational app** | 3NF schema + constraints + REST API + thin UI. **His strongest ground — he has shipped three of these.** |
| **C. LLM-assisted feature** — *"intelligent", "assistant", "summarise", "classify text"* | **API integration + guardrails** | Groq/Llama call, prompt design, domain constraint, a normalized store for the data around it |
| **D. Dashboard / reporting** | **Light star + queries + viz** | Skip bronze if the source is already clean; grain still stated |
| **E. Mixed — most likely** | **Normalized core + analytics layer on top** | OLTP tables for the writes, a small star for the reads |

**Option E is worth reaching for when the problem allows it.** Showing a normalized transactional
schema *and* a dimensional layer over it, and explaining why each exists, demonstrates the
trade-off rather than just asserting it. That is a stronger answer than either alone.

**Whichever is chosen, these hold regardless:**

- **State the grain** (analytics) or **the entities and their keys** (transactional) **before any
  table is created.**
- **Validate inputs** — a few `assert`s at the boundary, failing loudly.
- **Wrap multi-step writes in one transaction** so a partial failure rolls back.
- **Make it re-runnable.** Running twice should not duplicate or corrupt anything.
- **One command runs the whole thing**, printing a readable status per stage.

### The prepared default, in full (option A / D / E's analytics half)

```
SOURCE → BRONZE (raw, immutable, +_ingested_at, _source)
       → SILVER (deduped, typed, validated)
       → GOLD   (star schema: fact + dimensions)
       → ANALYTICS (SQL incl. window functions → chart / summary)
```

*Bronze is immutable because it is the replay tape — a transform bug is re-runnable without
re-fetching a source that may be slow, rate-limited or gone. Star over 3NF for the gold layer
because normalization optimises write integrity while the star optimises read speed, and a
dimension is loaded once and read constantly.*

---

## 4. Tech stack — defaults, overridable

| Layer | Default | Why |
|---|---|---|
| Language | **Python 3** | Agreed |
| Transforms | **pandas** | Known |
| Database | **SQLite** via `sqlalchemy` | Zero setup, no server, nothing to fail on strange wifi. Window functions since 3.25, upsert since 3.24 — everything needed. |
| API *(if option B/C)* | **Node + Express**, or **FastAPI** | Node is the one he has shipped |
| Charts | **matplotlib** | One dependency, always works |
| UI *(stretch)* | Streamlit, or a single HTML page | **Only if core is done by 14:30** |

**If HCL mandates a stack on the day, that wins — update this table first, then code.**
**If the problem is transactional and PostgreSQL is available, use it** — he knows it well.

> **WARNING - pandas is 3.0.0 on this machine, a brand-new major version.** Most tutorials and
> model training data assume 2.x. **Copy-on-Write is the default**, so chained assignment
> (`df[mask]['col'] = x`) **silently does nothing** - always `df.loc[mask, 'col'] = x`. Do not
> rely on `inplace=True`; reassign instead.
>
> **WARNING - `to_sql` issues CREATE TABLE OUTSIDE the surrounding transaction.** Verified
> 2026-10-10: an empty table survives a rollback, though rows never do. So **create the schema
> in its own step, then keep the transactional load append-only** - `pipeline.py` already does
> this via `create_schema()` + append-only `load_gold()`. **Do not "simplify" it back into one
> step.** See `PREFLIGHT.md` section 2 - this is also a prepared viva answer.

> **SQLite caveat:** weakly typed, it will store a string in an INTEGER column. The type-casting and
> the `assert`s are doing real work here, not ceremony.

---

## 5. The data model

> **>>> FILL AT 10:00 <<<**
>
> **If dimensional — GRAIN FIRST:** one row of the fact table = _..._
> Fact: _..._ (measures, each marked additive / semi-additive / non-additive)
> Dimensions: `dim_date` (**always build a real calendar table — almost every team forgets it**),
> `dim_...`, `dim_...`
> SCD: Type 1 by default for scope; `dim_...` should really be Type 2 → note it in the README.
>
> **If transactional — entities and keys:** _..._
> Normal form reached: _..._ · Constraints: _..._ · Where cascade vs restrict, and why: _..._

---

## 6. Repo layout

```
README.md          <- GRADED. Keep current as you build.
CLAUDE.md          <- this file
NOTES.local.md     <- PRIVATE. gitignored. See §7.
.gitignore
requirements.txt
/src   pipeline.py      <- all stages, one file, clear sections
/sql   analytics.sql
/docs  design.md        <- design-presentation content
/data  (input files)
```

**One `pipeline.py`, not a package:** at this size, splitting into modules costs more in import
friction than it buys in clarity. Clear function boundaries in one readable file is correct for a
four-hour build. **We would split the moment a second data source arrives.** *(A deliberate
trade-off — it goes in the presentation.)*

---

## 7. `NOTES.local.md` — private running log. **Maintain this continuously.**

**This file is never committed and never shown to the panel.** It is the human's own working record
and viva preparation. **Create it immediately and keep it updated as a standing task** — update at
natural checkpoints (a stage finished, a decision made, a bug found), not after every line.

Keep these five sections:

### 1. File map
| File / function | What it does | Why it exists |

### 2. Decision log
For every non-trivial choice: **what we chose · what we rejected · why · who decided.**
Timestamp each one. *This becomes the trade-offs answer in the viva.*

### 3. Code walkthrough
Per function, in plain English — what it does, how, and **what would break if it were removed**.
Enough that the human can explain the file without reading the code line by line.

### 4. AI attribution
| What was asked for | What you generated | How it was checked | Verdict |

Be specific and honest. **Record every mistake you made and the human caught** — the panel scores
*"do you understand the output instead of pasting it without understanding"*, and one concrete
caught mistake is worth more than any general claim of care.

### 5. 🔴 Things the human does not fully understand yet
**The most important section.** Whenever you write something the human has not explicitly confirmed
they follow — an unfamiliar idiom, a non-obvious query, a library behaviour — **add it here as an
open item.** The human closes these before 16:00.

**This list existing is the difference between finding a gap at 15:30 and finding it in front of
the panel.**

> **It is a rehearsal aid, not a script.** At 15:30 the human reads it, then explains their section
> to the team **with the file closed.** Anything that cannot be said without it is not yet known.

---

## 8. Scope — build in order, stop when the clock says stop

### MUST exist (a complete, demoable system)
A source read → validated, cleaned data → a modelled store → **three real queries answering the
user stories** → one chart or clean printed summary → one command that runs it all → README filled.

Plus, whatever the architecture: **3 `assert` data-quality checks**, **writes in a transaction**,
**re-runnable without duplicates**.

### STRETCH — only once the above runs and has been tested
SCD Type 2 on one dimension · idempotency demonstrated live by running twice · a Streamlit page or
single HTML view · `EXPLAIN` on one query with an indexing note.

### Explicitly NOT doing — a decision, not an omission
Orchestration · cloud · streaming · auth · Docker · a test framework (inline asserts instead) ·
multiple data sources · anything from the "does NOT have" list in §2.

---

## 9. Conventions

- Name functions for their stage: `extract`, `load_bronze`, `transform_silver`, `validate`,
  `build_gold`, `load_gold`, `run_analytics`.
- Tables: `bronze_<entity>`, `silver_<entity>`, `fact_<event>`, `dim_<thing>`; technical columns
  prefixed `_` (`_ingested_at`, `_source`).
- `main()` runs everything top to bottom with one status line per stage, so the demo is a single
  command.
- Commits are small and frequent, **each person under their own GitHub account**, and are
  **made by hand by the human** — the agent only ever suggests them (see the rule at the top).

---

## 10. Division of work

> **>>> FILL AT 10:15 <<<**
>
> | Person | Owns | Commits as |
> |---|---|---|
> | **Unique** | **Data model + core transforms + README** | `Unique0202` |
> | _..._ | Extraction / ingest + data-quality checks | _..._ |
> | _..._ | Queries + output/chart + demo | _..._ |

Each person is scored individually and **git history is the record of who did what.**

**15:30 — everyone stops for ten minutes and walks their section to the others.** HCL's stated
expectation: *"Divide the work, but every member should be able to explain any part of the design
and the code."*
