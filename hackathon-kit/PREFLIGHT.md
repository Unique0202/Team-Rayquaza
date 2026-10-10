# Preflight — run tonight, and again tomorrow morning

## 1. The automated half

```bash
python preflight.py
```

25 checks: Python, packages, SQLite features (window functions, recursive CTE, upsert,
rollback), pandas↔DB round trip, the join-fan-out guard, headless charting, git config, and a
**miniature bronze→silver→gold pipeline run twice to prove idempotency.**

Exit 0 = the stack will run. **Re-run it tomorrow morning before you leave the house.**

### Verified on this machine, 2026-10-10

| | |
|---|---|
| Python | 3.13.11 (miniconda) |
| pandas | **3.0.0** |
| SQLAlchemy | 2.1.4 — **was missing, installed 10 Oct** |
| matplotlib | 3.10.8 |
| SQLite | 3.51.0 (window functions need ≥3.25, upsert ≥3.24 — fine) |
| git | 2.54.0, user `Unique0202` |
| requests / openpyxl | present |

**Optional, not installed** — only matters if you choose that path: `streamlit` (stretch UI) ·
`psycopg2` (only if HCL mandates PostgreSQL). If you want either, install **tonight**:
`pip install streamlit psycopg2-binary`

---

## 2. 🔴 The finding worth more than the fix

Preflight caught a real one, and it is **a viva answer, not just a bug**.

**`pandas.to_sql` issues its `CREATE TABLE` outside the surrounding transaction.** So on a first
run, `with engine.begin()` does *not* give blanket atomicity — an **empty table survives a
rollback**, even though **no rows ever do**.

Proven on this machine:

| Pattern | After a mid-load failure |
|---|---|
| `to_sql` creating a new table inside `begin()` | table exists, **0 rows** |
| `to_sql` appending to an existing table inside `begin()` | **fully rolled back** |
| Two appends, second fails | **both rolled back** |

**The fix, now in `pipeline.py`:** `create_schema()` makes the tables empty first (DDL), then
`load_gold()` is strictly **append-only** inside the transaction (DML only) — which rolls back
completely. Verified: 0 rows leaked on a simulated mid-load failure.

**Say this if they ask about ACID.** It is specific, you found it by testing your own stack, and
it hits criterion 4 — *"understand the output instead of pasting it without understanding"* — far
harder than reciting the four letters:

> *"The load runs in a transaction so a partial failure rolls back. One thing I found while
> testing: pandas issues the CREATE TABLE outside the transaction, so on a first run an empty
> table can survive a rollback even though no rows do. So I create the schema in a separate step
> and keep the transactional write append-only — that way the data is genuinely atomic."*

---

## 3. ⚠️ pandas 3.0 — your AI tool needs telling

**You are on pandas 3.0.0, a brand-new major version.** Most tutorials, Stack Overflow answers and
model training data assume 2.x. The one that will actually bite:

- **Copy-on-Write is the default.** Chained assignment (`df[df.a > 1]['b'] = x`) **silently does
  nothing**. Always `df.loc[mask, 'b'] = x`.
- `inplace=True` behaves differently — just reassign: `df = df.drop(...)`.

Already noted in `CLAUDE.md` §4 so the agent sees it.

---

## 4. The manual half — a script cannot check these

### Git & GitHub — highest risk, because `ls-remote` only proved *read* access

- [ ] 🔴 **Do a real push tonight.** Create the hackathon repo, copy this kit in, commit, push.
      Read access working does **not** mean push will — GitHub needs a **PAT or SSH key**, not a
      password.
- [ ] Confirm credentials are **cached** — a second push must not prompt.
- [ ] `.gitignore` committed **before** your first real commit, so `NOTES.local.md` never enters
      history. Verify: `git check-ignore -v NOTES.local.md` should print a match.
- [ ] Repo URL known or open in a tab. Browser logged into GitHub on the laptop you're bringing.

### Your AI tool

- [ ] Logged in and working **on the laptop you're bringing**.
- [ ] Check usage/quota now — running out at 14:00 is a four-hour problem.
- [ ] **It needs internet.** See connectivity.
- [ ] 🔴 **Test it tonight on this kit:** open the folder, let it read `CLAUDE.md`, ask it to
      summarise the plan back to you. **If it can't restate the architecture and the constraints,
      fix the brief tonight — not at 10:05.**

### Connectivity

- [ ] 🔴 **Phone hotspot tested, with data balance.** Campus venue wifi is the single most likely
      thing to fail, and your AI tool is dead without a connection.
- [ ] Everything pip-installed **already**. Assume you cannot install anything tomorrow.

### The machine

- [ ] **Fresh-terminal test:** close everything, open a *new* terminal, run `python preflight.py`.
      Catches a PATH that only works in an already-configured shell.
- [ ] Battery actually holds charge · **charger packed** · extension cord if you have one, sockets
      are scarce and shared.
- [ ] Windows Update won't ambush you — check and reboot tonight, not tomorrow.
- [ ] Editor installed, opens the kit folder cleanly.

### Materials

- [ ] A **sample CSV** downloaded locally, in case their data is late or awkward.
- [ ] `hackathon-kit/` copied into the repo: `CLAUDE.md` · `NOTES.local.md` · `.gitignore` ·
      `README.md` · `requirements.txt` · `src/pipeline.py` · `docs/design.md` · `preflight.py`
- [ ] **Notebook and pen** — the 10:00–11:00 hour is paper work.
- [ ] College ID, charger, water.

---

## 5. Tomorrow morning, before you leave

```bash
python preflight.py      # expect: 25 passed, 0 failed
git push                 # expect: no credential prompt
```

Two commands. If both are clean, your machine will not be the reason anything fails.
