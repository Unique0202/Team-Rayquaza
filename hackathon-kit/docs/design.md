# Design presentation — 11:00, 10–15 minutes

Fill between 10:00 and 10:50. Rehearse once at 10:50. **Unique presents Architecture + Data model.**

---

## 1. The problem, restated  · 30s
_<One sentence, in the user's language. Not technical.>_

## 2. Who the user is  · 30s
_<Who, and what decision they make with the output.>_

## 3. User stories  · 2 min
1. As a _<role>_, I want _<capability>_ so that _<outcome>_.
2. ...
3. ...

_(HCL's deck names "user stories" twice. This section is expected, not optional.)_

## 4. Architecture  · 3 min  ← UNIQUE
```
SOURCE -> BRONZE (raw, immutable) -> SILVER (clean, deduped, validated)
       -> GOLD (star schema) -> ANALYTICS
```
Say it in plain words: **raw → cleaned → business-ready.**
Bronze is immutable because it is the replay tape.

## 5. Data model  · 3 min  ← UNIQUE
**"The grain is one row per _<...>_."** — say this as one clean sentence.

Fact: _<...>_ · Dimensions: _<...>_ · Surrogate keys because _<...>_

## 6. Trade-offs — what we rejected  · 2 min
Pick 2–3. **Most squads present an approach; almost none present the one they rejected.**

- Star over snowflake — fewer joins, accepted redundancy
- ELT over ETL — land raw, so transform bugs are re-runnable
- SCD Type 1 over Type 2 — scope; and we know we gave up history
- Full refresh over incremental — incremental needs a watermark we can't test today
- Batch over streaming — the decision is daily, not per-second
- Single module over a package — import friction costs more than it buys at this size

## 7. Scope  · 1 min
**By 16:00 we will have:** _<...>_
**We are deliberately NOT doing:** _<...>_

## 8. How we'll use AI  · 30s
As an accelerator for implementation. We specify the model and the transformations; we validate
with row-count checks, join-fan-out guards, and three data-quality asserts.

---

## 🔴 PANEL FEEDBACK — write verbatim, do not paraphrase

| # | What they said | What we'll change |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

**Act on at least one. Say so explicitly at the 16:00 demo.**
