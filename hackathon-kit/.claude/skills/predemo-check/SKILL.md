---
name: predemo-check
description: Use before the 16:00 demo, or when asked "are we ready" or "predemo". Read-only review of the project against the MUST list in CLAUDE.md section 8.
---

# Pre-demo check

Report pass / fail for each item. Do not edit anything and never run a git command that writes.

- [ ] The grain (or entities and keys) is stated in `CLAUDE.md` section 5 and at the top of `src/pipeline.py`
- [ ] One command runs everything and prints a status line per stage
- [ ] Running it twice gives the same result and no duplicates
- [ ] Multi-step writes are in one transaction (create schema first, then append-only)
- [ ] At least 3 real `assert` data-quality checks, and the status message count matches the number of asserts
- [ ] Every merge into a fact table uses `validate="many_to_one"`
- [ ] Three real queries answer the user stories, and each non-obvious one has a one-line comment
- [ ] One chart or a clean printed summary exists
- [ ] README has no placeholders left (`<...>`, `_<...>_`), including the panel feedback table and the team table
- [ ] `docs/design.md` is filled and panel feedback was acted on
- [ ] `NOTES.local.md` section 5 has no open items, and `git check-ignore -v NOTES.local.md` matches
- [ ] Nothing outside the agreed stack was added

Then run `python preflight.py` and report the result. Finish with the top three risks for the demo.
