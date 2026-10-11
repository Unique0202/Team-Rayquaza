---
name: checkpoint
description: Use at a natural checkpoint (a stage works, a decision is made, before something risky) or when the human says "checkpoint". Updates NOTES.local.md and suggests a commit for the human to run.
---

# Checkpoint

1. Read `git status` and `git diff` (read-only git only). Never run a git command that writes.
2. Update `NOTES.local.md` for what changed since the last checkpoint:
   - File map: new or changed files and functions
   - Decision log: what was chosen, rejected, why, who decided, with a timestamp
   - Code walkthrough: plain English, including what breaks if it is removed
   - AI attribution: what was asked, what was generated, how it was checked, and any mistake the human caught
   - Section 5: anything written that the human has not confirmed they follow
3. Give 2-3 lines of plain English on what was just built and why.
4. Suggest the commit and stop. Format:

```
Good point to commit. Suggested:
    git add <files>
    git commit -m "<short lowercase imperative one-liner>"
```

No `Co-Authored-By`, no "Generated with", no emoji, no `feat:` prefix, no body. Then wait for the human.
