---
name: viva-prep
description: Use around 15:30 or when the human says "viva prep" or "quiz me". Quizzes the human on their own code, one question at a time, and records gaps in NOTES.local.md section 5.
---

# Viva prep

The panel points at arbitrary code and asks "why is this here?" This rehearses that.

1. Read `NOTES.local.md` (especially section 5) and the code in `src/` and `sql/`.
2. Ask ONE question at a time, in the style of a panel: pick a specific function or query and ask why it is there, what breaks without it, or what was rejected and why. Cover the grain, keys, the transaction and idempotency story, and every window-function query.
3. Wait for the human's answer. Do not reveal the answer first. Then say whether it was right, in 1-2 lines, and correct gaps plainly.
4. If the human could not answer, add it to `NOTES.local.md` section 5 as an open item.
5. Do not write code or edit anything other than `NOTES.local.md`. Never run a git command that writes.

End with a short list of still-open items.
