# Getting your teammates checked — tonight

`preflight.py` imports nothing from this repo, so it runs standalone on any machine. **Tested on a
clean environment:** 9 critical failures → it printed 3 commands → pasted them → all green. Nobody
needs to ask you anything.

---

## The distribution path — push the repo, send two lines

**Do this first, because it kills three birds:** it proves *your* push works, it gets everyone's
git access tested before the day, and it hands them the checker.

```bash
cd <your hackathon repo>
cp -r /e/Placement_Kit/applications/HCL/hackathon-kit/* .
cp /e/Placement_Kit/applications/HCL/hackathon-kit/.gitignore .
git add -A && git commit -m "Hackathon kit: brief, pipeline skeleton, preflight" && git push
```

Then confirm `NOTES.local.md` did **not** go up:

```bash
git check-ignore -v NOTES.local.md     # must print a match
git ls-files | grep NOTES              # must print nothing
```

**Add both teammates as collaborators** (repo → Settings → Collaborators) so they can push
tomorrow. A teammate who can only read cannot have visible contribution — and they're scored on it.

---

## The message to send

> Hey — made a setup checker for tomorrow so we don't lose build time to someone's laptop dying.
>
> Takes 2 minutes:
>
> ```
> git clone <REPO_URL>
> cd <REPO_NAME>
> pip install -r requirements.txt
> python preflight.py
> ```
>
> It checks Python, pandas, SQLAlchemy, SQLite, git, and runs a mini pipeline to prove they all
> work together. **If something's missing it prints the exact command to fix it** — paste it, run
> the script again.
>
> You want to see: `PASSED 25  WARNINGS 2  FAILED 0`
>
> (2 warnings are fine — they're optional packages.)
>
> Found two genuinely broken things on mine, so it's worth the 2 min. Ping me if yours won't go
> green.
>
> Also before tomorrow:
> • **Do one real `git push`** to the repo, not just a clone — read access doesn't mean push
>   access, and GitHub wants a token, not your password
> • **Phone hotspot with data** — venue wifi is the usual thing that dies
> • **Your AI tool logged in on the laptop you're bringing**, and check you've got quota left
> • **Charger** — sockets will be scarce
>
> Bring a notebook and pen too. First hour is design on paper, laptops shut.

---

## Worth raising with the team: keep the agents out of git

You are committing by hand. If a teammate lets *their* AI tool commit, the shared history picks up
`Co-Authored-By` trailers and tool-shaped commit bursts anyway — so the decision only holds if all
three of you make it.

Easy way to put it, without sounding preachy:

> Quick one — let's all do our own commits by hand rather than letting the AI tool push for us.
> They score us individually on visible contribution, and a history of real commits from the three
> of us reads a lot better than tool-generated ones. Small and often is the goal.

It is also just true: frequent small commits from three named people is better evidence than one
bulk commit at 15:55.

---

## If someone reports a failure — triage

| What they see | What it means | Fix |
|---|---|---|
| `'python' is not recognized` | Python not on PATH | Reinstall from python.org with **"Add Python to PATH"** ticked. Or use `py` instead of `python` on Windows. |
| `'pip' is not recognized` | same | `python -m pip install ...` always works |
| `Python >= 3.9` fails | Python 3.8 or older | Must upgrade — `pipeline.py` uses `list[str]` annotations |
| `sqlite3 >= 3.25` fails | Very old Python | Upgrade Python; sqlite3 ships bundled with it |
| `git user.name is set` fails | Never configured git | `git config --global user.name "Their Name"` |
| Everything in §2 fails | Installed into a different Python than the one running | `python -m pip install -r requirements.txt` — same interpreter, guaranteed |
| Permission errors on install | Windows perms | `python -m pip install --user -r requirements.txt` |

**On a Mac** everything works identically except `python` → `python3` and `pip` → `pip3`.

**If a teammate's machine simply won't come good:** it is not a disaster. They can own the
analytics queries and the chart, both of which can be written on *your* machine while they drive.
**But sort it tonight** — at 12:30 tomorrow it costs the team an hour.

---

## What preflight does NOT check — each person does these by hand

1. 🔴 **A real `git push`.** The script only proves network reachability to github.com. Pushing
   needs a **Personal Access Token or SSH key** — a password is rejected. Everyone pushes once
   tonight, then pushes a second time and confirms **no prompt** (credentials cached).
2. 🔴 **AI tool logged in, with quota.** On the laptop they're actually bringing.
3. 🔴 **Phone hotspot, tested, with data balance.**
4. **Fresh-terminal test** — close everything, open a *new* terminal, run it again. Catches a PATH
   that only works in an already-open shell.
5. Battery holds charge · charger packed · Windows Update already done.

---

## Tomorrow morning, all three of you

```bash
python preflight.py      # expect 25 passed, 0 failed
git push                 # expect no credential prompt
```

---

> **One thing worth noticing:** sending the team a working checker the night before is itself the
> behaviour the rubric rewards — *individual ownership* and *communication*. You are not being
> graded tonight, but it sets who the squad looks at tomorrow when the problem statement lands.
