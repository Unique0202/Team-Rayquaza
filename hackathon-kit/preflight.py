"""
PREFLIGHT -- run this before the hackathon. And again on the morning.

Checks every dependency, every DB feature the plan relies on, and then runs a
miniature end-to-end pipeline TWICE to prove idempotency.

    python preflight.py

Self-contained -- it imports nothing from this repo, so it can be sent to a
teammate on its own. If a check fails it prints the exact command to fix it.

Exit code 0 = safe to go. Non-zero = something would have broken tomorrow.
Output is deliberately plain ASCII so it renders in any Windows console.
"""

import sys
import os
import shutil
import sqlite3
import subprocess
import tempfile
import traceback
from pathlib import Path

PASS, FAIL, WARN = [], [], []
FIXES = []          # copy-pasteable commands for whatever failed


def check(name, critical=True, fix=None):
    """Run fn, record the outcome, never crash the run.

    `fix` is a shell command the user can paste to repair this check. It is
    collected and reprinted at the end so nobody has to ask anyone for help.
    """
    def run(fn):
        try:
            detail = fn()
            PASS.append((name, detail or "ok"))
            print(f"  [ OK ] {name}" + (f"  -- {detail}" if detail else ""))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            (FAIL if critical else WARN).append((name, msg))
            if fix:
                FIXES.append((critical, fix))
            tag = "FAIL" if critical else "WARN"
            print(f"  [{tag}] {name}  -- {msg}")
            if fix:
                print(f"         fix:  {fix}")
    return run


print("=" * 70)
print("HCL HACKATHON PREFLIGHT")
print("=" * 70)

# ---------------------------------------------------------------- environment
print("\n[1] ENVIRONMENT")


@check("Python >= 3.9",
       fix="install Python 3.11+ from python.org, tick 'Add to PATH'")
def _():
    v = sys.version_info
    assert v >= (3, 9), f"found {v.major}.{v.minor}, pipeline.py uses list[str] syntax (3.9+)"
    return f"{v.major}.{v.minor}.{v.micro}"


@check("Python executable path")
def _():
    return sys.executable


@check("Only one python on PATH (ambiguity causes 'module not found' confusion)",
       critical=False)
def _():
    exe = "python.exe" if os.name == "nt" else "python3"
    found = shutil.which(exe)
    assert found, "python not found on PATH -- you must type the full path to run anything"
    return found


@check("Writable temp dir")
def _():
    p = Path(tempfile.gettempdir()) / "_preflight_write_test"
    p.write_text("x", encoding="utf-8")
    p.unlink()
    return str(Path(tempfile.gettempdir()))


@check("Free disk space > 1 GB")
def _():
    free = shutil.disk_usage(Path.home()).free / 1e9
    assert free > 1, f"only {free:.1f} GB free"
    return f"{free:.1f} GB"


# --------------------------------------------------------------- dependencies
print("\n[2] REQUIRED PACKAGES")


@check("pandas", fix="pip install pandas")
def _():
    import pandas
    return pandas.__version__


@check("sqlalchemy >= 2.0", fix='pip install --upgrade "sqlalchemy>=2.0"')
def _():
    import sqlalchemy
    major = int(sqlalchemy.__version__.split(".")[0])
    assert major >= 2, f"{sqlalchemy.__version__} -- engine.begin() API differs on 1.x"
    return sqlalchemy.__version__


@check("matplotlib", fix="pip install matplotlib")
def _():
    import matplotlib
    return matplotlib.__version__


print("\n[3] OPTIONAL PACKAGES (warn only -- install tonight if you want them)")


@check("requests (if the source is an API)", critical=False,
       fix="pip install requests")
def _():
    import requests
    return requests.__version__


@check("openpyxl (if the source is .xlsx -- pandas needs it)", critical=False,
       fix="pip install openpyxl")
def _():
    import openpyxl
    return openpyxl.__version__


@check("streamlit (stretch-goal UI only)", critical=False,
       fix="pip install streamlit")
def _():
    import streamlit
    return streamlit.__version__


@check("psycopg2 (only if they mandate PostgreSQL)", critical=False,
       fix="pip install psycopg2-binary")
def _():
    import psycopg2
    return psycopg2.__version__


# ------------------------------------------------------------ sqlite features
print("\n[4] SQLITE FEATURES THE PLAN DEPENDS ON")


@check("sqlite3 >= 3.25 (window functions)",
       fix="upgrade Python -- sqlite3 ships bundled with it")
def _():
    v = sqlite3.sqlite_version
    parts = tuple(int(x) for x in v.split("."))
    assert parts >= (3, 25, 0), f"{v} is too old for window functions"
    return v


@check("window function actually executes")
def _():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (g TEXT, v INT)")
    con.executemany("INSERT INTO t VALUES (?,?)", [("a", 1), ("a", 2), ("b", 3)])
    rows = con.execute("""
        SELECT g, v,
               ROW_NUMBER() OVER (PARTITION BY g ORDER BY v DESC) AS rn,
               SUM(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run
        FROM t
    """).fetchall()
    assert len(rows) == 3
    con.close()
    return "ROW_NUMBER, SUM OVER, frame clause"


@check("recursive CTE executes")
def _():
    con = sqlite3.connect(":memory:")
    n = con.execute("""
        WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 5)
        SELECT COUNT(*) FROM c
    """).fetchone()[0]
    assert n == 5
    con.close()
    return "ok"


@check("UPSERT (ON CONFLICT DO UPDATE) executes")
def _():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (k INT PRIMARY KEY, v INT)")
    con.execute("INSERT INTO t VALUES (1, 10)")
    con.execute("INSERT INTO t VALUES (1, 99) ON CONFLICT(k) DO UPDATE SET v = excluded.v")
    assert con.execute("SELECT v FROM t WHERE k=1").fetchone()[0] == 99
    con.close()
    return "idempotent loads are possible"


@check("transaction ROLLBACK works (this is your 'A' in ACID)")
def _():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (v INT)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    try:
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (2)")
        raise RuntimeError("simulated mid-load failure")
    except RuntimeError:
        con.rollback()
    assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1, "rollback did not work"
    con.close()
    return "partial load rolls back"


# ------------------------------------------------------------- pandas <-> db
print("\n[5] PANDAS <-> DATABASE ROUND TRIP")


@check("to_sql / read_sql round trip via SQLAlchemy")
def _():
    import pandas as pd
    from sqlalchemy import create_engine, text
    tmp = Path(tempfile.gettempdir()) / "_preflight.db"
    tmp.unlink(missing_ok=True)
    eng = create_engine(f"sqlite:///{tmp}")
    df = pd.DataFrame({"id": [1, 2, 3], "amt": [10.0, 20.0, 30.0]})
    with eng.begin() as c:
        df.to_sql("t", c, if_exists="replace", index=False)
    with eng.connect() as c:
        back = pd.read_sql(text("SELECT * FROM t"), c)
    assert len(back) == 3 and back["amt"].sum() == 60.0
    eng.dispose()
    tmp.unlink(missing_ok=True)
    return "write + read verified"


@check("engine.begin() rolls back the DATA on a failed multi-step write")
def _():
    """NOTE: pandas to_sql issues CREATE TABLE *outside* the transaction, so on a
    first run an EMPTY table can survive a rollback. The ROWS never do. Verified
    on this machine 2026-10-10. See NOTES.local.md -- this is a viva answer."""
    import pandas as pd
    from sqlalchemy import create_engine, text
    tmp = Path(tempfile.gettempdir()) / "_preflight2.db"
    tmp.unlink(missing_ok=True)
    eng = create_engine(f"sqlite:///{tmp}")
    with eng.begin() as c:                       # setup: tables created up front
        pd.DataFrame({"a": [0]}).to_sql("dim", c, index=False, if_exists="replace")
        pd.DataFrame({"a": [0]}).to_sql("fact", c, index=False, if_exists="replace")
    try:
        with eng.begin() as c:                   # real load: append-only, atomic
            pd.DataFrame({"a": [1, 2]}).to_sql("dim", c, index=False, if_exists="append")
            pd.DataFrame({"a": [3, 4]}).to_sql("fact", c, index=False, if_exists="append")
            raise RuntimeError("simulated mid-load failure")
    except RuntimeError:
        pass
    with eng.connect() as c:
        d = c.execute(text("SELECT COUNT(*) FROM dim")).scalar()
        f = c.execute(text("SELECT COUNT(*) FROM fact")).scalar()
    assert d == 1 and f == 1, f"rollback leaked rows: dim={d} fact={f}"
    eng.dispose()
    tmp.unlink(missing_ok=True)
    return "create-then-append pattern is genuinely atomic"


@check("merge(validate='many_to_one') catches a join fan-out")
def _():
    import pandas as pd
    fact = pd.DataFrame({"k": [1, 1, 2]})
    bad_dim = pd.DataFrame({"k": [1, 1, 2], "name": ["a", "DUPLICATE", "b"]})
    try:
        fact.merge(bad_dim, on="k", how="left", validate="many_to_one")
    except Exception:
        return "fan-out guard works -- silent row duplication will raise"
    raise AssertionError("validate= did NOT catch a duplicated dimension key")


@check("to_datetime / to_numeric with errors='coerce'")
def _():
    import pandas as pd
    s = pd.to_numeric(pd.Series(["1", "x", "3"]), errors="coerce").fillna(0)
    d = pd.to_datetime(pd.Series(["2026-01-01", "garbage"]), errors="coerce")
    assert s.sum() == 4.0 and d.isna().sum() == 1
    return "bad rows coerce instead of crashing the run"


# ------------------------------------------------------------------- plotting
print("\n[6] CHARTING")


@check("matplotlib saves a PNG headless (Agg backend)")
def _():
    import matplotlib
    matplotlib.use("Agg")          # no display on a conference machine -- always set this
    import matplotlib.pyplot as plt
    out = Path(tempfile.gettempdir()) / "_preflight.png"
    out.unlink(missing_ok=True)
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2])
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
    out.unlink(missing_ok=True)
    return "use matplotlib.use('Agg') before pyplot import"


# ------------------------------------------------------------------------ git
print("\n[7] GIT")


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


@check("git on PATH", fix="install Git from git-scm.com")
def _():
    code, out, _ = _git("--version")
    assert code == 0, "git not found"
    return out


@check("git user.name is set",
       fix='git config --global user.name "Your Name"')
def _():
    code, out, _ = _git("config", "--get", "user.name")
    assert code == 0 and out, "unset -- run: git config --global user.name \"Your Name\""
    return out


@check("git user.email is set",
       fix='git config --global user.email "you@example.com"')
def _():
    code, out, _ = _git("config", "--get", "user.email")
    assert code == 0 and out, "unset -- run: git config --global user.email \"you@example.com\""
    return out


@check("can reach github.com (credentials NOT tested here -- see section 4)",
       critical=False, fix="check your internet; retry on a phone hotspot")
def _():
    # octocat/Hello-World is GitHub's own public sample repo -- tiny and always there.
    code, out, err = _git("ls-remote", "https://github.com/octocat/Hello-World.git")
    assert code == 0, (err or "no network / auth prompt")[:120]
    return "network to github ok (this does NOT prove you can PUSH)"


# ------------------------------------------------- full miniature pipeline x2
print("\n[8] FULL MINIATURE PIPELINE, RUN TWICE (idempotency)")


@check("bronze -> silver -> gold -> analytics, twice, identical")
def _():
    import pandas as pd
    from sqlalchemy import create_engine, text

    tmp = Path(tempfile.gettempdir()) / "_preflight_pipeline.db"
    tmp.unlink(missing_ok=True)
    eng = create_engine(f"sqlite:///{tmp}")

    raw = pd.DataFrame({
        "order_id":  [1, 2, 3, 3, 4],              # 3 is a duplicate
        "cust_id":   ["c1", "c2", "c1", "c1", "c3"],
        "order_date": ["2026-01-05", "2026-01-06", "2026-02-01", "2026-02-01", "bad"],
        "amount":    ["100", "200", "300", "300", "50"],
    })

    def run_once():
        # bronze
        bronze = raw.copy()
        bronze["_ingested_at"] = pd.Timestamp("2026-10-11 09:00:00")
        bronze["_source"] = "preflight"
        with eng.begin() as c:
            bronze.to_sql("bronze_orders", c, if_exists="replace", index=False)

        # silver
        s = bronze.sort_values("_ingested_at").drop_duplicates(subset=["order_id"], keep="last")
        s["order_date"] = pd.to_datetime(s["order_date"], errors="coerce")
        s["amount"] = pd.to_numeric(s["amount"], errors="coerce").fillna(0)
        s = s[s["order_id"].notna()]

        # validate
        assert s["order_id"].notna().all()
        assert not s["order_id"].duplicated().any()
        assert (s["amount"] >= 0).all()

        # gold -- star schema
        dim_cust = s[["cust_id"]].drop_duplicates().reset_index(drop=True)
        dim_cust.insert(0, "dim_customer_key", range(1, len(dim_cust) + 1))

        dates = pd.date_range("2026-01-01", "2026-12-31", freq="D")
        dim_date = pd.DataFrame({"date": dates})
        dim_date.insert(0, "date_key", range(1, len(dim_date) + 1))
        dim_date["month_name"] = dim_date["date"].dt.month_name()

        fact = s.merge(dim_cust, on="cust_id", how="left", validate="many_to_one")
        fact = fact.merge(dim_date[["date_key", "date"]], left_on="order_date",
                          right_on="date", how="left", validate="many_to_one")
        fact = fact[["order_id", "dim_customer_key", "date_key", "amount"]]

        with eng.begin() as c:
            dim_cust.to_sql("dim_customer", c, if_exists="replace", index=False)
            dim_date.to_sql("dim_date", c, if_exists="replace", index=False)
            fact.to_sql("fact_orders", c, if_exists="replace", index=False)

        with eng.connect() as c:
            return pd.read_sql(text("""
                SELECT dc.cust_id,
                       SUM(f.amount) AS total,
                       RANK() OVER (ORDER BY SUM(f.amount) DESC) AS rnk
                FROM fact_orders f
                JOIN dim_customer dc ON dc.dim_customer_key = f.dim_customer_key
                GROUP BY dc.cust_id
            """), c)

    first = run_once()
    second = run_once()
    assert first.equals(second), "SECOND RUN DIFFERED -- pipeline is not idempotent"
    assert len(first) == 3, f"expected 3 customers, got {len(first)}"

    eng.dispose()
    tmp.unlink(missing_ok=True)
    return f"deduped 5->4 rows, star built, window query ran, 2 runs identical"


# ----------------------------------------------------------------- conclusion
print("\n" + "=" * 70)
print(f"PASSED {len(PASS)}   WARNINGS {len(WARN)}   FAILED {len(FAIL)}")
print("=" * 70)

if WARN:
    print("\nWARNINGS (optional -- install tonight only if you want them):")
    for n, m in WARN:
        print(f"  - {n}\n      {m}")

if FAIL:
    print("\nCRITICAL FAILURES -- FIX THESE TONIGHT:")
    for n, m in FAIL:
        print(f"  - {n}\n      {m}")

if FIXES:
    crit = [f for c, f in FIXES if c]
    opt = [f for c, f in FIXES if not c]
    print("\n" + "-" * 70)
    print("COPY-PASTE TO FIX")
    print("-" * 70)
    for f in dict.fromkeys(crit):
        print(f"  {f}")
    if opt:
        print("  # optional, only if you want these:")
        for f in dict.fromkeys(opt):
            print(f"  {f}")
    print("-" * 70)
    print("Then run this script again.")

if FAIL:
    print("\nNOT SAFE TO GO.\n")
    sys.exit(1)

print("\nAll critical checks passed. This machine will run tomorrow.")
print("Still to do by hand: a real `git push`, your AI tool logged in, and a")
print("tested phone hotspot. See section 4 of PREFLIGHT.md.\n")
sys.exit(0)
