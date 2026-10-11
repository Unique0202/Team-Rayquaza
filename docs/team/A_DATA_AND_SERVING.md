# Member A — Data & Serving

**You own:** the synthetic data, then the API. (**UI moved to B at 14:55. Docker removed at 15:25: panel said optional.**)
**Hand this whole PDF to your Claude session and say:** *"Read this document and `docs/DATA_SPEC.md`, then work
through Part 1 step by step. Explain each step to me in 2–3 plain lines before moving on."*

> The **Shared Contract** is attached at the end of this PDF (also in `docs/team/SHARED_CONTRACT.md`).
> Your files: `scripts/generate_data.py`, `data/**`, `src/api.py`, `tests/test_api.py`,
> `requirements.txt`. **Do not edit anyone else's files.**

---

## Part 1 — Synthetic data (13:30 → 14:15, the whole team waits on this)

Follow **`docs/DATA_SPEC.md` exactly** — it is a complete brief. Order of work:

1. `scripts/generate_data.py`: hidden per-transformer attributes → monthly operation + failure process →
   oil tests → planted edge cases E1–E8 → late-drop rehearsal files → `data_card.json` + `DATA_CARD.md`.
2. Run it twice; confirm identical output (`certutil -hashfile data\operation.csv MD5`).
3. Check the printed summary: ~38% untested, 40–60 failures, ≥25 in months 0–23, faulty units fail clearly more.
4. **14:15 — ask your human to commit and push `data/` + the script** so B and C can pull it.
   Suggested: `git add scripts/ data/` · `git commit -m "add synthetic data generator and data card"`.

**Done when:** all asserts in DATA_SPEC §9 pass and the files are pushed.

---

## Part 2 — API with stubs (14:15 → 15:00)

Build `src/api.py` (FastAPI + Pydantic v2) **against the contract shapes first, with fake values**, so it works
before B and C finish. Then swap in the real calls in Part 3.

- Endpoints (Contract §9): `POST /predict`, `POST /plan`, `POST /ingest`, `GET /model/info`,
  `GET /explain/{transformer_id}`, `GET /health`.
- Pydantic request models: `PredictRequest(transformer_ids: list[str] | None, as_of: str | None)`,
  `PlanRequest(budget: float, costs: dict | None)`.
- Response JSON exactly like HCL guide §6.1 — keys: `model_version`, `as_of`, `horizon_months` (12),
  `predictions[]` with `transformer_id, status, failure_prob_12m, prob_interval_90, health_score,
  impact_weight, risk_score, top_drivers, data_quality`. Every plan action carries `requires_approval: true`.
- `invalid_input` status: malformed body, unknown transformer id, or `as_of` earlier than the data.
- `/ingest`: multipart file + form field `table`; needs **`python-multipart`** (not installed — ask the human to
  run `pip install python-multipart` now, and add it to `requirements.txt`).
- `tests/test_api.py`: use FastAPI's `TestClient`; one test per endpoint checking status code + required keys.

Run locally: `uvicorn src.api:app --reload` → open `http://127.0.0.1:8000/docs`.

---

## Part 3 — Wire it up, UI, Docker (15:00 → 15:40)

1. Replace stubs with real calls (Contract §9): `build_features` → `score` → `compute_risk`; `make_plan` + `age_plan`;
   `load_csv`; `model_info`. Write `fact_risk` / `fact_plan` rows via C's functions if they provide a helper,
   otherwise ask C.
2. **UI: owned by B now.** Your `docker-compose.yml` just starts B's `ui/app.py` (`streamlit run ui/app.py --server.port 8501`).
3. `requirements.txt` (pin versions from `pip show`): the exact list in Contract §2 rule 6 (pandas, numpy,
   sqlalchemy, scikit-learn, scipy, joblib, matplotlib, fastapi, uvicorn, pydantic, python-multipart, streamlit,
   requests, pytest, httpx). That is the list we gave the panel, so nothing more and nothing less.
4. `Dockerfile` (python:3.13-slim, `pip install -r requirements.txt`, copy repo) and `docker-compose.yml` with two
   services: `api` (`uvicorn src.api:app --host 0.0.0.0 --port 8000`, runs `python -m src.pipeline` first) and
   `ui` (`streamlit run ui/app.py --server.port 8501`). **Build the image early (by 15:15)** — `pip install`
   inside Docker needs internet, and venue wifi is slow.

**Done when:** `docker compose up` → `/health` is green, `/predict` returns real predictions, the UI shows them.

---

## Part 4 — Demo rehearsal (15:40 → 16:10)

Rehearse the judges' live test (HCL §9.1): load `data/late_drop/oil_tests_late.csv` through `/ingest`, call
`/predict` again, run `/plan` with budget 4,000,000, and show what changed. Write the sample `curl` commands for
the README and send them to B.

---

## What the others are building (so you know what to call)

- **B (Unique) — Architecture:** bronze → silver → gold in `warehouse.db`; `build_features()` (the shared feature
  function you call in `/predict`), `run_pipeline()`, `load_csv()` (your `/ingest`).
- **C — Model & Plan:** `score()`, `model_info()`, `compute_risk()`, `make_plan()`, `age_plan()`;
  `models/model.joblib` + `model_card.json`.

## You must be able to explain (Q&A is individual)
How the data is generated and why it is realistic · each planted edge case · why every endpoint exists · how
`/ingest` → re-score → re-plan works · what Docker compose starts.
