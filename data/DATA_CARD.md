# Transformer fleet synthetic data card

transformer_fleet_synthetic v1.0 | 2026-10-11 | seed 42

## How it was made

Run `python scripts/generate_data.py` from the repository root. The generator uses Python, numpy and pandas, one seeded random stream, and the formulas in `docs/DATA_SPEC.md`. This is team-generated development and rehearsal data, not HCL's hidden evaluation data.

AI coding assistant: OpenAI Codex (GPT-6 family; exact deployment identifier not exposed). The exact user prompt was:

```text
read the Team-Rayquaza/docs/team/A_DATA_AND_SERVING.md and also read the problem statement file properly and nalso the remainng fiels in the team folder, and start working on part 1.
Team-Rayquaza/docs/team/A_DATA_AND_SERVING.md is my part whihc i have to do.
```

The referenced Member A brief directs the assistant to follow DATA_SPEC Part 1 step by step. No training metrics or held-out outcomes were consulted when constructing this dataset.

Environment: Python 3.9.6; numpy 2.0.2; pandas 2.2.3.

## Contents

| File | Rows | Dates |
|---|---:|---|
| customers_served.csv | 405 | Asset master |
| operation.csv | 14,118 | 2023-10-01 to 2026-09-01 |
| oil_tests.csv | 1,016 | 2023-10-02 to 2026-09-30 |
| costs_budget.csv | 6 | Reference |
| late_drop/oil_tests_late.csv | 40 | 2026-10-01 to 2026-10-10 |
| late_drop/costs_budget_late.csv | 6 | Reference |

## Simulation and assumptions

There are 400 units; 248 tested and 152 untested (38%). There are 48 failure events, including 26 in months 0–23. Exactly 40 units have a planted developing fault; 27 have at least two active post-onset tests with a rising marker gas.

Starting base hazard 0.0012 produced 41 failures. One deterministic rescale by 50 / 41 gives 0.0014634146. The same monthly random draws are replayed. A failure clears an existing fault immediately; a future scheduled onset is still allowed. Age continues increasing after a repair.

Active-fault monthly failure rate: 0.013111; no-active-fault rate: 0.002728. These rates use failures divided by exposed transformer-months, before telemetry gaps. 'Clearly higher' is checked as at least 2 times. These are synthetic rates, not measured utility statistics.

Rural/small units are sampled with lower test probability, then adjusted to 248 tested. Fault onsets are drawn uniformly in months 0–30; 32 tested faulty units are selected from those with two scheduled tests after onset, and 8 faults are untested. This visibility constraint intentionally biases the selected onsets toward earlier months. Six old units have only one test. Look-alike pairs copy hidden condition, weather, event draws and test dates; gas measurements differ by small noise. Customer and size proportions are initial quotas, with small changes from the planted pairs.

Planned repairs are selected from still-active surviving faults. Additional tests bracket each repair. Failure and repair events occur on the first of the month, so all subsequent tests see baseline gases. Random gaps affect only ordinary monthly telemetry, preserving every failure and repair label. October tests continue September's state; no October failure outcomes are generated.

Replacement costs 150,000 / 350,000 / 700,000 and inspection costs 10,000 / 15,000 / 25,000 for small / medium / large units are team assumptions authorised at the design review. Annual budget is 5,000,000 and rehearsal budget 4,000,000. Costs are not externally validated estimates.

## Traps and edge cases

Full row-level coordinates, including all missing months/cells, are in `data_card.json`. Its `faulty_units` and other hidden state are for validation only: the model must never read them.

E1: no tests; exercise low-confidence scoring:

```json
[
  "TR-0004",
  "TR-0007",
  "TR-0008",
  "TR-0013",
  "TR-0014",
  "TR-0016",
  "TR-0018",
  "TR-0019",
  "TR-0023",
  "TR-0027",
  "TR-0029",
  "TR-0031",
  "TR-0032",
  "TR-0034",
  "TR-0037",
  "TR-0038",
  "TR-0041",
  "TR-0050",
  "TR-0053",
  "TR-0057",
  "TR-0058",
  "TR-0059",
  "TR-0060",
  "TR-0062",
  "TR-0067",
  "TR-0068",
  "TR-0073",
  "TR-0076",
  "TR-0078",
  "TR-0079",
  "TR-0080",
  "TR-0081",
  "TR-0084",
  "TR-0088",
  "TR-0089",
  "TR-0091",
  "TR-0094",
  "TR-0095",
  "TR-0100",
  "TR-0101",
  "TR-0107",
  "TR-0108",
  "TR-0112",
  "TR-0114",
  "TR-0115",
  "TR-0117",
  "TR-0118",
  "TR-0121",
  "TR-0122",
  "TR-0126",
  "TR-0130",
  "TR-0134",
  "TR-0135",
  "TR-0136",
  "TR-0146",
  "TR-0149",
  "TR-0151",
  "TR-0152",
  "TR-0155",
  "TR-0156",
  "TR-0157",
  "TR-0158",
  "TR-0159",
  "TR-0160",
  "TR-0162",
  "TR-0163",
  "TR-0164",
  "TR-0167",
  "TR-0172",
  "TR-0174",
  "TR-0180",
  "TR-0181",
  "TR-0183",
  "TR-0188",
  "TR-0190",
  "TR-0197",
  "TR-0199",
  "TR-0200",
  "TR-0202",
  "TR-0204",
  "TR-0205",
  "TR-0209",
  "TR-0218",
  "TR-0219",
  "TR-0223",
  "TR-0227",
  "TR-0231",
  "TR-0234",
  "TR-0238",
  "TR-0239",
  "TR-0241",
  "TR-0246",
  "TR-0247",
  "TR-0249",
  "TR-0251",
  "TR-0252",
  "TR-0254",
  "TR-0260",
  "TR-0261",
  "TR-0268",
  "TR-0270",
  "TR-0273",
  "TR-0274",
  "TR-0275",
  "TR-0277",
  "TR-0281",
  "TR-0285",
  "TR-0288",
  "TR-0293",
  "TR-0294",
  "TR-0296",
  "TR-0298",
  "TR-0303",
  "TR-0306",
  "TR-0309",
  "TR-0311",
  "TR-0315",
  "TR-0317",
  "TR-0318",
  "TR-0320",
  "TR-0321",
  "TR-0322",
  "TR-0324",
  "TR-0327",
  "TR-0329",
  "TR-0331",
  "TR-0332",
  "TR-0337",
  "TR-0342",
  "TR-0344",
  "TR-0345",
  "TR-0347",
  "TR-0349",
  "TR-0353",
  "TR-0354",
  "TR-0357",
  "TR-0360",
  "TR-0366",
  "TR-0369",
  "TR-0371",
  "TR-0373",
  "TR-0379",
  "TR-0380",
  "TR-0383",
  "TR-0384",
  "TR-0385",
  "TR-0387",
  "TR-0388",
  "TR-0389",
  "TR-0393",
  "TR-0396",
  "TR-0397"
]
```

E2: old units with exactly one test:

```json
[
  "TR-0308",
  "TR-0255",
  "TR-0375",
  "TR-0021",
  "TR-0009",
  "TR-0083"
]
```

E3: similar condition, hospital versus rural impact:

```json
[
  [
    "TR-0267",
    "TR-0226"
  ],
  [
    "TR-0005",
    "TR-0221"
  ],
  [
    "TR-0339",
    "TR-0352"
  ]
]
```

E4: all six gases recorded as ppb under ppm headers; divide the flagged rows by 1000 in silver:

```json
[
  {
    "transformer_id": "TR-0307",
    "test_date": "2026-03-08"
  },
  {
    "transformer_id": "TR-0259",
    "test_date": "2025-11-13"
  },
  {
    "transformer_id": "TR-0011",
    "test_date": "2026-07-22"
  }
]
```

E5: planned repair clears the gas pattern:

```json
[
  {
    "transformer_id": "TR-0184",
    "repair_month": "2026-06-01"
  },
  {
    "transformer_id": "TR-0010",
    "repair_month": "2026-05-01"
  },
  {
    "transformer_id": "TR-0104",
    "repair_month": "2026-05-01"
  }
]
```

E6: replacing all faulty units exceeds the budget:

```json
{
  "transformer_ids": [
    "TR-0010",
    "TR-0025",
    "TR-0046",
    "TR-0066",
    "TR-0098",
    "TR-0099",
    "TR-0104",
    "TR-0127",
    "TR-0133",
    "TR-0155",
    "TR-0156",
    "TR-0161",
    "TR-0176",
    "TR-0179",
    "TR-0184",
    "TR-0187",
    "TR-0191",
    "TR-0207",
    "TR-0217",
    "TR-0231",
    "TR-0241",
    "TR-0245",
    "TR-0262",
    "TR-0285",
    "TR-0301",
    "TR-0310",
    "TR-0319",
    "TR-0322",
    "TR-0326",
    "TR-0336",
    "TR-0340",
    "TR-0341",
    "TR-0351",
    "TR-0357",
    "TR-0363",
    "TR-0366",
    "TR-0367",
    "TR-0370",
    "TR-0377",
    "TR-0390"
  ],
  "total_replace_cost": 14550000,
  "annual_budget": 5000000
}
```

E7: three exact duplicates plus lowercase TR-0045 and whitespace TR-0046:

```json
[
  "TR-0161",
  "TR-0154",
  "TR-0066",
  "TR-0045",
  "TR-0046"
]
```

E7: three repeated ordinary month rows for each of two units:

```json
[
  {
    "transformer_id": "TR-0377",
    "month": "2024-07-01"
  },
  {
    "transformer_id": "TR-0377",
    "month": "2024-04-01"
  },
  {
    "transformer_id": "TR-0377",
    "month": "2026-06-01"
  },
  {
    "transformer_id": "TR-0122",
    "month": "2026-05-01"
  },
  {
    "transformer_id": "TR-0122",
    "month": "2024-03-01"
  },
  {
    "transformer_id": "TR-0122",
    "month": "2025-03-01"
  }
]
```

E8: 288 ordinary monthly rows removed; about 1% each of load and temperature cells left empty. Exact IDs and months are recorded under `E8_missing_*` in the JSON card.

## Late-drop rehearsal

The oil file adds 30 follow-up tests and 10 first-ever tests, including at least two active faults. Use `as_of=2026-10-11` when scoring after ingest: the contract default 2026-10-01 excludes October tests under the strict-before feature rule. The budget file reduces all six annual_budget values by 20%.

## Checks

- source_schema_and_foreign_keys: passed
- positive_gases_costs_budgets: passed
- date_bounds: passed
- hcl_minimums: passed
- edge_cases_E1_E8: passed
- failure_count_and_signal: passed
- row_counts_in_memory: passed
- byte_identical_regeneration: passed (six CSVs and both data cards)
- row_counts_from_disk: passed (six CSVs)

The generator asserts schemas, references, date bounds, positive gas/cost values, all HCL minimums, E1–E8 recipes, failure totals and exposure-normalised signal. It serializes two independently generated datasets and compares every output byte. Actual CSV row counts are re-read and compared with the card. Feature leakage and train/test split checks belong to B/C, once their code exists.

## Known limits and privacy

The hazard is a simplified formula and gas patterns implement the supplied synthetic recipe, not an independently validated physical model or certified IEEE interpretation. There is seasonal ambient temperature but no observed weather, repair costs, network topology or independent field validation. Neglected units deliberately receive a 1.4x hazard multiplier; this planted association is not real-world evidence. Look-alike pairs are correlated; C should consider keeping each pair in the same validation group. Synthetic performance cannot establish performance on HCL or operational data.

All transformer IDs, customer counts and regions are synthetic. No real company or personal data is used.
