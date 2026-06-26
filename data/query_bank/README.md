# Query Bank

This directory starts empty in the scaffold state.

Its purpose is to hold the query-side artifacts produced during corpus indexing
and data ingestion, not to ship those artifacts in the checked-in scaffold.

## Critical Path

The required data ingestion flow is:

1. `sage stage data fetch-queries`
2. `sage stage data run-kaggle --wait`
3. `sage stage data pull-artifacts`
4. `sage stage data build-bank`

Required corpus outputs:

- `data/indexed_product_ids.json`
- `data/query_bank/query_bank.jsonl`
- `data/query_bank/manifest.json`

The canonical bank now contains:

- ESCI-overlap rows for `gate_calibration`, `retrieval_eval`, and
  `faithfulness_dev_seed` plus `faithfulness_final_seed`
- a checked-in manual `boundary_eval` slice for refusal, clarification,
  cautious-answer, and runtime freshness coverage

Recommended command:

```bash
sage stage data all
```

This critical-path build does not require `query_candidates.jsonl`.

## Optional Audit Lane

`scripts/import_esci_queries.py` can create:

- `data/query_bank/query_candidates.jsonl`

CLI wrapper:

```bash
sage stage data import-candidates
sage stage data import-candidates --version all
```

This file is useful for provenance, source inspection, and curation, but it is
not part of the strict critical path from raw source to canonical evaluation
bank.
