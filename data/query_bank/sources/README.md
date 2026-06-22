# Query Sources

This directory mostly behaves as a Stage 0 placeholder.

Most Stage 1 source files are local staging inputs fetched or copied here and
should not be treated as checked-in scaffold.

The exception is:

- `manual_boundary_queries_v2.jsonl`

That checked-in file is a required Stage 1 source specification for
`boundary_eval`. It provides refusal, clarification, low-evidence, negative,
and freshness boundary coverage that Amazon ESCI does not supply for the
narrowed Electronics corpus.

It is also a benchmark contract, not just a source stub:

- rows declare `boundary_type`, `expected_behavior`, `evaluation_surface`,
  `challenge_family`, `challenge_tags`, and single-author provenance fields
- the slice now distinguishes easy `policy_terminal` cases from true
  `runtime_e2e` cases that should reach retrieval and explanation
- Stage 1 validation now expects meaningful breadth, including runtime recency
  coverage, before the source is considered benchmark-ready

## Recommended ESCI Fetch

Use:

```bash
sage stage data fetch-queries
```

The expected raw input for the query-bank builders is:

```text
data/query_bank/sources/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
```

After the raw source is staged locally, you can:

- run `sage stage data import-candidates` for optional candidate inspection
- run `sage stage data build-bank` after the Kaggle corpus anchor
  `data/indexed_product_ids.json` exists

During `sage stage data build-bank`, the ESCI overlap rows are merged with the
checked-in manual boundary slice to form one canonical query bank.

The Stage 1 critical path is:

1. `sage stage data fetch-queries`
2. `sage stage data run-kaggle --wait`
3. `sage stage data pull-artifacts`
4. `sage stage data build-bank`
