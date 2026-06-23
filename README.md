---
title: Sage
emoji: 🦉
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---
<!-- HF Spaces metadata above; hidden on HF, visible on GitHub -->

[![CI](https://github.com/vxa8502/sage-recommendations/actions/workflows/ci.yml/badge.svg)](https://github.com/vxa8502/sage-recommendations/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Demo](https://img.shields.io/badge/demo-Live-green)](https://vxa8502-sage.hf.space)

# Sage

A recommendation system that refuses to hallucinate. Sage grounds product explanations in retrieved customer-review evidence and declines to overclaim when evidence is weak.

```json
{
  "query": "budget bluetooth headphones",
  "recommendations": [{
    "explanation": "Reviewers say \"For $18 Bluetooth headphones there is no better pair\" [review_141313]...",
    "confidence": {"is_grounded": true},
    "citations_verified": true
  }]
}
```

**Try it:** [vxa8502-sage.hf.space](https://vxa8502-sage.hf.space)

---

## Workflow Status

The checked-in repo now targets a strict Stage 0 scaffold state.

What that means:

- `data/` shows scaffold structure, placeholder directories, README guidance,
  and the small checked-in manual boundary source used for `boundary_eval`
- Stage 1 data-staging outputs are intentionally absent from the clone-state
- experiment and evaluation artifacts are intentionally absent from the
  clone-state

The workflow is modular:

1. Stage 1 data staging
2. Stage 2 experiments
3. Stage 3 formal evaluation

The experiment charter still lives in
[home/EXPERIMENTATION.md](/Users/victoriaalabi/Projects/sage/home/EXPERIMENTATION.md:1),
but experiments do not begin until Stage 1 has produced the local query-bank
and corpus-alignment artifacts. Stage 1 now produces a judged retrieval holdout
plus a disjoint `faithfulness_seed` pool; the frozen explanation benchmark is
materialized later from that seed pool during Stage 2. It also merges a small
checked-in `boundary_eval` slice so refusal, clarification, low-evidence, and
recency-sensitive behavior are part of the canonical Stage 1 bank rather than
an ad hoc prompt set.

---

## Architecture

```
User Query: "wireless earbuds for running"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      SAGE API (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│  1. EMBED         │  E5-small (384-dim)                    │
│  2. CACHE CHECK   │  Exact + semantic cache                │
│  3. RETRIEVE      │  Qdrant vector search                  │
│  4. AGGREGATE     │  Chunk → Product                       │
│  5. EXPLAIN       │  Claude/GPT + evidence                 │
│  6. VERIFY        │  Quote / citation / faithfulness checks│
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Response:                                                  │
│  - Product ID + score                                       │
│  - Explanation with [citations]                             │
│  - Confidence metadata                                      │
│  - Verification results                                     │
└─────────────────────────────────────────────────────────────┘
```

**Data flow after Stage 1:** Amazon Electronics reviews → filtering and
chunking → embeddings in Qdrant → product aggregation → evidence-grounded
explanations.

---

## Limitations

| Constraint | Behavior |
|------------|----------|
| Insufficient evidence | Refuses to explain |
| Low relevance | Refuses to explain |
| Single category (Electronics) | Architecture supports broader coverage; current corpus is narrower |
| No image features | Text-only retrieval |
| English only | Embedding setup is English-first |
| Cold start | Hosted latency and cache behavior should be remeasured after the reset |

---

## Quick Start

```bash
git clone https://github.com/vxa8502/sage-recommendations && cd sage-recommendations
cp .env.example .env   # Then add QDRANT_URL, QDRANT_API_KEY, and one LLM key
```

**Hosted Qdrant:** Create a Qdrant Cloud cluster first and copy its endpoint + API key into `.env`.

**Run locally against your hosted cluster:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pipeline,api,anthropic]"
sage health
sage qdrant status
sage data build
sage serve
```

**CLI:** `sage --help`

If you are already inside `.venv`, use `sage ...`.
`-m` is a Python flag, so `-m sage.cli ...` by itself is not a shell command.
Fallback form when you want it explicitly: `python -m sage.cli ...`

**Reduced CLI surface:** The public CLI keeps the core paths only: build data, run the full evaluation, demo the system, serve the API, inspect Qdrant, clear rerunnable artifacts, and run contributor checks.

**Clean restart:** `sage reset artifacts` clears saved evaluation outputs only.
`sage reset eval-dev` is the simpler paired reset for the sampled Stage 3 dev
lane.
`sage reset experiments` clears the broader rerunnable experimentation surface
while preserving any Stage 1 staging artifacts that already exist locally.
`sage reset stage0` goes further and returns local `data/` state to the Stage 0
scaffold contract. Add `--dry-run` first if you want a preview.

**Make:** `make help` shows the canonical CLI surface. `make ci-fresh` is the one special-case Make target kept to recreate `.venv` and catch GitHub-CI-only failures.

---

## Workflow

### Stage 1: Data Staging

The checked-in repo does not ship staged data. Stage 1 is responsible for
creating it.

Expected Stage 1 outputs:

- `data/indexed_product_ids.json`
- `data/query_bank/query_bank.jsonl`
- `data/query_bank/manifest.json`
- optionally `data/query_bank/query_candidates.jsonl` as supplemental
  raw-source inventory

Within the canonical bank, the key Stage 1 subsets are:

- `gate_calibration`
- `retrieval_eval`
- `faithfulness_seed`
- `boundary_eval`

Every canonical query-bank row is also expected to carry structured provenance
covering source, curation mode, selection policy, and subset assignment.

Typical Stage 1 sequence:

1. Stage raw ESCI query data under `data/query_bank/sources/`
2. Keep the checked-in `data/query_bank/sources/manual_boundary_queries_v1.jsonl`
   source alongside it
3. Run `scripts/kaggle_pipeline.py` with valid Kaggle and Qdrant credentials
4. Build the overlap-filtered canonical bank with
   `scripts/build_esci_overlap_query_bank.py`
5. Freeze query-bank metadata in `data/query_bank/manifest.json`

If those local Stage 1 outputs already exist, `sage stage data all` now blocks
before overwriting them. Use `--allow-overwrite` when you intentionally want an
in-place refresh, or `sage reset stage0` when you want to go back to the clean
scaffold boundary first.

### Stage 2: Experiments

Experiments choose configs. They happen after Stage 1 has fixed the corpus and
query-bank reality, and they stop before the frozen metrics pass.

The exact repo-state contract for this stage lives in
[data/README.md](/Users/victoriaalabi/Projects/sage/data/README.md:167).

Core work:

- retrieval experiments
- evidence-gate calibration
- materializing frozen explanation cases from `faithfulness_seed`
- explanation grounding comparisons on frozen cases
- report-only Sofia-lite slice checks on recency-sensitive and
  complaint-oriented queries

Expected Stage 2 outputs:

- decision artifacts under `data/calibration/` or another non-reporting
  experiment location
- frozen explanation artifacts under `data/explanations/`
  (`faithfulness_cases.jsonl`, `faithfulness_case_outcomes.jsonl`, and the
  manifest for the chosen retrieval profile)
- active config or code updated to reflect the intended Stage 3 settings
- optional provisional boundary diagnostics before the formal Stage 3 gate
- no frozen reportable evaluation snapshot yet

### Stage 3: Formal Evaluation

Formal evaluation is the frozen, reportable metrics pass run after Stage 2 has
settled the intended config story.

## After Stage 1

Once `scripts/kaggle_pipeline.py` has populated hosted Qdrant successfully and
the query-bank artifacts have been created locally, the recommended next steps
are:

```bash
sage reset experiments --dry-run
sage stage experiments all
# review the holdout artifacts, choose both decisions, and update config if promoting:
sage stage experiments finalize --decision baseline-retained --retrieval-decision baseline-retained --with-boundary
# if both decisions are already known and the repo config already matches them:
sage stage experiments full --decision baseline-retained --retrieval-decision baseline-retained --with-boundary
# then begin Stage 3:
sage eval run
sage eval summary
```

Notes:

- the exact command-by-command Stage 2 runbook now lives in
  [data/README.md](/Users/victoriaalabi/Projects/sage/data/README.md:388)
- `sage stage experiments all` stops at the decision checkpoint on purpose; it
  does not rewrite config automatically
- the default Stage 2 holdout now evaluates `retrieval_dev_holdout` only; add
  `--subsets retrieval_dev_holdout,faithfulness_dev_seed` only when you
  explicitly want a diagnostic read on the dev seed pool before case freezing
- `sage stage experiments finalize` freezes faithfulness artifacts from the
  current repo config only after you pass an explicit
  `--decision baseline-retained|candidate-promoted` and
  `--retrieval-decision baseline-retained|candidate-promoted`; it also verifies
  that the current gate and retrieval settings still match the reviewed
  holdout-backed decisions before freezing or running the optional provisional
  boundary check
- `sage stage experiments full` is the two-command replay path for a known
  Stage 2 decision: pair it with `sage reset experiments` when you want one
  command to rerun calibration, holdout, and finalize without removing the
  explicit decision requirements
- `sage eval run` now expects the finalized Stage 2 handoff metadata in the
  frozen faithfulness manifest; direct `materialize_faithfulness_cases.py`
  output alone is not enough for the canonical Stage 3 workflow
- `sage eval run` now evaluates the full frozen `faithfulness_cases` set by
  default; use `--samples <n>` only when you explicitly want a deterministic
  stratified sample
- RAGAS scope is now configured separately via `--ragas-samples`; leaving it at
  the default runs the reference metric on the full evaluated case set
- Skip `sage data build` after a successful bulk indexing run unless you explicitly want to rebuild the hosted collection locally.
- `sage demo` is still useful as an ad hoc smoke test, but it is not the Stage 2 handoff artifact.
- the holdout and faithfulness artifacts now carry lightweight, report-only
  query-slice summaries for recency-sensitive and complaint-oriented asks
- `sage eval boundary` can be used as a late-Stage-2 guardrail check, but its
  saved diagnostics are still provisional until the formal Stage 3 pass
- For day-to-day work, prefer the CLI directly when a CLI entrypoint exists.
- `sage eval summary` only becomes meaningful after a fresh evaluation run has
  been saved.

---

## API Reference

### POST /recommend

```bash
curl -X POST https://vxa8502-sage.hf.space/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless earbuds for running", "k": 3, "explain": true}'
```

Returns products with grounded explanations, confidence metadata, and verified citations.

### POST /recommend/stream

Server-sent events for token-by-token explanation streaming.

### GET /health, /metrics, /cache/stats

Health check, Prometheus metrics, and cache statistics.

---

## Evaluation

```bash
sage eval dev        # run the sampled Stage 3 dev lane
sage eval run        # generate the full evaluation suite after Stage 1/2
sage eval summary    # print the latest saved evaluation snapshot, if one exists
python scripts/evaluate_boundary_behavior.py  # run the boundary_eval guardrail benchmark directly
sage reset eval-dev --dry-run
```

The checked-in repo intentionally does not ship a local metric snapshot. Treat
the first saved Stage 3 run as the baseline for the current working cycle.
`sage eval run` now expects frozen Stage 2 faithfulness cases to exist under
`data/explanations/faithfulness_cases.jsonl`.
It also evaluates the full frozen case set by default and records explicit
scope metadata when you choose a sampled run or a separately sampled RAGAS
reference pass.
The dedicated boundary benchmark writes `boundary_behavior_latest.json` under
`data/eval_results/` for canonical full-scope runs. Query-limited boundary
smokes now write `boundary_behavior_dev_latest.json` instead so they cannot
overwrite the trusted Stage 2 / Stage 3 guardrail artifact.
The exact Stage 3 entry criteria, invariants, outputs, and completion test now
live in [data/README.md](/Users/victoriaalabi/Projects/sage/data/README.md).
The same file now also contains both the canonical full-scope Stage 3 path and
the cleaner sampled dev-iteration path.

---

## Project Structure (Key Directories)

```
sage/
├── adapters/       # External integrations (Qdrant, LLM, verification)
├── api/            # FastAPI routes, middleware, Prometheus metrics
├── config/         # Settings, logging, thresholds
├── core/           # Domain models, aggregation, verification, chunking
├── services/       # Business logic (retrieval, explanation, cache)
scripts/
├── pipeline.py     # Data ingestion and embedding
├── evaluation.py   # Retrieval evaluation and baselines
├── evaluate_boundary_behavior.py # Guardrail benchmark for refusal/clarification behavior
├── faithfulness.py # Explanation verification and grounding checks
├── load_test.py    # Hosted latency and cache behavior measurement
```

---

## License

Academic/portfolio use only. Uses Amazon Reviews 2023 dataset.
