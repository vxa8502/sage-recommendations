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

## Evaluation Results

Formal evaluation over 423,165 Qdrant points (Amazon Electronics reviews), 227 ESCI-graded retrieval queries, and 120 frozen explanation cases.

| Metric | Value | Notes |
|--------|-------|-------|
| NDCG@10 | 0.134 | 227 ESCI queries; global IDCG; retrieval holdout baseline 0.193 |
| Hit@10 | 0.308 | — |
| Recall@10 | 0.209 | — |
| Claim-level HHEM | 0.939 avg · 97.4% pass | Primary faithfulness instrument; threshold 0.5; 120/120 cases |
| RAGAS faithfulness | 0.759 ± 0.364 | Canonical n=120; secondary diagnostic |
| Cached P99 latency | ~121 ms | Meets < 500 ms target ✓ |
| Uncached latency | ~5,400 ms | Cache carries the latency SLO |
| Refusal rate | ~10% | Low-evidence queries; by design |

**Faithfulness:** HHEM is the primary instrument — it runs on every query at serving time (same 0.5 threshold as the runtime gate), covers all 120 cases, and is deterministic. RAGAS is secondary and diagnostic: the 0.759 canonical score reflects retrieval noise (queries where the top chunk describes a competing product), not generation fabrication.

**Retrieval:** NDCG@10 0.134 vs the retrieval holdout baseline 0.193 under the same corpus and formula. The gap is real and is the primary open item for a v2 upgrade (hybrid dense+sparse BM25).

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
│  5. EXPLAIN       │  Claude Sonnet + evidence              │
│  6. VERIFY        │  Quote / citation / HHEM checks        │
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

**Stack:** E5-small-v2 embeddings · Qdrant Cloud (HNSW, int8 quantization) · Claude Sonnet (`claude-sonnet-4-6`) · in-memory two-layer cache (L1 exact-match, L2 semantic similarity) · FastAPI · Prometheus metrics

**Data flow:** Amazon Electronics reviews → filtering and chunking → embeddings in Qdrant → evidence-grounded explanations.

---

## Limitations

| Constraint | Behavior |
|------------|----------|
| Insufficient evidence | Refuses to explain |
| Low relevance | Refuses to explain |
| Single category (Electronics) | Architecture supports broader coverage; corpus is narrower |
| No image features | Text-only retrieval |
| English only | Embedding setup is English-first |
| Cache dependency | Uncached cold-path latency (~5,400 ms) does not meet the < 500 ms SLO |

---

## Quick Start

```bash
git clone https://github.com/vxa8502/sage-recommendations && cd sage-recommendations
cp .env.example .env   # Add QDRANT_URL, QDRANT_API_KEY, and ANTHROPIC_API_KEY
```

**Hosted Qdrant:** Create a Qdrant Cloud cluster first and copy its endpoint + API key into `.env`.

**Run the demo locally:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pipeline,api,anthropic]"
sage health
sage qdrant status
sage data build        # embed and index reviews into Qdrant
sage serve
```

**Full pipeline** (data staging + experiments + evaluation):
```bash
sage stage data all                  # index corpus and build query bank
sage stage experiments all           # retrieval and gate calibration
sage stage experiments finalize \
  --decision baseline-retained \
  --retrieval-decision baseline-retained \
  --with-boundary                    # freeze explanation cases
sage eval run                        # canonical evaluation
sage eval summary                    # print saved results
```

**CLI:** `sage --help` · `make help`

**Clean restart:** `sage reset artifacts` clears evaluation outputs. `sage reset stage0` returns `data/` to the scaffold state (add `--dry-run` first for a preview).

---

## Evaluation

```bash
sage eval dev        # sampled dev-lane run (fast)
sage eval run        # full canonical evaluation
sage eval summary    # print the latest saved snapshot
sage eval boundary   # refusal / clarification guardrail benchmark
```

The repo does not ship a pre-built metric snapshot. The first `sage eval run` after a successful corpus indexing and calibration run becomes the baseline for the working cycle. `sage eval run` enforces a promotion gate and exits non-zero if the run is below threshold, incomplete, or sampled — preventing a dev-lane result from silently becoming the canonical headline.

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

## Project Structure

```
sage/
├── adapters/       # External integrations (Qdrant, LLM, HHEM verification)
├── api/            # FastAPI routes, middleware, Prometheus metrics
├── config/         # Settings, logging, thresholds
├── core/           # Domain models, aggregation, verification, chunking
├── services/       # Business logic (retrieval, explanation, evaluation, cache)
scripts/
├── pipeline.py                       # Data ingestion and embedding
├── evaluation.py                     # Retrieval evaluation (NDCG, Hit, Recall, MRR)
├── faithfulness.py                   # HHEM + RAGAS explanation verification
├── evaluate_boundary_behavior.py     # Refusal/clarification guardrail benchmark
├── load_test.py                      # Latency and cache behavior measurement
```

---

## License

Academic/portfolio use only. Uses Amazon Reviews 2023 dataset.
