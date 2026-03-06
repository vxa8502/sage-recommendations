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

A recommendation system that refuses to hallucinate. Every claim is a verified quote from real customer reviews. When evidence is sparse, it refuses rather than guesses.

```json
{
  "query": "budget bluetooth headphones",
  "recommendations": [{
    "explanation": "Reviewers say \"For $18 Bluetooth headphones there is no better pair\" [review_141313]...",
    "confidence": {"hhem_score": 0.78, "is_grounded": true},
    "citations_verified": true
  }]
}
```

**Try it:** [vxa8502-sage.hf.space](https://vxa8502-sage.hf.space)

---

## Results

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| NDCG@10 | > 0.30 | **0.487** [0.37, 0.60] | 95% CI via bootstrap (n=42 queries) |
| Faithfulness (claim-level HHEM) | > 0.85 | **0.968** | 96.8% of individual quoted claims verified |
| Faithfulness (full-explanation HHEM) | - | 0.20 | 20% of full explanations pass; stricter but penalizes refusals |
| Faithfulness (RAGAS) | - | 0.50 | Penalizes citation-heavy style and graceful refusals |
| Human evaluation | > 3.5/5 | **3.6**/5 | Single-rater; no inter-rater reliability |
| P99 latency | < 500ms | **283ms** | Production load test |

**Faithfulness metrics:** Three measurements capture different aspects. Claim-level HHEM (96.8%) validates each quoted claim individually - the primary metric since Sage uses explicit citations. Full-explanation HHEM (20%) and RAGAS (0.50) score entire responses holistically, penalizing graceful refusals as "failures." All three reported for transparency.

**Human evaluation limitation:** Single-rater (developer). Usefulness (3.06) and satisfaction (3.04) have high variance (std ~1.7), suggesting inconsistent quality across query types.

---

## Architecture

```
User Query: "wireless earbuds for running"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      SAGE API (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│  1. EMBED         │  E5-small (384-dim)           ~20ms    │
│  2. CACHE CHECK   │  Exact + semantic (0.92 sim)  ~1ms     │
│  3. RETRIEVE      │  Qdrant vector search         ~50ms    │
│  4. AGGREGATE     │  Chunk → Product (MAX score)  ~1ms     │
│  5. EXPLAIN       │  Claude/GPT + evidence        ~300ms   │
│  6. VERIFY        │  HHEM hallucination check     ~50ms    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Response:                                                  │
│  - Product ID + score                                       │
│  - Explanation with [citations]                             │
│  - HHEM confidence score                                    │
│  - Quote verification results                               │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:** Amazon Electronics reviews (1M raw) → 5-core filter → 334K reviews → semantic chunking → 423K chunks in Qdrant. *([pipeline.py](scripts/pipeline.py) | [Kaggle notebook](scripts/kaggle_pipeline.ipynb))*

---

## Limitations

| Constraint | Behavior |
|------------|----------|
| Insufficient evidence (< 2 chunks) | Refuses to explain |
| Low relevance (top score < 0.7) | Refuses to explain |
| Single category (Electronics) | Architecture supports multi-category; data constraint only |
| No image features | Text-only retrieval |
| English only | E5 primarily English-trained |
| Cold start | First request ~10s (HF wake), then P99 < 500ms |

---

## Quick Start

```bash
git clone https://github.com/vxa8502/sage-recommendations && cd sage-recommendations
cp .env.example .env   # Then add ANTHROPIC_API_KEY or OPENAI_API_KEY
```

**Docker:** `docker compose up`

**Local:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pipeline,api,anthropic]"
make qdrant-up && make data && make serve
```

---

## API Reference

### POST /recommend

```bash
curl -X POST https://vxa8502-sage.hf.space/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless earbuds for running", "k": 3, "explain": true}'
```

Returns products with grounded explanations, HHEM confidence scores, and verified citations.

### POST /recommend/stream

Server-sent events for token-by-token explanation streaming.

### GET /health, /metrics, /cache/stats

Health check, Prometheus metrics, and cache statistics.

---

## Evaluation

```bash
make eval          # ~5 min: standard pre-commit
make eval-full     # ~17 min: complete suite + load test
```

---

## Project Structure (Key Directories)

```
sage/
├── adapters/       # External integrations (Qdrant, LLM, HHEM)
├── api/            # FastAPI routes, middleware, Prometheus metrics
├── config/         # Settings, logging, query templates
├── core/           # Domain models, aggregation, verification, chunking
├── services/       # Business logic (retrieval, explanation, cache)
scripts/
├── pipeline.py     # Data ingestion and embedding
├── evaluation.py   # NDCG, precision, recall, novelty, baselines
├── faithfulness.py # HHEM, RAGAS, grounding delta
├── human_eval.py   # Interactive human evaluation
├── load_test.py    # P99 latency benchmarking
```

---

## License

Academic/portfolio use only. Uses Amazon Reviews 2023 dataset.
