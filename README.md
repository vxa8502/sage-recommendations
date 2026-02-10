---
title: Sage
emoji: 🦉
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---
<!-- HF Spaces metadata above; hidden on HF, visible on GitHub -->

# Sage

A recommendation system that refuses to hallucinate.

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

**Try it:** [vxa8502-sage.hf.space](https://vxa8502-sage.hf.space) (API explorer with Swagger UI)

---

## The Problem

Product recommendations without explanations are black boxes. Users see "You might like X" but never learn *why*. When you ask an LLM to explain, it confidently invents features and fabricates reviews.

**Sage is different:** Every claim is a verified quote from real customer reviews. When evidence is sparse, it refuses rather than guesses. Human evaluation scored trust at **4.0/5** because honesty beats confident fabrication.

---

## Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| NDCG@10 (recommendation quality) | > 0.30 | 0.487 | Pass |
| Claim-level faithfulness (HHEM) | > 0.85 | 0.968 | Pass |
| Human evaluation (n=50) | > 3.5/5 | 3.6/5 | Pass |
| P99 latency (production) | < 500ms | 283ms | Pass |
| Median latency (cached) | < 100ms | 88ms | Pass |

**Grounding impact:** Explanations generated WITH evidence score 73% on HHEM. WITHOUT evidence: 2.6%. RAG grounding reduces hallucination by 70 percentage points.

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

**Data flow:** 1M Amazon reviews → 5-core filter → 334K reviews → semantic chunking → 423K chunks in Qdrant. *([pipeline.py](scripts/pipeline.py) | [Kaggle notebook](scripts/kaggle_pipeline.ipynb))*

---

## Why This Architecture?

The key insight: **hallucination happens when evidence is weak, not when the model is bad.**

When you give an LLM one short review as context, it fills in the gaps with plausible-sounding fabrications. The solution is refusing to explain when evidence is insufficient.

| Decision | Alternative | Why This Choice |
|----------|-------------|-----------------|
| **E5-small** (384-dim) | E5-large, BGE-large | Faster inference, same accuracy on product reviews. Latency > marginal gains. |
| **Qdrant** | Pinecone, Weaviate | Free cloud tier, payload filtering, clean Python SDK. |
| **Semantic chunking** | Fixed-window | Preserves complete arguments; better quote verification. |
| **HHEM** (Vectara) | GPT-4 judge, NLI models | Purpose-built for RAG hallucination; no API cost. |
| **Claim-level evaluation** | Full-explanation | Isolates which claims hallucinate; more actionable. |
| **Quality gate** (refuse) | Always answer | 64% refusal rate → 4.0/5 trust. Honesty > coverage. |

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Single category** (Electronics) | Can't recommend across categories | Architecture supports multi-category; data constraint only |
| **No image features** | Misses visual product attributes | Could add CLIP embeddings in future |
| **English only** | Non-English reviews have lower retrieval quality | E5 is primarily English-trained |
| **Cache invalidation manual** | Stale explanations possible | TTL-based expiry (1 hour); manual `/cache/clear` |
| **LLM latency on free tier** | P99 ~4s with explanations | Retrieval alone is 283ms; cache hits are 88ms |
| **No user personalization** | Same results for all users | Would need user history for collaborative filtering |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/vxa8502/sage-recommendations
cd sage
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

docker compose up
curl http://localhost:8000/health
```

### Local Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pipeline,api,anthropic]"

cp .env.example .env
# Edit .env: add API keys

make qdrant-up          # Start local Qdrant
make data               # Load data (or use Qdrant Cloud)
make serve              # Start API at localhost:8000
```

### Environment Variables

```bash
# Required (one of)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic   # or "openai"

# Optional: Qdrant Cloud (instead of local)
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=...
```

---

## API Reference

### POST /recommend

```bash
curl -X POST https://vxa8502-sage.hf.space/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless earbuds for running", "k": 3, "explain": true}'
```

Returns ranked products with:
- Explanation grounded in customer reviews
- HHEM confidence score (0-1)
- Quote verification results
- Evidence chunks with citations

### POST /recommend/stream

Server-sent events for token-by-token explanation streaming.

### GET /health

```json
{"status": "healthy", "qdrant_connected": true, "llm_reachable": true}
```

### GET /metrics

Prometheus metrics: `sage_request_latency_seconds`, `sage_cache_events_total`, `sage_errors_total`.

### GET /cache/stats

```json
{
  "size": 42,
  "max_entries": 1000,
  "exact_hits": 10,
  "semantic_hits": 5,
  "misses": 27,
  "evictions": 0,
  "hit_rate": 0.35,
  "ttl_seconds": 3600.0,
  "similarity_threshold": 0.92
}
```

---

## Evaluation

```bash
make eval          # ~5 min: standard pre-commit
make eval-full     # ~17 min: complete automated suite + load test
```

See `make help` for all targets (including `eval-quick`, `load-test`).

---

## Project Structure (Key Directories)

```
sage/
├── adapters/       # External integrations (Qdrant, LLM, HHEM)
├── api/            # FastAPI routes, middleware, Prometheus metrics
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

## Failure Modes (By Design)

| Condition | System Behavior |
|-----------|-----------------|
| Insufficient evidence (< 2 chunks) | Refuses to explain |
| Low relevance (top score < 0.7) | Refuses to explain |
| Quote not found in evidence | Falls back to paraphrased claims |
| HHEM score < 0.5 | Flags as uncertain |

The system refuses to hallucinate rather than confidently stating unsupported claims.

---

## License

Academic/portfolio use only. Uses Amazon Reviews 2023 dataset.
