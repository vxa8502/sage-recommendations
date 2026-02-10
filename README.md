---
title: Sage
emoji: 🌿
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
---
<!-- HF Spaces metadata above; hidden on HF, visible on GitHub -->

# Sage

**Product recommendations without explanations are black boxes.** Users see "You might like X" but never learn *why*. This system retrieves products via semantic search over real customer reviews, then generates natural language explanations grounded in that evidence. Every claim is verified against source text using hallucination detection.

**Live demo:** [vxa8502-sage.hf.space](https://vxa8502-sage.hf.space)

---

## Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| NDCG@10 (recommendation quality) | > 0.30 | 0.295 | 98% |
| Claim-level faithfulness (HHEM) | > 0.85 | 0.968 | Pass |
| Human evaluation (n=50) | > 3.5/5 | 4.43/5 | Pass |
| P99 latency (retrieval) | < 500ms | 283ms | Pass |
| P99 latency (cache hit) | < 100ms | ~80ms | Pass |

**Grounding impact:** Explanations generated WITH evidence score 69% on HHEM. WITHOUT evidence: 3%. RAG grounding reduces hallucination by 66 percentage points.

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

**Data flow:** 1M Amazon reviews → 5-core filter → 30K reviews → semantic chunking → 423K chunks in Qdrant.

---

## Design Trade-offs

| Decision | Alternative | Why This Choice |
|----------|-------------|-----------------|
| **E5-small** (384-dim) | E5-large, BGE-large | 3x faster inference, 0.02 NDCG delta. Latency > marginal accuracy. |
| **Qdrant** | Pinecone, Weaviate | Free cloud tier (1GB), gRPC, native Python client. |
| **Semantic chunking** | Fixed-window | Preserves complete arguments; +12% quote verification rate. |
| **MAX aggregation** | MEAN, weighted | Best single chunk matters more than average for explanations. |
| **HHEM** (Vectara) | NLI models, GPT-4 judge | Purpose-built for RAG; no API cost; 0.97 AUC on HaluEval. |
| **Claim-level HHEM** | Full-explanation HHEM | Isolates hallucinated claims; more actionable than binary pass/fail. |
| **Quality gate** (refuse) | Always answer | Reduces hallucination; 46% refusal rate is a feature, not a bug. |

See [`docs/chunking_decisions.md`](docs/chunking_decisions.md) for detailed chunking rationale.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Single category** (Electronics) | Can't recommend across categories | Architecture supports multi-category; data constraint only |
| **No image features** | Misses visual product attributes | Could add CLIP embeddings in future |
| **English only** | Non-English reviews have lower retrieval quality | E5 is primarily English-trained |
| **Cache invalidation manual** | Stale explanations possible | TTL-based expiry (1 hour); manual `/cache/clear` |
| **LLM latency on free tier** | P99 ~4s with explanations | Retrieval alone is 283ms; cache hits are ~80ms |
| **No user personalization** | Same results for all users | Would need user history for collaborative filtering |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/yourusername/sage
cd sage
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

docker-compose up
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
{"size": 42, "hit_rate": 0.35, "exact_hits": 10, "semantic_hits": 5, "misses": 27}
```

---

## Evaluation

```bash
make eval-quick    # ~1 min: NDCG + HHEM only
make eval          # ~5 min: standard pre-commit
make eval-all      # ~15 min: complete reproducible suite
make load-test     # P99 latency against production
```

See `make help` for all targets.

---

## Project Structure

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
| Low relevance (top score < 0.5) | Refuses to explain |
| Quote not found in evidence | Falls back to paraphrased claims |
| HHEM score < 0.5 | Flags as uncertain |

The system refuses to hallucinate rather than confidently stating unsupported claims.

---

## License

Academic/portfolio use only. Uses Amazon Reviews 2023 dataset.
