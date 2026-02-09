---
title: Sage
emoji: 🌿
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
---

# Sage

RAG-powered product recommendation system with explainable AI. Retrieves relevant products via semantic search over customer reviews, generates natural language explanations grounded in evidence, and verifies faithfulness using hallucination detection.

## Targets

| Metric | Target |
|--------|--------|
| Recommendation Quality (NDCG@10) | > 0.30 |
| Explanation Faithfulness (RAGAS) | > 0.85 |
| System Latency (P99) | < 500ms |
| Human Evaluation (n=50) | > 3.5/5.0 |

## Tech Stack

- **Embeddings:** E5-small (384-dim)
- **Vector DB:** Qdrant with semantic caching
- **LLM:** Claude Sonnet / GPT-4o-mini
- **Faithfulness:** HHEM (Vectara hallucination detector) + quote verification
- **API:** FastAPI with async handlers and streaming support
- **Metrics:** Prometheus (latency histograms, cache hit rates, error counts)

## Quick Start

### Option 1: Docker (easiest)

```bash
git clone https://github.com/vxa8502/sage-recommendations
cd sage-recommendations
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY)

docker-compose up
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,pipeline,api,anthropic]"  # or openai

cp .env.example .env
# Edit .env: add LLM key + Qdrant (local via `make qdrant-up` or Qdrant Cloud)

make data                  # Load data and embeddings
make serve                 # Start API
```

## Environment Variables

```bash
# Required
LLM_PROVIDER=anthropic              # or "openai"
ANTHROPIC_API_KEY=your_key_here

# Optional: Qdrant Cloud (for deployment or instead of local)
# QDRANT_URL=https://your-cluster.cloud.qdrant.io
# QDRANT_API_KEY=your_qdrant_key
```

## API Reference

### POST /recommend

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless earbuds for running", "k": 3, "explain": true}'
```

Returns ranked products with explanations grounded in customer reviews, HHEM confidence scores, and citation verification.

### POST /recommend/stream

Stream recommendations with token-by-token explanation delivery (SSE).

### GET /health

Service health check.

### GET /metrics

Prometheus metrics: latency histograms, cache hit rates, error counts.

### GET /cache/stats

Cache performance statistics.

## Failure Modes (By Design)

| Condition | System Behavior |
|-----------|-----------------|
| Insufficient evidence | Refuses to explain |
| Quote not found in source | Falls back to paraphrased claims |
| HHEM confidence below threshold | Flags explanation as uncertain |

The system refuses to hallucinate rather than confidently stating unsupported claims.

## Development

```bash
make test      # Run tests
make lint      # Run linter
make eval      # Run evaluation suite
make all       # Full pipeline
```

## Project Structure

```
sage/
├── adapters/           # External integrations (Qdrant, LLM, HHEM)
├── api/                # FastAPI routes, middleware, metrics
├── config/             # Settings, constants, queries
├── core/               # Domain models, aggregation, verification
├── services/           # Business logic (retrieval, explanation, cache)
scripts/
├── pipeline.py         # Data ingestion and embedding
├── demo.py             # Interactive demo
├── evaluation.py       # Recommendation metrics (NDCG, precision, recall)
├── faithfulness.py     # RAGAS + HHEM faithfulness evaluation
├── explanation.py      # Explanation quality tests
├── human_eval.py       # Human evaluation workflow
├── sanity_checks.py    # Spot checks and calibration
├── load_test.py        # Latency benchmarking
├── eda.py              # Exploratory data analysis
tests/
├── test_api.py
├── test_evidence.py
├── test_aggregation.py
```

## Future Work

1. **Cross-encoder reranking** for improved precision on top-k candidates
2. **User feedback loops** for learning from implicit signals
3. **Hybrid retrieval** with BM25 + dense fusion
4. **Expanded human evaluation** with stratified sampling

## License

Academic research only (uses Amazon Reviews 2023 dataset).
