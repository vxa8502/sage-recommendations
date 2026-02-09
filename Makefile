.PHONY: all setup data data-validate eval eval-deep eval-quick demo demo-interview reset reset-hard check-env qdrant-up qdrant-down qdrant-status eda serve serve-dev docker-build docker-run deploy-info human-eval-generate human-eval human-eval-analyze test lint typecheck ci info summary metrics-snapshot health help

# ---------------------------------------------------------------------------
# Configurable Variables (override: make demo QUERY="gaming mouse")
# ---------------------------------------------------------------------------

QUERY ?= wireless headphones with noise cancellation
TOP_K ?= 1
SAMPLES ?= 10
SEED ?= 42
PORT ?= 8000

# ---------------------------------------------------------------------------
# Environment Check
# ---------------------------------------------------------------------------

check-env:
	@echo "Checking environment..."
	@python -c "\
	import os; from dotenv import load_dotenv; load_dotenv(); \
	a = os.getenv('ANTHROPIC_API_KEY', ''); o = os.getenv('OPENAI_API_KEY', ''); \
	exit(0) if (a or o) else exit(1)" || \
		(echo "ERROR: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set (checked shell + .env)" && exit 1)
	@python -c "\
	from sage.adapters.vector_store import get_client; \
	c = get_client(); c.get_collections(); print('Qdrant OK')" 2>/dev/null || \
		(echo "ERROR: Cannot connect to Qdrant. Check QDRANT_URL in .env or run 'make qdrant-up' for local." && exit 1)
	@echo "Environment OK"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:
	@echo "=== SETUP ==="
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[pipeline,api,anthropic,openai]"
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

# Download, filter, chunk, embed, index to Qdrant
data: check-env
	@echo "=== DATA PIPELINE ==="
	python scripts/pipeline.py
	@echo "Verifying outputs..."
	@test -d data/splits || (echo "FAIL: data/splits/ not created" && exit 1)
	@test -f data/splits/train.parquet || (echo "FAIL: train.parquet not created" && exit 1)
	@echo "Data pipeline complete"

# Validate data outputs exist and have expected structure
data-validate:
	@echo "Validating data outputs..."
	@test -f data/splits/train.parquet || (echo "FAIL: train.parquet missing" && exit 1)
	@test -f data/splits/test.parquet || (echo "FAIL: test.parquet missing" && exit 1)
	@python -c "\
	import pandas as pd; import numpy as np; from pathlib import Path; \
	t = pd.read_parquet('data/splits/train.parquet'); \
	e = list(Path('data').glob('embeddings_*.npy')); \
	emb = np.load(e[0]) if e else None; \
	print(f'Train: {len(t):,} rows, {t.parent_asin.nunique():,} products'); \
	print(f'Embeddings: {emb.shape if emb is not None else \"not found\"}'); \
	assert len(t) > 1000, 'Train set too small'; \
	assert emb is not None and emb.shape[1] == 384, 'Embedding dimension mismatch'; \
	print('Validation passed')"

# Exploratory data analysis (generates figures + report)
eda:
	@echo "=== EDA ANALYSIS ==="
	@mkdir -p data/figures
	@mkdir -p reports
	python scripts/eda.py
	@echo "Figures saved to data/figures/"
	@echo "Report generated: reports/eda_report.md"

# ---------------------------------------------------------------------------
# Evaluation Suite
# ---------------------------------------------------------------------------

# Standard evaluation: primary metrics, spot-checks, explanation tests, faithfulness
eval: check-env
	@test -d data/splits || (echo "ERROR: Run 'make data' first" && exit 1)
	@echo "=== EVALUATION SUITE ===" && \
	echo "" && \
	echo "--- Building evaluation datasets ---" && \
	python scripts/build_eval_dataset.py && \
	python scripts/build_natural_eval_dataset.py && \
	echo "" && \
	echo "--- Recommendation evaluation (LOO history) ---" && \
	python scripts/evaluation.py --dataset eval_loo_history.json --section primary && \
	echo "" && \
	echo "--- Recommendation evaluation (natural queries) ---" && \
	python scripts/evaluation.py --dataset eval_natural_queries.json --section primary && \
	echo "" && \
	echo "--- Explanation tests ---" && \
	python scripts/explanation.py --section basic && \
	python scripts/explanation.py --section gate && \
	python scripts/explanation.py --section verify && \
	python scripts/explanation.py --section cold && \
	echo "" && \
	echo "--- Faithfulness evaluation (HHEM + RAGAS) ---" && \
	python scripts/faithfulness.py --samples $(SAMPLES) --ragas && \
	echo "" && \
	echo "--- Sanity checks (spot) ---" && \
	python scripts/sanity_checks.py --section spot && \
	echo "" && \
	echo "=== EVALUATION COMPLETE ==="

# Deep evaluation: all ablations, baselines, calibration, failure analysis
eval-deep: check-env
	@test -d data/eval || (echo "ERROR: Run 'make eval' first to build eval datasets" && exit 1)
	@echo "=== DEEP EVALUATION (ablations + baselines) ===" && \
	echo "" && \
	echo "--- Full recommendation evaluation (LOO history) ---" && \
	python scripts/evaluation.py --dataset eval_loo_history.json --section all --baselines && \
	echo "" && \
	echo "--- Full recommendation evaluation (natural queries) ---" && \
	python scripts/evaluation.py --dataset eval_natural_queries.json --section all && \
	echo "" && \
	echo "--- All sanity checks (incl. calibration) ---" && \
	python scripts/sanity_checks.py --section all && \
	echo "" && \
	echo "--- Faithfulness failure analysis ---" && \
	python scripts/faithfulness.py --analyze && \
	python scripts/faithfulness.py --adjusted && \
	echo "" && \
	echo "=== DEEP EVALUATION COMPLETE ==="

# Quick eval: skip RAGAS (faster iteration)
eval-quick: check-env
	@test -d data/splits || (echo "ERROR: Run 'make data' first" && exit 1)
	@echo "=== QUICK EVALUATION (no RAGAS) ==="
	python scripts/build_eval_dataset.py && \
	python scripts/build_natural_eval_dataset.py && \
	python scripts/evaluation.py --dataset eval_loo_history.json --section primary && \
	python scripts/faithfulness.py --samples 5
	@echo "Quick eval complete"

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# Interactive recommendation with explanation
demo: check-env
	@echo "=== DEMO ==="
	python scripts/demo.py --query "$(QUERY)" --top-k $(TOP_K)

# Interview demo: 3 queries showcasing cache hit
demo-interview: check-env
	@echo "=== SAGE INTERVIEW DEMO ==="
	@echo ""
	@echo "--- Query 1: Basic ---"
	python scripts/demo.py --query "wireless earbuds for running" --top-k 1
	@echo ""
	@echo "--- Query 2: Complex (retrieval depth) ---"
	python scripts/demo.py --query "noise cancelling headphones for office with long battery" --top-k 1
	@echo ""
	@echo "--- Query 3: Cache Hit (same as Query 1) ---"
	python scripts/demo.py --query "wireless earbuds for running" --top-k 1
	@echo ""
	@echo "=== Demo Complete ==="

# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

all: qdrant-up data eval demo
	@python scripts/summary.py

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

serve: check-env
	@echo "=== SAGE API ==="
	python -m sage.api.run

serve-dev: check-env
	@echo "=== SAGE API (dev) ==="
	uvicorn sage.api.app:create_app --factory --reload --port $${PORT:-8000}

docker-build:
	docker build -t sage:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env -e PORT=8000 sage:latest

deploy-info:
	@echo "DEPLOY TO RENDER:"
	@echo "  1. Push to GitHub"
	@echo "  2. Connect repo at https://dashboard.render.com"
	@echo "  3. Set env vars: QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY"
	@echo "  4. Render auto-detects render.yaml and deploys"

# ---------------------------------------------------------------------------
# Human Evaluation
# ---------------------------------------------------------------------------

human-eval-generate: check-env
	@echo "=== GENERATING HUMAN EVAL SAMPLES ==="
	python scripts/human_eval.py --generate --seed $(SEED)

human-eval: check-env
	@echo "=== HUMAN EVALUATION ==="
	python scripts/human_eval.py --annotate

human-eval-analyze:
	@echo "=== HUMAN EVAL ANALYSIS ==="
	python scripts/human_eval.py --analyze

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

lint:
	ruff check sage/ scripts/ tests/
	ruff format --check sage/ scripts/ tests/

typecheck:
	mypy sage/ --ignore-missing-imports

test:
	python -m pytest tests/ -v

ci: lint typecheck test
	@echo "All CI checks passed"

# ---------------------------------------------------------------------------
# Info & Metrics
# ---------------------------------------------------------------------------

info:
	@python -c "\
	import sys; from sage.config import EMBEDDING_MODEL, QDRANT_URL, LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL; \
	print('Sage v0.1.0'); \
	print(f'Python: {sys.version_info.major}.{sys.version_info.minor}'); \
	print(f'Embedding: {EMBEDDING_MODEL}'); \
	print(f'Qdrant: {QDRANT_URL}'); \
	print(f'LLM: {LLM_PROVIDER} ({ANTHROPIC_MODEL if LLM_PROVIDER == \"anthropic\" else OPENAI_MODEL})')"

summary:
	@python scripts/summary.py

metrics-snapshot:
	@python -c "\
	import json; from pathlib import Path; \
	r = Path('data/eval_results'); \
	loo = json.load(open(r/'eval_loo_history_latest.json', encoding='utf-8')) if (r/'eval_loo_history_latest.json').exists() else {}; \
	faith = json.load(open(r/'faithfulness_latest.json', encoding='utf-8')) if (r/'faithfulness_latest.json').exists() else {}; \
	human = json.load(open(r/'human_eval_latest.json', encoding='utf-8')) if (r/'human_eval_latest.json').exists() else {}; \
	pm = loo.get('primary_metrics', {}); mm = faith.get('multi_metric', {}); \
	print('=== SAGE METRICS ==='); \
	print(f'NDCG@10:     {pm.get(\"ndcg_at_10\", \"n/a\")}'); \
	print(f'Claim HHEM:  {mm.get(\"claim_level_avg_score\", \"n/a\")}'); \
	print(f'Quote Verif: {mm.get(\"quote_verification_rate\", \"n/a\")}'); \
	print(f'Human Eval:  {human.get(\"overall_helpfulness\", \"n/a\")}/5.0 (n={human.get(\"n_samples\", 0)})')"

health:
	@curl -sf http://localhost:$(PORT)/health | python -m json.tool 2>/dev/null || \
		echo "API not running at localhost:$(PORT). Start with: make serve"

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

# Clear processed data, keep raw download cache and Qdrant Cloud data
reset:
	@echo "Clearing processed data..."
	rm -f data/reviews_prepared_*.parquet
	rm -f data/embeddings_*.npy
	rm -rf data/splits/
	rm -rf data/eval/
	rm -f data/eval_results/eval_*.json
	rm -f data/eval_results/faithfulness_*.json
	@echo "  (human_eval_*.json preserved — use rm -rf data/eval_results/ to clear)"
	rm -rf data/figures/
	rm -f reports/eda_report.md
	@echo "Done. (Raw cache + Qdrant preserved — use 'make reset-hard' to clear all)"

# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

kaggle-test:
	@echo "=== TESTING KAGGLE PIPELINE (local, 100K) ==="
	python scripts/kaggle_pipeline.py

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

# Hard reset: remove EVERYTHING (ground zero for fresh start)
reset-hard: reset
	@echo "Clearing Qdrant collection..."
	@python -c "\
	from sage.adapters.vector_store import get_client; \
	c = get_client(); c.delete_collection('sage_reviews'); \
	print('  Collection deleted')" 2>/dev/null || \
		echo "  Qdrant not reachable, skipping collection cleanup"
	@echo "Removing raw download cache..."
	rm -f data/reviews_[0-9]*.parquet
	rm -f data/reviews_full.parquet
	rm -rf data/qdrant_storage/
	@echo "Removing human eval data..."
	rm -rf data/human_eval/
	rm -f data/eval_results/human_eval_*.json
	@echo "Removing e2e success results..."
	rm -f data/eval_results/e2e_success_*.json
	@echo "Removing any remaining eval results..."
	rm -rf data/eval_results/
	@echo "Hard reset complete. Project at ground zero."

# ---------------------------------------------------------------------------
# Qdrant Management
# ---------------------------------------------------------------------------

qdrant-up:
	@echo "Starting Qdrant..."
	@docker info > /dev/null 2>&1 || \
		(echo "ERROR: Docker is not running. Start Docker Desktop first." && exit 1)
	@docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
		-v "$$(pwd)/data/qdrant_storage:/qdrant/storage" \
		qdrant/qdrant:latest 2>/dev/null || \
		docker start qdrant 2>/dev/null || true
	@echo "Waiting for Qdrant..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		python -c "from sage.adapters.vector_store import get_client; get_client().get_collections()" 2>/dev/null && break; \
		sleep 1; \
	done
	@python -c "\
	from sage.adapters.vector_store import get_client; from sage.config import QDRANT_URL; \
	get_client().get_collections(); print(f'Qdrant running at {QDRANT_URL}')" 2>/dev/null || \
		(echo "ERROR: Qdrant failed to start within 10 seconds" && exit 1)

qdrant-down:
	@echo "Stopping Qdrant..."
	@docker stop qdrant 2>/dev/null || true
	@docker rm qdrant 2>/dev/null || true
	@echo "Qdrant stopped"

qdrant-status:
	@python -c "\
	from sage.adapters.vector_store import get_client, get_collection_info; \
	c = get_client(); info = get_collection_info(c); \
	[print(f'  {k}: {v}') for k, v in info.items()]" 2>/dev/null || \
		echo "Qdrant not reachable"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo "Sage - RAG Recommendation System"
	@echo ""
	@echo "QUICK START:"
	@echo "  make setup         Create venv and install dependencies"
	@echo "  make data          Load, chunk, embed, and index reviews"
	@echo "  make demo          Run demo query (customizable: QUERY, TOP_K)"
	@echo "  make all           Full pipeline (data + eval + demo + summary)"
	@echo ""
	@echo "DEMO:"
	@echo "  make demo                      Single recommendation with explanation"
	@echo "  make demo QUERY=\"gaming mouse\" Custom query"
	@echo "  make demo-interview            3-query showcase (includes cache hit)"
	@echo ""
	@echo "INFO & METRICS:"
	@echo "  make info            Show version, models, and URLs"
	@echo "  make summary         Print evaluation summary"
	@echo "  make metrics-snapshot Quick metrics display"
	@echo "  make health          Check API health (requires running server)"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make data            Load, chunk, embed, and index reviews"
	@echo "  make data-validate   Validate data outputs"
	@echo "  make eda             Exploratory data analysis (generates figures)"
	@echo "  make eval            Standard evaluation (SAMPLES=10 default)"
	@echo "  make eval-deep       Deep evaluation (all ablations + baselines)"
	@echo "  make eval-quick      Quick eval (skip RAGAS)"
	@echo ""
	@echo "API:"
	@echo "  make serve           Start API server (PORT=8000)"
	@echo "  make serve-dev       Start API with auto-reload"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo "  make deploy-info     Show Render deployment instructions"
	@echo ""
	@echo "HUMAN EVALUATION:"
	@echo "  make human-eval-generate  Generate 50 eval samples (SEED=42)"
	@echo "  make human-eval           Rate samples interactively"
	@echo "  make human-eval-analyze   Compute results from ratings"
	@echo ""
	@echo "QUALITY:"
	@echo "  make lint            Run ruff linter and formatter check"
	@echo "  make typecheck       Run mypy type checking"
	@echo "  make test            Run unit tests"
	@echo "  make ci              Run all CI checks (lint + typecheck + test)"
	@echo ""
	@echo "QDRANT:"
	@echo "  make qdrant-up       Start Qdrant vector database (Docker)"
	@echo "  make qdrant-down     Stop Qdrant"
	@echo "  make qdrant-status   Check Qdrant status"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make reset           Clear local generated data (preserves Qdrant)"
	@echo "  make reset-hard      Reset + clear Qdrant + raw data cache"
	@echo ""
	@echo "VARIABLES:"
	@echo "  QUERY    Demo query (default: wireless headphones...)"
	@echo "  TOP_K    Number of results (default: 1)"
	@echo "  SAMPLES  Faithfulness eval samples (default: 10)"
	@echo "  SEED     Random seed for human eval (default: 42)"
	@echo "  PORT     API port (default: 8000)"
