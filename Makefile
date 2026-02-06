.PHONY: all setup data eval eval-deep eval-quick demo reset reset-hard check-env qdrant-up qdrant-down qdrant-status eda serve serve-dev docker-build docker-run deploy-info human-eval-generate human-eval human-eval-analyze test lint typecheck help

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

# Exploratory data analysis (generates figures for reports/eda_report.md)
eda:
	@echo "=== EDA ANALYSIS ==="
	@mkdir -p data/figures
	python scripts/eda.py
	@echo "Figures saved to data/figures/"
	@echo "View report: reports/eda_report.md"

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
	python scripts/faithfulness.py --samples 10 --ragas && \
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
	python scripts/demo.py --query "wireless headphones with noise cancellation" --top-k 1

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
	python scripts/human_eval.py --generate

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

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

# Clear processed data, keep raw download cache
reset:
	@echo "Clearing processed data..."
	rm -f data/reviews_prepared_*.parquet
	rm -f data/embeddings_*.npy
	rm -rf data/splits/
	rm -rf data/eval/
	rm -f data/eval_results/eval_*.json
	rm -f data/eval_results/faithfulness_*.json
	@echo "  (human_eval_*.json preserved — use rm -rf data/eval_results/ to clear)"
	rm -rf data/explanations/
	rm -rf data/figures/
	@echo "Clearing Qdrant collection..."
	@python -c "\
	from sage.adapters.vector_store import get_client; \
	c = get_client(); c.delete_collection('sage_reviews'); \
	print('  Collection deleted')" 2>/dev/null || \
		echo "  Qdrant not reachable, skipping collection cleanup"
	@echo "Done. (Raw download cache preserved — use 'make reset-hard' to clear)"

# Hard reset: also remove raw download cache (forces re-download from HuggingFace)
reset-hard: reset
	@echo "Removing raw download cache..."
	rm -f data/reviews_[0-9]*.parquet
	rm -f data/reviews_full.parquet
	rm -rf data/qdrant_storage/
	@echo "Hard reset complete."

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
		curl -sf http://localhost:6333/collections > /dev/null 2>&1 && break; \
		sleep 1; \
	done
	@curl -sf http://localhost:6333/collections > /dev/null 2>&1 && \
		echo "Qdrant running at localhost:6333" || \
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
	@echo "SETUP:"
	@echo "  make setup         Create venv and install dependencies"
	@echo "  make qdrant-up     Start Qdrant vector database (Docker)"
	@echo "  make qdrant-down   Stop Qdrant"
	@echo "  make qdrant-status Check Qdrant status"
	@echo ""
	@echo "PIPELINE:"
	@echo "  make data          Load, chunk, embed, and index reviews"
	@echo "  make eda           Exploratory data analysis (generates figures)"
	@echo "  make eval          Standard evaluation (primary metrics + RAGAS + spot-checks)"
	@echo "  make eval-deep     Deep evaluation (all ablations + baselines + calibration)"
	@echo "  make eval-quick    Quick eval (skip RAGAS)"
	@echo "  make demo          Run demo query"
	@echo "  make all           Full pipeline (data + eval + demo + summary)"
	@echo ""
	@echo "API:"
	@echo "  make serve         Start API server (port 8000)"
	@echo "  make serve-dev     Start API with auto-reload"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make deploy-info   Show Render deployment instructions"
	@echo ""
	@echo "HUMAN EVALUATION:"
	@echo "  make human-eval-generate  Generate 50 eval samples"
	@echo "  make human-eval           Rate samples interactively"
	@echo "  make human-eval-analyze   Compute results from ratings"
	@echo ""
	@echo "QUALITY:"
	@echo "  make lint          Run ruff linter and formatter check"
	@echo "  make typecheck     Run mypy type checking"
	@echo "  make test          Run unit tests"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make reset         Clear generated data and Qdrant collection"
	@echo "  make reset-hard    Reset + clear raw data cache"
	@echo ""
	@echo "PREREQUISITES:"
	@echo "  - Docker installed (for Qdrant)"
	@echo "  - ANTHROPIC_API_KEY or OPENAI_API_KEY set in .env"
	@echo "  - Python venv activated with dependencies installed"
