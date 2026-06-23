.DEFAULT_GOAL := help

.PHONY: ci-fresh help

PROJECT_PYTHON := .venv/bin/python
PROJECT_SAGE := .venv/bin/sage
BOOTSTRAP_PYTHON := $(shell python3 -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))' 2>/dev/null || python -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))' 2>/dev/null || command -v python3 || command -v python)
PYTHON := $(if $(wildcard $(PROJECT_PYTHON)),$(PROJECT_PYTHON),$(BOOTSTRAP_PYTHON))
CLI := $(if $(wildcard $(PROJECT_SAGE)),$(PROJECT_SAGE),$(PYTHON) -m sage.cli)
PROJECT_CLI := $(if $(wildcard $(PROJECT_SAGE)),$(PROJECT_SAGE),$(PROJECT_PYTHON) -m sage.cli)

ci-fresh:
	rm -rf .venv
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PROJECT_PYTHON) -m pip install -e ".[dev,pipeline,api,anthropic,openai]"
	$(PROJECT_CLI) lint
	$(PROJECT_CLI) typecheck
	$(PROJECT_CLI) test
	@echo "Full CI passed (fresh venv)"

help:
	@printf "Make targets:\n"
	@printf "  make help     Show this help and the canonical CLI commands\n"
	@printf "  make ci-fresh Recreate .venv and run lint, typecheck, and tests\n"
	@printf "\n"
	@printf "Common CLI commands:\n"
	@printf "  sage health\n"
	@printf "  sage stage data check\n"
	@printf "  sage stage data all --with-candidates\n"
	@printf "  sage data build\n"
	@printf "  sage qdrant status\n"
	@printf "  sage eval run\n"
	@printf "  sage eval summary\n"
	@printf "  sage demo --query \"wireless earbuds for running\"\n"
	@printf "  sage serve --port 8000\n"
	@printf "  sage fmt | sage lint | sage typecheck | sage test | sage ci\n"
	@printf "  sage reset artifacts | sage reset experiments --dry-run\n"
	@printf "\n"
	@printf "Full CLI help:\n\n"
	@$(CLI) --help
