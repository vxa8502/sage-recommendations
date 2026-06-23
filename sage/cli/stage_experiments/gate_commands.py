from __future__ import annotations

import argparse
from pathlib import Path

from sage.data.query_bank.sources.esci._config import (
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
)

from .contracts import (
    RetrievalAggregation,
)
from .paths import (
    _gate_calibration_analysis_path,
    _gate_calibration_output_path,
    _gate_holdout_output_path,
    _query_bank_path,
)
from .prereqs import _require_stage2_prereqs
from ..shared import (
    load_dotenv_if_available,
    run_command,
)
from ..script_command import script_command
from .command_utils import (
    _parse_holdout_subset_selection,
)


def _run_gate_calibration(
    *,
    query_bank_path: str | Path | None = None,
    output: str | Path | None = None,
    analysis_path: str | Path | None = None,
    analyze_only: bool = False,
    query_limit: int | None = None,
    top_k: int | None = None,
    min_rating: float | None = None,
    aggregation: RetrievalAggregation | None = None,
    strict_retrieval: bool = False,
    max_failed_queries: int | None = None,
    max_failure_rate: float | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)

    output_path = _gate_calibration_output_path(output)
    analysis_output_path = _gate_calibration_analysis_path(analysis_path)
    command = (
        script_command("scripts/calibrate_token_threshold.py")
        .option("--query-bank-path", resolved_query_bank_path)
        .option("--output", output_path)
        .option("--analysis-output", analysis_output_path)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--min-rating", min_rating)
        .optional("--aggregation", aggregation)
        .flag("--analyze-only", analyze_only)
        .flag("--strict-retrieval", strict_retrieval)
        .optional("--max-failed-queries", max_failed_queries)
        .optional("--max-failure-rate", max_failure_rate)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_calibrate_gate(args: argparse.Namespace) -> None:
    _run_gate_calibration(
        query_bank_path=getattr(args, "query_bank_path", None),
        output=getattr(args, "output", None),
        analysis_path=getattr(args, "analysis_path", None),
        analyze_only=getattr(args, "analyze_only", False),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        min_rating=getattr(args, "min_rating", None),
        aggregation=getattr(args, "aggregation", None),
        strict_retrieval=getattr(args, "strict_retrieval", False),
        max_failed_queries=getattr(args, "max_failed_queries", None),
        max_failure_rate=getattr(args, "max_failure_rate", None),
    )


def _run_gate_holdout(
    *,
    query_bank_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
    output: str | Path | None = None,
    subsets: str | None = None,
    query_limit: int | None = None,
    top_k: int | None = None,
    min_rating: float | None = None,
    aggregation: RetrievalAggregation | None = None,
    candidate_tokens: int | None = None,
    candidate_chunks: int | None = None,
    candidate_score: float | None = None,
    strict_retrieval: bool = False,
    max_failed_queries: int | None = None,
    max_failure_rate: float | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)
    subset_selection = _parse_holdout_subset_selection(subsets)
    if DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG not in subset_selection:
        print(
            "Warning: this holdout run has no promotion-eligible subset. "
            "It will be diagnostic only."
        )
    elif "faithfulness_dev_seed" in subset_selection:
        print(
            "Note: `faithfulness_dev_seed` will be included as a diagnostic-only "
            "subset alongside the promotion holdout."
        )

    resolved_analysis_path = _gate_calibration_analysis_path(analysis_path)
    output_path = _gate_holdout_output_path(output)
    command = (
        script_command("scripts/evaluate_evidence_gate_holdout.py")
        .option("--query-bank-path", resolved_query_bank_path)
        .option("--analysis-path", resolved_analysis_path)
        .option("--output", output_path)
        .optional("--subsets", subsets)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--min-rating", min_rating)
        .optional("--aggregation", aggregation)
        .optional("--candidate-tokens", candidate_tokens)
        .optional("--candidate-chunks", candidate_chunks)
        .optional("--candidate-score", candidate_score)
        .flag("--strict-retrieval", strict_retrieval)
        .optional("--max-failed-queries", max_failed_queries)
        .optional("--max-failure-rate", max_failure_rate)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_holdout_gate(args: argparse.Namespace) -> None:
    _run_gate_holdout(
        query_bank_path=getattr(args, "query_bank_path", None),
        analysis_path=getattr(args, "analysis_path", None),
        output=getattr(args, "output", None),
        subsets=getattr(args, "subsets", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        min_rating=getattr(args, "min_rating", None),
        aggregation=getattr(args, "aggregation", None),
        candidate_tokens=getattr(args, "candidate_tokens", None),
        candidate_chunks=getattr(args, "candidate_chunks", None),
        candidate_score=getattr(args, "candidate_score", None),
        strict_retrieval=getattr(args, "strict_retrieval", False),
        max_failed_queries=getattr(args, "max_failed_queries", None),
        max_failure_rate=getattr(args, "max_failure_rate", None),
    )
