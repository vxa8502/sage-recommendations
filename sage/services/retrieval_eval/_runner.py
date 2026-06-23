"""Top-level retrieval config comparison workflow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sage.config import get_logger, log_banner, log_section
from sage.data._artifact_io import write_json_object
from sage.data.query_bank import build_query_bank_identity

from ._arguments import parse_args
from ._artifacts import _build_artifact
from ._config import _current_retrieval_config, _resolve_candidate_config
from ._evaluation import _evaluate_subsets, _load_corpus_alignment
from ._types import RetrievalConfig

logger = get_logger(__name__)


def _log_run_header(
    *,
    args: Any,
    baseline_config: RetrievalConfig,
    candidate_config: RetrievalConfig,
    corpus_alignment: dict[str, Any],
) -> None:
    log_banner(logger, "EVALUATE RETRIEVAL CONFIGS")
    logger.info("Role: %s", args.comparison_role)
    logger.info("Subsets: %s", ", ".join(args.subsets))
    logger.info("Baseline: %s", baseline_config.to_dict())
    logger.info("Candidate: %s", candidate_config.to_dict())
    logger.info("Top-K: %d", args.top_k)
    logger.info("Output: %s", args.output)
    logger.info(
        "Corpus alignment OK: fingerprint=%s points=%s",
        corpus_alignment["corpus_fingerprint"],
        corpus_alignment["collection_points_count"],
    )


def _log_summary(artifact: dict[str, Any], output_path: Path) -> None:
    summary = artifact["summary"]
    recommendation = summary["recommendation"]
    log_section(logger, "Summary")
    logger.info("Combined baseline: %s", summary["baseline_metrics"])
    logger.info("Combined candidate: %s", summary["candidate_metrics"])
    logger.info("Decision status: %s", recommendation["decision_status"])
    logger.info("Recommended config: %s", recommendation["recommended_config"])
    logger.info("Artifact written: %s", output_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    baseline_config = _current_retrieval_config()
    candidate_config = _resolve_candidate_config(args, baseline=baseline_config)
    query_bank_identity = build_query_bank_identity(args.query_bank_path)
    corpus_alignment = _load_corpus_alignment()

    _log_run_header(
        args=args,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        corpus_alignment=corpus_alignment,
    )
    evaluations = _evaluate_subsets(
        subset_selection=args.subsets,
        query_bank_path=args.query_bank_path,
        query_limit=args.query_limit,
        top_k=args.top_k,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
    )
    artifact = _build_artifact(
        args=args,
        query_bank_identity=query_bank_identity,
        corpus_alignment=corpus_alignment,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        evaluations=evaluations,
    )
    write_json_object(args.output, artifact)
    _log_summary(artifact, args.output)
