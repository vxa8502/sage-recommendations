from __future__ import annotations

import argparse
from pathlib import Path


from .contracts import (
    DEFAULT_RETRIEVAL_AGGREGATION,
    RetrievalAggregation,
)
from .paths import (
    _query_bank_path,
)
from .case_paths import (
    _resolve_bundle_paths,
    _resolve_materialization_paths,
)
from .prereqs import _require_stage2_prereqs
from ..shared import (
    ensure_llm_credentials,
    load_dotenv_if_available,
    run_command,
)
from ..script_command import script_command


def _freeze_bundles(
    *,
    query_bank_path: str | Path | None = None,
    surface: str = "dev",
    subset_tag: str | None = None,
    output: str | Path | None = None,
    outcomes_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    query_limit: int | None = None,
    top_k: int | None = None,
    min_rating: float | None = None,
    profile_label: str | None = None,
    aggregation: RetrievalAggregation = DEFAULT_RETRIEVAL_AGGREGATION,
    reference_timestamp_ms: int | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)
    resolved_paths = _resolve_bundle_paths(
        surface=surface,
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
    )

    command = (
        script_command("scripts/freeze_faithfulness_seed_bundles.py")
        .option("--surface", surface)
        .option("--query-bank-path", resolved_query_bank_path)
        .option("--output", resolved_paths["bundles_path"])
        .option("--outcomes-output", resolved_paths["outcomes_path"])
        .option("--manifest-output", resolved_paths["manifest_path"])
        .optional("--subset-tag", subset_tag)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--min-rating", min_rating)
        .optional("--profile-label", profile_label)
        .optional("--aggregation", aggregation)
        .optional("--reference-timestamp-ms", reference_timestamp_ms)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_freeze_bundles(args: argparse.Namespace) -> None:
    _freeze_bundles(
        query_bank_path=getattr(args, "query_bank_path", None),
        surface=getattr(args, "surface", "dev"),
        subset_tag=getattr(args, "subset_tag", None),
        output=getattr(args, "output", None),
        outcomes_output=getattr(args, "outcomes_output", None),
        manifest_output=getattr(args, "manifest_output", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        min_rating=getattr(args, "min_rating", None),
        profile_label=getattr(args, "profile_label", None),
        aggregation=getattr(
            args,
            "aggregation",
            DEFAULT_RETRIEVAL_AGGREGATION,
        ),
        reference_timestamp_ms=getattr(args, "reference_timestamp_ms", None),
    )


def _materialize_cases(
    *,
    surface: str = "dev",
    bundles_path: str | Path | None = None,
    bundle_outcomes_path: str | Path | None = None,
    bundles_manifest_path: str | Path | None = None,
    output: str | Path | None = None,
    outcomes_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    gate_min_chunks: int | None = None,
    gate_min_tokens: int | None = None,
    gate_min_score: float | None = None,
    profile_label: str | None = None,
    min_rating: float | None = None,
    aggregation: RetrievalAggregation = DEFAULT_RETRIEVAL_AGGREGATION,
) -> None:
    load_dotenv_if_available()
    resolved_bundle_paths = _resolve_bundle_paths(
        surface=surface,
        output=bundles_path,
        outcomes_output=bundle_outcomes_path,
        manifest_output=bundles_manifest_path,
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
    )
    resolved_case_paths = _resolve_materialization_paths(
        surface=surface,
        output=output,
        outcomes_output=outcomes_output,
        manifest_output=manifest_output,
        profile_label=profile_label,
        min_rating=min_rating,
        aggregation=aggregation,
    )

    command = (
        script_command("scripts/materialize_faithfulness_cases.py")
        .option("--surface", surface)
        .option("--bundles-path", resolved_bundle_paths["bundles_path"])
        .option("--bundle-outcomes-path", resolved_bundle_paths["outcomes_path"])
        .option("--bundles-manifest-path", resolved_bundle_paths["manifest_path"])
        .option("--output", resolved_case_paths["cases_path"])
        .option("--outcomes-output", resolved_case_paths["outcomes_path"])
        .option("--manifest-output", resolved_case_paths["manifest_path"])
        .optional("--gate-min-chunks", gate_min_chunks)
        .optional("--gate-min-tokens", gate_min_tokens)
        .optional("--gate-min-score", gate_min_score)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_materialize_cases(args: argparse.Namespace) -> None:
    _materialize_cases(
        surface=getattr(args, "surface", "dev"),
        bundles_path=getattr(args, "bundles_path", None),
        bundle_outcomes_path=getattr(args, "bundle_outcomes_path", None),
        bundles_manifest_path=getattr(args, "bundles_manifest_path", None),
        output=getattr(args, "output", None),
        outcomes_output=getattr(args, "outcomes_output", None),
        manifest_output=getattr(args, "manifest_output", None),
        gate_min_chunks=getattr(args, "gate_min_chunks", None),
        gate_min_tokens=getattr(args, "gate_min_tokens", None),
        gate_min_score=getattr(args, "gate_min_score", None),
        profile_label=getattr(args, "profile_label", None),
        min_rating=getattr(args, "min_rating", None),
        aggregation=getattr(
            args,
            "aggregation",
            DEFAULT_RETRIEVAL_AGGREGATION,
        ),
    )


def _run_boundary(
    *,
    query_bank_path: str | Path | None = None,
    subset_tag: str | None = None,
    query_limit: int | None = None,
    top_k: int | None = None,
    min_rating: float | None = None,
    aggregation: RetrievalAggregation | None = None,
    max_evidence: int | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)
    ensure_llm_credentials()

    command = (
        script_command("scripts/evaluate_boundary_behavior.py")
        .option("--query-bank-path", resolved_query_bank_path)
        .optional("--subset-tag", subset_tag)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--min-rating", min_rating)
        .optional("--aggregation", aggregation)
        .optional("--max-evidence", max_evidence)
        .option("--artifact-scope", "dev" if query_limit is not None else "canonical")
        .to_list()
    )
    run_command(command)


def command_stage_experiments_boundary(args: argparse.Namespace) -> None:
    _run_boundary(
        query_bank_path=getattr(args, "query_bank_path", None),
        subset_tag=getattr(args, "subset_tag", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        min_rating=getattr(args, "min_rating", None),
        aggregation=getattr(args, "aggregation", None),
        max_evidence=getattr(args, "max_evidence", None),
    )
