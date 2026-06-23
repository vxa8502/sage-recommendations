"""Retrieval config resolution for baseline/candidate comparison runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from sage.config import RUNTIME_RETRIEVAL_AGGREGATION, RUNTIME_RETRIEVAL_MIN_RATING
from sage.data._artifact_io import load_required_json_object_file
from sage.data.faithfulness import (
    infer_retrieval_profile,
    normalize_retrieval_profile_label,
)

from ._settings import VALID_AGGREGATIONS
from ._types import RetrievalConfig


def _resolve_profile_label(
    *,
    explicit_label: str | None,
    min_rating: float | None,
    aggregation: str,
) -> str:
    if explicit_label is not None:
        return normalize_retrieval_profile_label(explicit_label)
    return infer_retrieval_profile(min_rating, aggregation=aggregation)


def _current_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        aggregation=RUNTIME_RETRIEVAL_AGGREGATION,
        min_rating=RUNTIME_RETRIEVAL_MIN_RATING,
        retrieval_profile=infer_retrieval_profile(
            RUNTIME_RETRIEVAL_MIN_RATING,
            aggregation=RUNTIME_RETRIEVAL_AGGREGATION,
        ),
    )


def _config_from_payload(
    payload: object,
    *,
    context: str,
) -> RetrievalConfig:
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: {context} must be a JSON object.")

    aggregation = payload.get("aggregation")
    min_rating = payload.get("min_rating")
    retrieval_profile = payload.get("retrieval_profile")
    if not isinstance(retrieval_profile, str) or not retrieval_profile.strip():
        retrieval_profile = payload.get("profile")

    if not isinstance(aggregation, str) or aggregation not in VALID_AGGREGATIONS:
        raise SystemExit(f"ERROR: {context} is missing a valid `aggregation` value.")
    if min_rating is not None and (
        isinstance(min_rating, bool) or not isinstance(min_rating, (int, float))
    ):
        raise SystemExit(f"ERROR: {context} has an invalid `min_rating` value.")
    if not isinstance(retrieval_profile, str) or not retrieval_profile.strip():
        retrieval_profile = infer_retrieval_profile(
            float(min_rating) if min_rating is not None else None,
            aggregation=aggregation,
        )

    return RetrievalConfig(
        aggregation=aggregation,
        min_rating=float(min_rating) if min_rating is not None else None,
        retrieval_profile=normalize_retrieval_profile_label(retrieval_profile),
    )


def _load_candidate_config_from_artifact(path: Path) -> RetrievalConfig:
    try:
        payload = load_required_json_object_file(
            path,
            description="Candidate retrieval config artifact",
            error_cls=ValueError,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(
            "ERROR: candidate retrieval config artifact is not available.\n"
            f"Artifact: {path}"
        ) from exc

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise SystemExit(
            "ERROR: candidate retrieval config artifact is missing `methodology`.\n"
            f"Artifact: {path}"
        )
    return _config_from_payload(
        methodology.get("candidate_config"),
        context=f"{path} methodology.candidate_config",
    )


def _has_explicit_candidate_overrides(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.candidate_min_rating,
            args.candidate_aggregation,
            args.candidate_profile_label,
        )
    )


def _candidate_config_source(args: argparse.Namespace) -> str:
    if args.candidate_config_path is None:
        return "explicit_cli_overrides"
    if _has_explicit_candidate_overrides(args):
        return f"{args.candidate_config_path} + explicit_cli_overrides"
    return str(args.candidate_config_path)


def _resolve_candidate_config(
    args: argparse.Namespace,
    *,
    baseline: RetrievalConfig,
) -> RetrievalConfig:
    candidate = (
        _load_candidate_config_from_artifact(args.candidate_config_path)
        if args.candidate_config_path is not None
        else baseline
    )

    if _has_explicit_candidate_overrides(args):
        aggregation = args.candidate_aggregation or candidate.aggregation
        min_rating = (
            args.candidate_min_rating
            if args.candidate_min_rating is not None
            else candidate.min_rating
        )
        candidate = RetrievalConfig(
            aggregation=aggregation,
            min_rating=min_rating,
            retrieval_profile=_resolve_profile_label(
                explicit_label=args.candidate_profile_label,
                min_rating=min_rating,
                aggregation=aggregation,
            ),
        )

    if candidate == baseline:
        raise SystemExit(
            "ERROR: candidate retrieval config matches the current baseline. "
            "Provide a distinct candidate config or point to a fit artifact with "
            "different retrieval settings."
        )
    return candidate
