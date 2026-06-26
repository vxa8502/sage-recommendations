from __future__ import annotations

import argparse
from pathlib import Path


from .contracts import (
    DEFAULT_TOP_K,
    RetrievalAggregation,
)
from .paths import (
    _query_bank_path,
    _retrieval_fit_output_path,
    _retrieval_holdout_output_path,
)
from .prereqs import _require_stage2_prereqs
from ..shared import (
    cli_display_command,
    display_path,
    load_dotenv_if_available,
    run_command,
)
from ..script_command import script_command
from .status_commands import _check_stage2


def _run_fit_retrieval(
    *,
    query_bank_path: str | Path | None = None,
    output: str | Path | None = None,
    subsets: str | None = None,
    query_limit: int | None = None,
    top_k: int | None = None,
    candidate_min_rating: float | None = None,
    candidate_aggregation: RetrievalAggregation | None = None,
    candidate_profile_label: str | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)

    output_path = _retrieval_fit_output_path(output)
    command = (
        script_command("scripts/evaluate_retrieval_configs.py")
        .option("--query-bank-path", resolved_query_bank_path)
        .option("--output", output_path)
        .option("--comparison-role", "fit")
        .optional("--subsets", subsets)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--candidate-min-rating", candidate_min_rating)
        .optional("--candidate-aggregation", candidate_aggregation)
        .optional("--candidate-profile-label", candidate_profile_label)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_fit_retrieval(args: argparse.Namespace) -> None:
    _run_fit_retrieval(
        query_bank_path=getattr(args, "query_bank_path", None),
        output=getattr(args, "output", None),
        subsets=getattr(args, "subsets", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        candidate_min_rating=getattr(args, "candidate_min_rating", None),
        candidate_aggregation=getattr(args, "candidate_aggregation", None),
        candidate_profile_label=getattr(args, "candidate_profile_label", None),
    )


def _run_holdout_retrieval(
    *,
    query_bank_path: str | Path | None = None,
    analysis_path: str | Path | None = None,
    output: str | Path | None = None,
    subsets: str | None = None,
    query_limit: int | None = None,
    top_k: int | None = None,
    candidate_min_rating: float | None = None,
    candidate_aggregation: RetrievalAggregation | None = None,
    candidate_profile_label: str | None = None,
) -> None:
    load_dotenv_if_available()
    resolved_query_bank_path = _query_bank_path(query_bank_path)
    _require_stage2_prereqs(query_bank_path=resolved_query_bank_path)

    output_path = _retrieval_holdout_output_path(output)
    resolved_analysis_path = _retrieval_fit_output_path(analysis_path)
    explicit_candidate_supplied = any(
        value is not None
        for value in (
            candidate_min_rating,
            candidate_aggregation,
            candidate_profile_label,
        )
    )
    command_builder = (
        script_command("scripts/evaluate_retrieval_configs.py")
        .option("--query-bank-path", resolved_query_bank_path)
        .option("--output", output_path)
        .option("--comparison-role", "holdout")
    )
    if analysis_path is not None or not explicit_candidate_supplied:
        command_builder.option("--candidate-config-path", resolved_analysis_path)
    command = (
        command_builder.optional("--subsets", subsets)
        .optional("--query-limit", query_limit)
        .optional("--top-k", top_k)
        .optional("--candidate-min-rating", candidate_min_rating)
        .optional("--candidate-aggregation", candidate_aggregation)
        .optional("--candidate-profile-label", candidate_profile_label)
        .to_list()
    )
    run_command(command)


def command_stage_experiments_holdout_retrieval(args: argparse.Namespace) -> None:
    _run_holdout_retrieval(
        query_bank_path=getattr(args, "query_bank_path", None),
        analysis_path=getattr(args, "analysis_path", None),
        output=getattr(args, "output", None),
        subsets=getattr(args, "subsets", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", None),
        candidate_min_rating=getattr(args, "candidate_min_rating", None),
        candidate_aggregation=getattr(args, "candidate_aggregation", None),
        candidate_profile_label=getattr(args, "candidate_profile_label", None),
    )


def command_stage_experiments_all_retrieval(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    _check_stage2(query_bank_path=getattr(args, "query_bank_path", None))
    _run_fit_retrieval(
        query_bank_path=getattr(args, "query_bank_path", None),
        output=getattr(args, "fit_output", None),
        subsets=getattr(args, "fit_subsets", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", DEFAULT_TOP_K),
        candidate_min_rating=getattr(args, "candidate_min_rating", None),
        candidate_aggregation=getattr(args, "candidate_aggregation", None),
        candidate_profile_label=getattr(args, "candidate_profile_label", None),
    )
    fit_output = _retrieval_fit_output_path(getattr(args, "fit_output", None))
    _run_holdout_retrieval(
        query_bank_path=getattr(args, "query_bank_path", None),
        analysis_path=str(fit_output),
        output=getattr(args, "holdout_output", None),
        subsets=getattr(args, "holdout_subsets", None),
        query_limit=getattr(args, "query_limit", None),
        top_k=getattr(args, "top_k", DEFAULT_TOP_K),
        candidate_min_rating=getattr(args, "candidate_min_rating", None),
        candidate_aggregation=getattr(args, "candidate_aggregation", None),
        candidate_profile_label=getattr(args, "candidate_profile_label", None),
    )

    print("Retrieval artifacts ready")
    print("Suggested next steps:")
    print(f"  - inspect {display_path(fit_output)}")
    print(
        f"  - inspect {display_path(_retrieval_holdout_output_path(getattr(args, 'holdout_output', None)))}"
    )
    if getattr(args, "query_limit", None) is not None:
        print(
            "  - rerun without `--query-limit` before treating this retrieval "
            "comparison as canonical"
        )
    else:
        print(
            "  - if the candidate retrieval config wins on holdout, rerun "
            f"'{cli_display_command('stage', 'experiments', 'freeze-bundles')}' "
            "and then rematerialize faithfulness cases before trusting later "
            "explanation metrics"
        )
    print(
        f"  - run '{cli_display_command('stage', 'experiments', 'all')}' after "
        "retrieval settings are settled so the gate lane works from the intended "
        "retrieval baseline"
    )
