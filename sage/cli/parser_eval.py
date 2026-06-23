from __future__ import annotations

import argparse

from .parser_common import (
    _BOUNDARY_MIN_RATING_HELP,
    _add_query_bank_path_argument,
    _add_retrieval_runtime_arguments,
    _lazy_command,
    _parse_positive_int,
    _parse_sample_limit,
)
from .shared import (
    DEFAULT_DEV_RAGAS_SAMPLES,
    DEFAULT_DEV_REQUESTS,
    DEFAULT_DEV_SAMPLES,
    DEFAULT_RAGAS_SAMPLES,
    DEFAULT_REQUESTS,
    DEFAULT_SAMPLES,
    DEFAULT_URL,
)


def add_eval_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    eval_parser = subparsers.add_parser("eval", help="Evaluation workflows")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    run_parser = eval_subparsers.add_parser(
        "run", help="Run the full reproducible evaluation workflow"
    )
    run_parser.add_argument(
        "--samples",
        type=_parse_sample_limit,
        default=DEFAULT_SAMPLES,
        help=(
            "Evaluate all frozen faithfulness cases by default; pass an integer "
            "for a deterministic stratified sample"
        ),
    )
    run_parser.add_argument(
        "--ragas-samples",
        type=_parse_sample_limit,
        default=DEFAULT_RAGAS_SAMPLES,
        help="Optional separate RAGAS cap; defaults to all evaluated cases",
    )
    run_parser.add_argument("--url", default=DEFAULT_URL)
    run_parser.add_argument(
        "--requests", type=_parse_positive_int, default=DEFAULT_REQUESTS
    )
    run_parser.set_defaults(func=_lazy_command("sage.cli.evaluation", "command_eval"))

    dev_parser = eval_subparsers.add_parser(
        "dev", help="Run the sampled evaluation dev lane"
    )
    dev_parser.add_argument(
        "--samples",
        type=_parse_sample_limit,
        default=DEFAULT_DEV_SAMPLES,
        help=(
            "Deterministic stratified faithfulness sample size for the evaluation dev lane"
        ),
    )
    dev_parser.add_argument(
        "--ragas-samples",
        type=_parse_sample_limit,
        default=DEFAULT_DEV_RAGAS_SAMPLES,
        help="Optional separate RAGAS cap for the evaluation dev lane",
    )
    dev_parser.add_argument("--url", default=DEFAULT_URL)
    dev_parser.add_argument(
        "--requests", type=_parse_positive_int, default=DEFAULT_DEV_REQUESTS
    )
    dev_parser.set_defaults(
        func=_lazy_command("sage.cli.evaluation", "command_eval_dev")
    )

    summary_parser = eval_subparsers.add_parser(
        "summary", help="Print the latest evaluation summary"
    )
    summary_parser.set_defaults(
        func=_lazy_command("sage.cli.evaluation", "command_eval_summary")
    )

    boundary_parser = eval_subparsers.add_parser(
        "boundary",
        help="Run the boundary_eval refusal/clarification guardrail benchmark",
    )
    _add_query_bank_path_argument(
        boundary_parser,
        default="data/query_bank/query_bank.jsonl",
        help_text="Path to the canonical boundary-eval query-bank JSONL",
    )
    boundary_parser.add_argument("--subset-tag", default="boundary_eval")
    _add_retrieval_runtime_arguments(
        boundary_parser,
        top_k_default=3,
        min_rating_help=_BOUNDARY_MIN_RATING_HELP,
    )
    boundary_parser.add_argument("--max-evidence", type=_parse_positive_int, default=3)
    boundary_parser.set_defaults(
        func=_lazy_command("sage.cli.evaluation", "command_eval_boundary")
    )
