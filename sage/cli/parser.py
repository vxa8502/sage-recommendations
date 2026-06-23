from __future__ import annotations

import argparse

from .parser_common import (
    _lazy_command,
)
from .parser_data import add_data_parser
from .parser_eval import add_eval_parser
from .parser_project import add_project_parsers
from .parser_qdrant import add_qdrant_parser
from .parser_reset import add_reset_parser
from .parser_runtime import add_runtime_parsers
from .stage_data.parser import add_stage_data_parser
from .stage_experiments.parser import add_stage_experiments_parser
from .shared import (
    CLI_FALLBACK,
    CLI_NAME,
)


def build_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  sage health
  sage stage data check
  sage stage data all --with-candidates
  sage stage experiments all-retrieval --candidate-min-rating 4
  sage stage experiments all
  sage stage experiments full --decision baseline-retained --retrieval-decision baseline-retained --with-boundary
  sage stage experiments finalize --decision baseline-retained --retrieval-decision baseline-retained --with-boundary
  sage qdrant status
  sage qdrant stamp-anchor
  sage data build --subset-size 1000
  sage eval dev
  sage eval run --samples all --ragas-samples 25 --url https://vxa8502-sage.hf.space
  sage eval boundary --query-limit 10
  sage eval summary
  sage reset eval-dev --dry-run
  sage reset baseline --dry-run
  sage reset experiments --dry-run
  sage demo --query "wireless earbuds for running"
  sage serve --port 8000
  sage ci

If you are already inside `.venv`, prefer `sage ...`.
Fallback form: `python -m sage.cli ...`
For day-to-day workflows, use the CLI directly.
`make help` mirrors this help, and `make ci-fresh` is kept for fresh-venv CI parity checks.
"""

    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Canonical project CLI for Sage workflows and contributor commands.",
        epilog=examples + f"\nConsole-script fallback: {CLI_FALLBACK}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser(
        "health", help="Run a quick local environment and Qdrant health check"
    )
    health_parser.set_defaults(func=_lazy_command("sage.cli.state", "command_health"))

    stage_parser = subparsers.add_parser(
        "stage", help="Stage data artifacts before experiments"
    )
    stage_subparsers = stage_parser.add_subparsers(dest="stage_command", required=True)
    add_stage_data_parser(stage_subparsers)
    add_stage_experiments_parser(stage_subparsers)

    add_data_parser(subparsers)
    add_eval_parser(subparsers)
    add_runtime_parsers(subparsers)
    add_reset_parser(subparsers)
    add_qdrant_parser(subparsers)
    add_project_parsers(subparsers)

    return parser
