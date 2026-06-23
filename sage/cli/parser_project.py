from __future__ import annotations

import argparse

from .parser_common import _lazy_command


def add_project_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    fmt_parser = subparsers.add_parser("fmt", help="Format the codebase with Ruff")
    fmt_parser.set_defaults(func=_lazy_command("sage.cli.project", "command_fmt"))

    lint_parser = subparsers.add_parser("lint", help="Run Ruff lint and format checks")
    lint_parser.set_defaults(func=_lazy_command("sage.cli.project", "command_lint"))

    typecheck_parser = subparsers.add_parser("typecheck", help="Run mypy")
    typecheck_parser.set_defaults(
        func=_lazy_command("sage.cli.project", "command_typecheck")
    )

    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.set_defaults(func=_lazy_command("sage.cli.project", "command_test"))

    ci_parser = subparsers.add_parser("ci", help="Run lint, typecheck, and tests")
    ci_parser.set_defaults(func=_lazy_command("sage.cli.project", "command_ci"))
