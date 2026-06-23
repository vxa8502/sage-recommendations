from __future__ import annotations

import argparse
import sys

from .shared import run_command


def _run_fmt() -> None:
    run_command((sys.executable, "-m", "ruff", "format", "sage/", "scripts/", "tests/"))
    run_command(
        (sys.executable, "-m", "ruff", "check", "--fix", "sage/", "scripts/", "tests/")
    )


def _run_lint() -> None:
    run_command((sys.executable, "-m", "ruff", "check", "sage/", "scripts/", "tests/"))
    run_command(
        (
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "sage/",
            "scripts/",
            "tests/",
        )
    )


def _run_typecheck() -> None:
    run_command((sys.executable, "-m", "mypy", "sage/", "--ignore-missing-imports"))


def _run_tests() -> None:
    run_command((sys.executable, "-m", "pytest", "tests/", "-v"))


def command_fmt(_args: argparse.Namespace) -> None:
    _run_fmt()


def command_lint(_args: argparse.Namespace) -> None:
    _run_lint()


def command_typecheck(_args: argparse.Namespace) -> None:
    _run_typecheck()


def command_test(_args: argparse.Namespace) -> None:
    _run_tests()


def command_ci(_args: argparse.Namespace) -> None:
    _run_lint()
    _run_typecheck()
    _run_tests()
    print("CI checks passed")
