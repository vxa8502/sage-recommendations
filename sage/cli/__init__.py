"""
Canonical project CLI for Sage.

This package is the single orchestration surface for project workflows.
Use the installed `sage` console script inside `.venv`.
Make remains optional sugar on top of these commands.
"""

from __future__ import annotations

__all__ = ["build_parser", "main"]


def build_parser():
    from .parser import build_parser as _build_parser

    return _build_parser()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
