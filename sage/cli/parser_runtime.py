from __future__ import annotations

import argparse

from .parser_common import (
    _lazy_command,
    _parse_port,
    _parse_positive_int,
)
from .shared import (
    DEFAULT_PORT,
    DEFAULT_QUERY,
    DEFAULT_TOP_K,
)


def add_runtime_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    demo_parser = subparsers.add_parser("demo", help="Run a recommendation demo")
    demo_parser.add_argument("--query", default=DEFAULT_QUERY)
    demo_parser.add_argument("--top-k", type=_parse_positive_int, default=DEFAULT_TOP_K)
    demo_parser.add_argument("--json", action="store_true")
    demo_parser.set_defaults(func=_lazy_command("sage.cli.runtime", "command_demo"))

    serve_parser = subparsers.add_parser("serve", help="Run the production API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=_parse_port, default=DEFAULT_PORT)
    serve_parser.set_defaults(func=_lazy_command("sage.cli.runtime", "command_serve"))
