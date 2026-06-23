from __future__ import annotations

import argparse

from .script_command import script_command
from .shared import ensure_env, python_command, run_command


def command_demo(args: argparse.Namespace) -> None:
    ensure_env()
    command = (
        script_command("scripts/demo.py")
        .option("--query", args.query)
        .option("--top-k", args.top_k)
        .flag("--json", args.json)
        .to_list()
    )
    run_command(command)


def command_serve(args: argparse.Namespace) -> None:
    ensure_env()
    run_command(
        python_command(
            "-m",
            "sage.api.run",
            "--host",
            args.host,
            "--port",
            str(args.port),
        )
    )
