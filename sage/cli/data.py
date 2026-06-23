from __future__ import annotations

import argparse

from .shared import data_dir, ensure_qdrant, run_command
from .script_command import script_command


def command_data_build(args: argparse.Namespace) -> None:
    ensure_qdrant()

    command = (
        script_command("scripts/pipeline.py")
        .flag("--force", args.force)
        .optional("--subset-size", args.subset_size)
        .to_list()
    )

    print("=== DATA PIPELINE ===")
    run_command(command)

    train_path = data_dir() / "splits" / "train.parquet"
    if not train_path.exists():
        raise SystemExit("FAIL: train.parquet not created")

    print("Data pipeline complete")
