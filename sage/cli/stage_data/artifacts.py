from __future__ import annotations

from .kaggle import (
    _require_command,
    _require_kaggle_stage_prereqs,
    _run_kaggle_stage_kernel,
    _stage_kernel_ref,
    _wait_for_kaggle_kernel,
)
from .paths import (
    DEFAULT_KAGGLE_CHUNK_PATTERN,
    DEFAULT_KAGGLE_OUTPUT_PATTERN,
    _stage_indexed_product_ids_path,
)
from .validation import (
    _require_stage_overwrite_ack,
    _stage_overwrite_targets,
)
from ..shared import data_dir, load_dotenv_if_available, run_command


def _run_stage_kaggle(
    *,
    accelerator: str,
    subset_size: int,
    wait: bool,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    load_dotenv_if_available()
    _require_kaggle_stage_prereqs()
    kernel_ref = _stage_kernel_ref()

    _run_kaggle_stage_kernel(
        kernel_ref=kernel_ref,
        accelerator=accelerator,
        subset_size=subset_size,
    )

    if wait:
        _wait_for_kaggle_kernel(
            kernel_ref,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )


def _pull_stage_artifacts(
    *,
    wait: bool,
    poll_seconds: int,
    timeout_seconds: int,
    include_chunk_manifest: bool,
    allow_overwrite: bool,
) -> None:
    load_dotenv_if_available()
    _require_kaggle_stage_prereqs()
    kernel_ref = _stage_kernel_ref()
    _require_stage_overwrite_ack(
        label="sage stage data pull-artifacts",
        target_paths=_stage_overwrite_targets(
            include_pull_artifacts=True,
            include_chunk_manifest=include_chunk_manifest,
            kernel_ref=kernel_ref,
        ),
        allow_overwrite=allow_overwrite,
        rerun_command=("stage", "data", "pull-artifacts"),
    )

    if wait:
        _wait_for_kaggle_kernel(
            kernel_ref,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )

    kaggle = _require_command(
        "kaggle", install_hint="Install it with `pip install kaggle`."
    )
    destination = data_dir()
    destination.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            kaggle,
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(destination),
            "-o",
            "--file-pattern",
            DEFAULT_KAGGLE_OUTPUT_PATTERN,
        ]
    )
    if include_chunk_manifest:
        run_command(
            [
                kaggle,
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(destination),
                "-o",
                "--file-pattern",
                DEFAULT_KAGGLE_CHUNK_PATTERN,
            ]
        )

    indexed_product_ids = _stage_indexed_product_ids_path()
    if not indexed_product_ids.exists():
        raise SystemExit(
            "ERROR: Kaggle output download completed, but data/indexed_product_ids.json "
            "was not found."
        )
    print(f"Pulled Stage 1 artifacts into {destination}")
