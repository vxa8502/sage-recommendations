from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from sage.data._artifact_io import write_json_object
from .paths import (
    DEFAULT_KAGGLE_ACCELERATOR,
    DEFAULT_KAGGLE_PACKAGE_DATASET,
    DEFAULT_STAGE_KERNEL_CONFIG_NAME,
    GPU_DISABLED_TOKENS,
)
from ..shared import PROJECT_ROOT, run_command


def _require_command(name: str, *, install_hint: str | None = None) -> str:
    executable = shutil.which(name)
    if executable is None:
        message = f"ERROR: Required command '{name}' is not installed or not on PATH."
        if install_hint:
            message += f" {install_hint}"
        raise SystemExit(message)
    return executable


def _gh_auth_available() -> bool:
    if os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"):
        return True
    gh = shutil.which("gh")
    if gh is None:
        return False
    result = subprocess.run(
        [gh, "auth", "status"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _kaggle_auth_available() -> bool:
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    access_token = Path.home() / ".kaggle" / "access_token"
    legacy_json = Path.home() / ".kaggle" / "kaggle.json"
    return access_token.exists() or legacy_json.exists()


def _stage_kernel_ref() -> str:
    kernel_ref = os.getenv("SAGE_KAGGLE_KERNEL", "").strip()
    if not kernel_ref:
        raise SystemExit(
            "ERROR: SAGE_KAGGLE_KERNEL is not set. Add a value like "
            "'your-kaggle-username/sage-stage-data' to .env or the environment."
        )
    if "/" not in kernel_ref:
        raise SystemExit(
            "ERROR: SAGE_KAGGLE_KERNEL must use the 'owner/kernel-slug' format."
        )
    return kernel_ref


def _stage_kernel_title(kernel_ref: str) -> str:
    title = os.getenv("SAGE_KAGGLE_KERNEL_TITLE", "").strip()
    if title:
        return title
    slug = kernel_ref.split("/", 1)[1]
    return slug.replace("-", " ").title()


def _stage_package_dataset() -> str:
    return os.getenv(
        "SAGE_KAGGLE_PACKAGE_DATASET",
        DEFAULT_KAGGLE_PACKAGE_DATASET,
    ).strip()


def _stage_accelerator() -> str:
    return os.getenv(
        "SAGE_KAGGLE_ACCELERATOR",
        DEFAULT_KAGGLE_ACCELERATOR,
    ).strip()


def _gpu_requested(accelerator: str) -> bool:
    return accelerator.strip().lower() not in GPU_DISABLED_TOKENS


def _require_kaggle_stage_prereqs() -> None:
    _require_command("kaggle", install_hint="Install it with `pip install kaggle`.")
    if not _kaggle_auth_available():
        raise SystemExit(
            "ERROR: Kaggle CLI auth was not detected. Set KAGGLE_API_TOKEN, "
            "KAGGLE_USERNAME/KAGGLE_KEY, or configure ~/.kaggle credentials."
        )
    _stage_kernel_ref()


def _fetch_kaggle_kernel_status_text(kernel_ref: str) -> str:
    kaggle = _require_command(
        "kaggle", install_hint="Install it with `pip install kaggle`."
    )
    result = subprocess.run(
        [kaggle, "kernels", "status", kernel_ref],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise SystemExit(
            f"ERROR: Could not fetch Kaggle kernel status for '{kernel_ref}'."
            + (f" {stderr}" if stderr else "")
        )
    return result.stdout.strip()


def _classify_kaggle_kernel_status(status_text: str) -> str:
    normalized = status_text.lower()
    if any(token in normalized for token in ("complete", "completed", "success")):
        return "complete"
    if any(token in normalized for token in ("error", "failed", "cancel", "killed")):
        return "failed"
    if any(
        token in normalized for token in ("running", "queued", "pending", "starting")
    ):
        return "running"
    return "unknown"


def _wait_for_kaggle_kernel(
    kernel_ref: str,
    *,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    started = time.monotonic()
    last_status_text: str | None = None

    while True:
        status_text = _fetch_kaggle_kernel_status_text(kernel_ref)
        state = _classify_kaggle_kernel_status(status_text)

        if status_text != last_status_text:
            print(status_text)
            last_status_text = status_text

        if state == "complete":
            print("Kaggle run completed")
            return
        if state == "failed":
            raise SystemExit(
                "ERROR: Kaggle run failed. Inspect the kernel on Kaggle before "
                "pulling artifacts."
            )
        if time.monotonic() - started >= timeout_seconds:
            raise SystemExit("ERROR: Timed out waiting for the Kaggle run to finish.")
        time.sleep(max(poll_seconds, 1))


def _prepare_kaggle_kernel_workspace(
    *,
    kernel_ref: str,
    package_dataset: str,
    accelerator: str,
    subset_size: int,
) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="sage-stage-data-"))
    script_source = PROJECT_ROOT / "scripts" / "kaggle_pipeline.py"
    script_target = workspace / "kaggle_pipeline.py"
    shutil.copy2(script_source, script_target)
    enable_gpu = _gpu_requested(accelerator)

    metadata = {
        "id": kernel_ref,
        "title": _stage_kernel_title(kernel_ref),
        "code_file": script_target.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": True,
        "dataset_sources": [package_dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    write_json_object(workspace / "kernel-metadata.json", metadata)
    write_json_object(
        workspace / DEFAULT_STAGE_KERNEL_CONFIG_NAME,
        {"subset_size": subset_size},
    )

    return workspace


def _run_kaggle_stage_kernel(
    *,
    kernel_ref: str,
    accelerator: str,
    subset_size: int,
) -> None:
    package_dataset = _stage_package_dataset()
    workspace = _prepare_kaggle_kernel_workspace(
        kernel_ref=kernel_ref,
        package_dataset=package_dataset,
        accelerator=accelerator,
        subset_size=subset_size,
    )
    print(f"Prepared Kaggle workspace: {workspace}")
    print(f"Kernel: {kernel_ref}")
    print(f"Package dataset: {package_dataset}")
    print(f"Subset size: {subset_size}")
    print(f"GPU requested via metadata: {_gpu_requested(accelerator)}")

    kaggle = _require_command(
        "kaggle", install_hint="Install it with `pip install kaggle`."
    )
    command = [kaggle, "kernels", "push", "-p", str(workspace)]
    run_command(command, cwd=PROJECT_ROOT)
