from __future__ import annotations

from .kaggle import _require_command
from .paths import (
    DEFAULT_ESCI_REPO,
    _stage_candidate_pool_path,
    _stage_esci_examples_path,
    _stage_esci_repo_dir,
)
from .validation import (
    _require_stage_overwrite_ack,
    _stage_overwrite_targets,
)
from ..shared import (
    cli_display_command,
    load_dotenv_if_available,
    remove_path,
    run_command,
)
from ..script_command import script_command


def _fetch_stage_queries(*, force: bool) -> None:
    load_dotenv_if_available()
    destination = _stage_esci_repo_dir()
    examples_path = _stage_esci_examples_path()

    if destination.exists():
        if force:
            remove_path(destination)
        elif examples_path.exists():
            print(f"Query source already staged at {destination}")
            return
        else:
            raise SystemExit(
                f"ERROR: {destination} already exists but the expected ESCI examples "
                "file is missing. Remove it or rerun with --force."
            )

    gh = _require_command(
        "gh", install_hint="Install GitHub CLI from https://cli.github.com/"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command([gh, "repo", "clone", DEFAULT_ESCI_REPO, str(destination)])

    if not examples_path.exists():
        raise SystemExit(
            "ERROR: ESCI repo cloned, but the expected examples parquet was not found "
            f"at {examples_path}."
        )

    print(f"Staged raw query source at {destination}")


def _import_stage_candidates(
    *,
    locale: str,
    version: str,
    allow_overwrite: bool,
) -> None:
    load_dotenv_if_available()
    examples_path = _stage_esci_examples_path()
    if not examples_path.exists():
        raise SystemExit(
            f"ERROR: Raw ESCI source not found at {examples_path}. Run "
            f"'{cli_display_command('stage', 'data', 'fetch-queries')}' first."
        )
    _require_stage_overwrite_ack(
        label="sage stage data import-candidates",
        target_paths=_stage_overwrite_targets(include_candidates=True),
        allow_overwrite=allow_overwrite,
        rerun_command=("stage", "data", "import-candidates"),
    )

    command = (
        script_command("scripts/import_esci_queries.py")
        .option("--input", examples_path)
        .option("--locale", locale)
        .option("--version", version)
        .to_list()
    )
    run_command(command)

    candidate_pool = _stage_candidate_pool_path()
    if not candidate_pool.exists():
        raise SystemExit(
            "ERROR: Candidate import completed but no candidate pool was written."
        )
    print(f"Candidate pool ready at {candidate_pool}")
