from __future__ import annotations

import argparse
import os
from pathlib import Path

from .shared import (
    PROJECT_ROOT,
    cli_display_command,
    data_dir,
    dedupe_paths,
    display_path,
    load_dotenv_if_available,
    remove_path,
    results_dir,
)

BASELINE_PRESERVED_SOURCE_NAMES = {
    ".gitkeep",
    "README.md",
    "manual_boundary_queries_v2.jsonl",
}


def _evaluation_reset_targets() -> list[Path]:
    data_root = data_dir()
    asset_root = PROJECT_ROOT / "assets"
    targets = [
        data_root / "eval",
        data_root / "human_eval",
        results_dir(),
    ]
    targets.extend(sorted(data_root.glob("eda_stats*.json")))
    targets.extend(sorted(asset_root.glob("*.png")))
    return dedupe_paths(targets)


def _experimentation_reset_targets() -> list[Path]:
    data_root = data_dir()
    targets = _evaluation_reset_targets()
    targets.extend(
        [
            data_root / "calibration",
            data_root / "explanations",
            data_root / "figures",
        ]
    )
    return dedupe_paths(targets)


def _baseline_reset_targets() -> list[Path]:
    data_root = data_dir()
    query_bank_root = data_root / "query_bank"
    sources_root = query_bank_root / "sources"
    targets = _experimentation_reset_targets()
    targets.extend(
        [
            data_root / "indexed_product_ids.json",
            query_bank_root / "query_bank.jsonl",
            query_bank_root / "manifest.json",
            query_bank_root / "query_candidates.jsonl",
            query_bank_root / "split_leakage_audit.json",
        ]
    )
    targets.extend(sorted(data_root.glob("chunks_*.jsonl")))
    targets.extend(sorted(data_root.glob("*.log")))
    if sources_root.exists():
        targets.extend(
            child
            for child in sources_root.iterdir()
            if child.name not in BASELINE_PRESERVED_SOURCE_NAMES
        )
    return dedupe_paths(targets)


def _baseline_scaffold_placeholders() -> tuple[Path, ...]:
    data_root = data_dir()
    return (
        data_root / "calibration" / ".gitkeep",
        data_root / "eval_results" / ".gitkeep",
        data_root / "explanations" / ".gitkeep",
        data_root / "figures" / ".gitkeep",
        data_root / "query_bank" / ".gitkeep",
        data_root / "query_bank" / "sources" / ".gitkeep",
    )


def _restore_baseline_scaffold() -> tuple[str, ...]:
    restored: list[str] = []
    for path in _baseline_scaffold_placeholders():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        restored.append(display_path(path))
    return tuple(restored)


def _run_reset(
    *,
    label: str,
    targets: list[Path],
    preserved: tuple[str, ...],
    dry_run: bool,
    completion_lines: tuple[str, ...],
) -> None:
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    action = "Would clear" if dry_run else "Clearing"
    print(f"{action} {label}...")

    if existing:
        for path in existing:
            print(f"  - {display_path(path)}")
    else:
        print("  Nothing to clear.")

    if dry_run:
        print("Dry run only; no files were removed.")
        print("Preview only; rerun without --dry-run to apply these changes.")
    else:
        for path in existing:
            remove_path(path)

    print(f"  Preserved: {', '.join(preserved)}")
    if not dry_run:
        for line in completion_lines:
            print(line)


def command_reset_artifacts(args: argparse.Namespace) -> None:
    _run_reset(
        label="rerunnable evaluation artifacts",
        targets=_evaluation_reset_targets(),
        preserved=(
            "data/query_bank/",
            "data/indexed_product_ids.json",
            "data/query_bank/sources/",
            "data/calibration/",
            "hosted Qdrant",
        ),
        dry_run=getattr(args, "dry_run", False),
        completion_lines=(
            f"Done. Run '{cli_display_command('eval', 'run')}' to regenerate the automated evaluation set.",
        ),
    )


def command_reset_eval_dev(args: argparse.Namespace) -> None:
    _run_reset(
        label="rerunnable evaluation dev artifacts",
        targets=_evaluation_reset_targets(),
        preserved=(
            "data/query_bank/",
            "data/indexed_product_ids.json",
            "data/query_bank/sources/",
            "data/calibration/",
            "hosted Qdrant",
        ),
        dry_run=getattr(args, "dry_run", False),
        completion_lines=(
            f"Done. Run '{cli_display_command('eval', 'dev')}' for the sampled evaluation dev lane.",
        ),
    )


def command_reset_experiments(args: argparse.Namespace) -> None:
    _run_reset(
        label="rerunnable evaluation and experimentation artifacts",
        targets=_experimentation_reset_targets(),
        preserved=(
            "data/query_bank/",
            "data/query_bank/manifest.json",
            "data/query_bank/query_candidates.jsonl",
            "data/indexed_product_ids.json",
            "data/query_bank/sources/",
            "hosted Qdrant",
        ),
        dry_run=getattr(args, "dry_run", False),
        completion_lines=(
            "Done. Foundations stayed in place so you can restart from the canonical query bank.",
            "Suggested next steps:",
            "  - review home/EXPERIMENTATION.md",
            f"  - run '{cli_display_command('stage', 'experiments', 'all-retrieval', '--candidate-min-rating', '4')}' if retrieval is still the active calibration bottleneck",
            f"  - run '{cli_display_command('stage', 'experiments', 'all')}' for the clean calibration iteration path",
            f"  - run '{cli_display_command('stage', 'experiments', 'finalize', '--decision', 'baseline-retained', '--retrieval-decision', 'baseline-retained')}' or '{cli_display_command('stage', 'experiments', 'finalize', '--decision', 'candidate-promoted', '--retrieval-decision', 'candidate-promoted')}' after the winning gate and retrieval configs are reflected in repo code",
            f"  - run '{cli_display_command('eval', 'run')}' when you want a fresh end-to-end artifact set",
        ),
    )


def command_reset_baseline(args: argparse.Namespace) -> None:
    dry_run = getattr(args, "dry_run", False)
    _run_reset(
        label=(
            "local ingestion/calibration/evaluation artifacts "
            "to return to the baseline scaffold contract"
        ),
        targets=_baseline_reset_targets(),
        preserved=(
            "data/README.md",
            "data/.gitignore",
            "data/query_bank/README.md",
            "data/query_bank/sources/README.md",
            "data/query_bank/sources/manual_boundary_queries_v2.jsonl",
            "hosted Qdrant",
        ),
        dry_run=dry_run,
        completion_lines=(
            "Done. Local data state is back at the baseline scaffold boundary.",
        ),
    )
    if dry_run:
        return

    restored = _restore_baseline_scaffold()
    print("  Restored scaffold placeholders:")
    for path in restored:
        print(f"  - {path}")
    print(
        f"Next: run '{cli_display_command('stage', 'data', 'check')}' or "
        f"'{cli_display_command('stage', 'data', 'all')}' when you're ready to stage again."
    )


def command_qdrant_status(_args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    try:
        from sage.adapters.vector_store import (
            get_client,
            get_collection_info,
            get_corpus_anchor,
        )
        from sage.services.corpus_alignment import get_corpus_alignment_status

        client = get_client()
        info = get_collection_info(client)
    except Exception:
        print("Configured Qdrant cluster not reachable")
        return

    for key, value in info.items():
        print(f"{key}: {value}")

    remote_anchor = get_corpus_anchor(client)
    print(f"remote_anchor_present: {remote_anchor is not None}")
    if isinstance(remote_anchor, dict):
        for key in (
            "corpus_fingerprint",
            "chunk_count",
            "collection_points_count",
            "stamped_at",
        ):
            if key in remote_anchor:
                print(f"remote_{key}: {remote_anchor[key]}")

    local_anchor_path = data_dir() / "indexed_product_ids.json"
    aligned, details = get_corpus_alignment_status(anchor_path=local_anchor_path)
    print(f"corpus_alignment_ready: {aligned}")
    if aligned:
        for key in ("corpus_fingerprint", "chunk_count", "collection_points_count"):
            if key in details:
                print(f"alignment_{key}: {details[key]}")
    elif "error" in details:
        print(f"corpus_alignment_error: {details['error']}")


def command_qdrant_stamp_anchor(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    from sage.services.corpus_alignment import CorpusAlignmentError, stamp_corpus_anchor

    try:
        result = stamp_corpus_anchor(anchor_path=args.anchor, force=args.force)
    except CorpusAlignmentError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    for key, value in result.items():
        print(f"{key}: {value}")


def command_health(_args: argparse.Namespace) -> None:
    load_dotenv_if_available()

    qdrant_url_configured = bool(os.getenv("QDRANT_URL"))
    llm_configured = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

    qdrant_connected = False
    collection_info: dict[str, object] | None = None

    if qdrant_url_configured:
        try:
            from sage.adapters.vector_store import get_client, get_collection_info

            client = get_client()
            collection_info = get_collection_info(client)
            qdrant_connected = True
        except Exception:
            qdrant_connected = False

    if qdrant_connected and llm_configured:
        status = "healthy"
    elif qdrant_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    print(f"status: {status}")
    print(f"qdrant_url_configured: {qdrant_url_configured}")
    print(f"qdrant_connected: {qdrant_connected}")
    print(f"llm_credentials_configured: {llm_configured}")

    if collection_info is not None:
        for key, value in collection_info.items():
            print(f"{key}: {value}")

    if status == "unhealthy":
        raise SystemExit(1)
