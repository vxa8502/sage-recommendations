from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QUERY = "example product query"
DEFAULT_URL = "https://vxa8502-sage.hf.space"
DEFAULT_REQUESTS = 100
DEFAULT_SAMPLES: int | None = None
DEFAULT_RAGAS_SAMPLES: int | None = None
DEFAULT_DEV_REQUESTS = 25
DEFAULT_DEV_SAMPLES: int | None = 25
DEFAULT_DEV_RAGAS_SAMPLES: int | None = 10
DEFAULT_TOP_K = 1
DEFAULT_PORT = 8000
CLI_NAME = "sage"
CLI_FALLBACK = "python -m sage.cli"


@dataclass(frozen=True)
class Step:
    title: str
    commands: tuple[tuple[str, ...], ...]
    optional_message: str | None = None
    suppress_stderr: bool = False


def data_dir() -> Path:
    return Path(os.getenv("SAGE_DATA_DIR", PROJECT_ROOT / "data"))


def results_dir() -> Path:
    return data_dir() / "eval_results"


def manual_boundary_source_path() -> Path:
    """Return the checked-in manual ingestion boundary query source."""
    return (
        PROJECT_ROOT
        / "data"
        / "query_bank"
        / "sources"
        / "manual_boundary_queries_v2.jsonl"
    )


def python_command(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def cli_display_command(*args: str) -> str:
    return " ".join((CLI_NAME, *args))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_status_line(key: str, value: object) -> None:
    print(f"{key}: {value}")


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw_item in value:
        if isinstance(raw_item, str) and raw_item.strip():
            items.append(raw_item.strip())
    return items


def normalize_query_ids(value: object) -> list[str] | None:
    normalized = normalize_string_list(value)
    if not isinstance(value, list) or len(normalized) != len(value):
        return None
    return sorted(normalized)


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def run_command(
    command: tuple[str, ...] | list[str],
    *,
    optional_message: str | None = None,
    suppress_stderr: bool = False,
    cwd: Path = PROJECT_ROOT,
) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stderr=subprocess.DEVNULL if suppress_stderr else None,
        )
    except subprocess.CalledProcessError as exc:
        if optional_message is not None:
            print(optional_message)
            return
        raise SystemExit(exc.returncode) from exc


def capture_output(command: tuple[str, ...] | list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.rstrip()


def run_steps(title: str, steps: list[Step], footer: list[str]) -> None:
    print(f"=== {title} ===")
    for step in steps:
        print()
        print(f"--- {step.title} ---")
        for command in step.commands:
            run_command(
                command,
                optional_message=step.optional_message,
                suppress_stderr=step.suppress_stderr,
            )
    print()
    for line in footer:
        print(line)


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def ensure_llm_credentials() -> None:
    load_dotenv_if_available()
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print("LLM credentials OK")
        return
    raise SystemExit(
        "ERROR: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set "
        "(checked shell + .env)."
    )


def ensure_qdrant() -> None:
    load_dotenv_if_available()
    if not os.getenv("QDRANT_URL"):
        raise SystemExit(
            "ERROR: QDRANT_URL is not set. Configure a hosted Qdrant cluster in "
            ".env or the environment before running Sage."
        )

    try:
        from sage.adapters.vector_store import get_client
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Qdrant client dependencies are unavailable. Run setup first."
        ) from exc

    try:
        client = get_client()
        client.get_collections()
    except Exception as exc:
        raise SystemExit(
            "ERROR: Cannot connect to the configured Qdrant cluster. Check "
            "QDRANT_URL and QDRANT_API_KEY in .env or the environment."
        ) from exc

    print("Qdrant OK")


def ensure_env() -> None:
    ensure_llm_credentials()
    ensure_qdrant()
    print("Environment OK")


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            return
        path.unlink()
    except FileNotFoundError:
        # Another reset step or process may have removed the path after the
        # existence check. Treat that as an already-completed delete.
        return
