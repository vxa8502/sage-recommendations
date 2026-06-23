from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeAlias

JsonObject: TypeAlias = dict[str, Any]


def _load_json_payload_file(
    path: str | Path,
    *,
    description: str,
    error_cls: type[ValueError],
    missing_error_cls: type[Exception],
) -> Any:
    """Load a JSON file with consistent missing and parse errors."""
    filepath = Path(path)
    if not filepath.exists():
        raise missing_error_cls(f"{description} not found: {filepath}")

    try:
        with filepath.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise error_cls(
            f"Invalid JSON format in {description.lower()}: {filepath} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc


def _iter_jsonl_rows(
    path: str | Path,
    *,
    label: str,
) -> Iterator[tuple[Any, int]]:
    """Yield parsed JSONL rows with consistent error messages."""
    filepath = Path(path)
    with filepath.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped), line_no
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL format in {label}: {filepath} "
                    f"(line {line_no}, column {exc.colno})"
                ) from exc


def iter_jsonl_object_rows(
    path: str | Path,
    *,
    label: str,
    row_description: str,
) -> Iterator[tuple[JsonObject, int]]:
    """Yield JSONL object rows with one shared row-shape error contract."""
    filepath = Path(path)
    for raw, line_no in _iter_jsonl_rows(filepath, label=label):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Each {row_description} row must be a JSON object in "
                f"{label} line {line_no}, got {type(raw).__name__}: {filepath}"
            )
        yield raw, line_no


def _load_json_object_file(
    path: str | Path,
    *,
    description: str,
    error_cls: type[ValueError],
    missing_error_cls: type[Exception],
    require_nonempty: bool,
) -> JsonObject:
    """Load a JSON object file with consistent parse and emptiness handling."""
    filepath = Path(path)
    payload = _load_json_payload_file(
        filepath,
        description=description,
        error_cls=error_cls,
        missing_error_cls=missing_error_cls,
    )

    if not isinstance(payload, dict):
        raise error_cls(
            f"{description} must contain a JSON object, got "
            f"{type(payload).__name__}: {filepath}"
        )
    if require_nonempty and not payload:
        raise error_cls(f"{description} is empty in {filepath}")
    return payload


def load_optional_json_object_file(
    path: str | Path,
    *,
    description: str,
    error_cls: type[ValueError] = ValueError,
) -> JsonObject:
    """Load a JSON object file, preserving FileNotFoundError for missing paths."""
    return _load_json_object_file(
        path,
        description=description,
        error_cls=error_cls,
        missing_error_cls=FileNotFoundError,
        require_nonempty=False,
    )


def load_required_json_object_file(
    path: str | Path,
    *,
    description: str,
    error_cls: type[ValueError] = ValueError,
) -> JsonObject:
    """Load a required JSON object file, using error_cls for missing paths too."""
    return _load_json_object_file(
        path,
        description=description,
        error_cls=error_cls,
        missing_error_cls=error_cls,
        require_nonempty=True,
    )


def load_json_array_file(
    path: str | Path,
    *,
    description: str,
    error_cls: type[ValueError] = ValueError,
    missing_error_cls: type[Exception] = FileNotFoundError,
    require_nonempty: bool = False,
) -> list[Any]:
    """Load a JSON array file with the shared artifact error contract."""
    filepath = Path(path)
    payload = _load_json_payload_file(
        filepath,
        description=description,
        error_cls=error_cls,
        missing_error_cls=missing_error_cls,
    )

    if not isinstance(payload, list):
        raise error_cls(
            f"{description} must contain a JSON array, got "
            f"{type(payload).__name__}: {filepath}"
        )
    if require_nonempty and not payload:
        raise error_cls(f"{description} is empty in {filepath}")
    return payload


def write_jsonl_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> Path:
    """Persist JSONL rows with stable UTF-8 encoding."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=ensure_ascii,
                    sort_keys=sort_keys,
                )
                + "\n"
            )
    return filepath


def write_json_object(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
) -> Path:
    """Persist a JSON object with stable pretty-printing."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(
            dict(payload),
            f,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
        )
        f.write("\n")
    return filepath
