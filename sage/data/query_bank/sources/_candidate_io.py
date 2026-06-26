"""File I/O helpers for query candidate import."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._candidate_models import _SourceRows


def _infer_delimiter(path: Path) -> str:
    """Infer delimiter from extension or small sample."""
    if path.suffix.lower() == ".tsv":
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","

    with path.open(encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return "\t"


def _resolve_fieldname(
    fieldnames: list[str],
    *,
    explicit: str | None,
    candidates: tuple[str, ...],
    required: bool,
    context: str,
) -> str | None:
    """Resolve a fieldname from explicit config or known alternatives."""
    available = {name: name for name in fieldnames}

    if explicit is not None:
        if explicit not in available:
            raise ValueError(
                f"Column '{explicit}' not found in {context}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return explicit

    for candidate in candidates:
        if candidate in available:
            return candidate

    if required:
        raise ValueError(
            f"Could not find required column in {context}. "
            f"Tried: {', '.join(candidates)}"
        )

    return None


@contextmanager
def _open_tabular_rows(
    path: Path,
) -> Iterator[_SourceRows]:
    """Open a local CSV/TSV file and stream row dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    chosen_delimiter = _infer_delimiter(path)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=chosen_delimiter)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No header row found in source file: {path}")
        yield _SourceRows(fieldnames=fieldnames, rows=reader)


@contextmanager
def _open_parquet_rows(path: Path) -> Iterator[_SourceRows]:
    """Open a local Parquet file and iterate rows without record-list copies."""
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Parquet import requires pandas. Install the pipeline dependencies "
            "before importing ESCI queries."
        ) from exc

    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet import requires a Parquet engine such as pyarrow. "
            "Install pyarrow before importing ESCI queries."
        ) from exc

    fieldnames = [str(column) for column in df.columns]
    if not fieldnames:
        raise ValueError(f"No columns found in Parquet source file: {path}")

    def _iter_rows() -> Iterator[dict[str, Any]]:
        for values in df.itertuples(index=False, name=None):
            yield {fieldname: value for fieldname, value in zip(fieldnames, values)}

    yield _SourceRows(fieldnames=fieldnames, rows=_iter_rows())


@contextmanager
def _open_source_rows(
    path: Path,
) -> Iterator[_SourceRows]:
    """Open supported source formats as one-pass row iterators."""
    if path.suffix.lower() == ".parquet":
        # Look up _open_parquet_rows via the parent candidates module so
        # tests can monkeypatch candidates._open_parquet_rows and have the
        # patch take effect here.
        import sys

        _candidates = sys.modules.get("sage.data.query_bank.sources.candidates")
        _impl = (
            getattr(_candidates, "_open_parquet_rows", None)
            if _candidates is not None
            else None
        ) or _open_parquet_rows
        with _impl(path) as source_rows:
            yield source_rows
        return

    with _open_tabular_rows(path) as source_rows:
        yield source_rows
