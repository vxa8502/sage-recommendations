"""Persistence helpers for split leakage audit artifacts."""

from __future__ import annotations

from pathlib import Path

from sage.data._artifact_io import write_json_object
from sage.data.split_leakage._config import SPLIT_LEAKAGE_AUDIT_PATH
from sage.data.split_leakage._types import JsonObject


def save_split_leakage_audit(
    audit: JsonObject,
    path: str | Path = SPLIT_LEAKAGE_AUDIT_PATH,
) -> Path:
    """Save the split-leakage audit as stable pretty-printed JSON."""
    if not isinstance(audit, dict):
        raise TypeError(
            "Split leakage audit must be a dict before it can be saved, got "
            f"{type(audit).__name__}"
        )

    return write_json_object(path, audit)
