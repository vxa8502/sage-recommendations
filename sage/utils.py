"""
Shared utility functions.
"""

import json
from datetime import datetime
from pathlib import Path


def save_results(data: dict, prefix: str, directory: Path | None = None) -> Path:
    """Save results as both timestamped and latest JSON files.

    Args:
        data: Serializable dict to save.
        prefix: File prefix (e.g., "faithfulness", "human_eval").
        directory: Target directory. Defaults to RESULTS_DIR from config.

    Returns:
        Path to the timestamped file.
    """
    if directory is None:
        from sage.config import RESULTS_DIR
        directory = RESULTS_DIR

    directory.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_file = directory / f"{prefix}_{ts}.json"
    latest_file = directory / f"{prefix}_latest.json"

    for path in (ts_file, latest_file):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return ts_file
