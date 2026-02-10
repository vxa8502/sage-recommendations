"""
Evaluation dataset loading utilities.
"""

import json

from sage.config import DATA_DIR
from sage.core import EvalCase


EVAL_DIR = DATA_DIR / "eval"


def load_eval_cases(filename: str) -> list[EvalCase]:
    """
    Load evaluation cases from JSON file.

    Args:
        filename: Filename in eval directory.

    Returns:
        List of EvalCase objects.
    """
    filepath = EVAL_DIR / filename

    with open(filepath) as f:
        data = json.load(f)

    return [
        EvalCase(
            query=d["query"],
            relevant_items=d["relevant_items"],
            user_id=d.get("user_id"),
        )
        for d in data
    ]
