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

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or data fails validation.
    """
    filepath = EVAL_DIR / filename

    # Load JSON with helpful error messages
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON format in evaluation file: {filepath} "
            f"(line {e.lineno}, column {e.colno})"
        )

    # Handle empty list gracefully
    if not data:
        return []

    # Validate structure is a list
    if not isinstance(data, list):
        raise ValueError(
            f"Evaluation file must contain a JSON array, got {type(data).__name__}: "
            f"{filepath}"
        )

    # Validate each case
    cases = []
    for i, d in enumerate(data):
        # Check required fields
        if "query" not in d:
            raise ValueError(f"Missing 'query' field in case {i}: {filepath}")
        if "relevant_items" not in d:
            raise ValueError(f"Missing 'relevant_items' field in case {i}: {filepath}")

        # Validate relevant_items is a dict
        relevant_items = d["relevant_items"]
        if not isinstance(relevant_items, dict):
            raise ValueError(
                f"'relevant_items' must be a dict in case {i}, "
                f"got {type(relevant_items).__name__}: {filepath}"
            )

        # Validate relevance scores are numeric
        for product_id, score in relevant_items.items():
            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"Relevance score for '{product_id}' must be numeric in case {i}, "
                    f"got {type(score).__name__}: '{score}'"
                )

        cases.append(
            EvalCase(
                query=d["query"],
                relevant_items=relevant_items,
                user_id=d.get("user_id"),
            )
        )

    return cases
