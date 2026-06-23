"""CLI wrapper for frozen-case faithfulness evaluation.

Usage:
    python scripts/faithfulness.py
    python scripts/faithfulness.py --ragas
    python scripts/faithfulness.py --samples all
    python scripts/faithfulness.py --samples 20
    python scripts/faithfulness.py --ragas --ragas-samples 25
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SAMPLES: int | None = None
DEFAULT_RAGAS_SAMPLES: int | None = None


def _parse_sample_limit(value: str) -> int | None:
    """Parse an integer sample cap or the sentinel `all`."""
    lowered = value.strip().lower()
    if lowered in {"all", "full"}:
        return None
    try:
        parsed = int(lowered)
    except ValueError as exc:  # pragma: no cover - argparse handles presentation
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "sample limit must be a positive integer or 'all'"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation")
    parser.add_argument(
        "--surface",
        choices=("dev", "final"),
        default="final",
        help=(
            "Artifact surface to evaluate when --cases-path is not supplied "
            "(default: final)"
        ),
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=_parse_sample_limit,
        default=DEFAULT_SAMPLES,
        help=(
            "Evaluate all frozen cases by default; pass an integer for a "
            "deterministic stratified sample"
        ),
    )
    parser.add_argument("--ragas", action="store_true", help="Include RAGAS evaluation")
    parser.add_argument(
        "--ragas-samples",
        type=_parse_sample_limit,
        default=DEFAULT_RAGAS_SAMPLES,
        help="Optional RAGAS-specific case cap; defaults to the full evaluated set",
    )
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=None,
        help=(
            "Optional frozen faithfulness-case JSONL to evaluate. Defaults to the "
            "surface-specific canonical path."
        ),
    )
    parser.add_argument(
        "--outcomes-path",
        type=Path,
        default=None,
        help=(
            "Optional faithfulness-case outcomes JSONL used for coverage reporting. "
            "Defaults to the sibling outcomes file implied by --cases-path."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Optional frozen-case manifest JSON used to load the calibration freeze "
            "timestamp. Defaults to the sibling manifest implied by --cases-path."
        ),
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help=(
            "Run the experimental grounding delta diagnostic "
            "(not part of default evaluation reporting)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from sage.data.faithfulness import (
        FaithfulnessCasesEmptyError,
        FaithfulnessCasesManifestError,
        faithfulness_cases_path_for_surface,
        normalize_faithfulness_surface,
    )
    from sage.data.query_bank import QueryBankSubsetEmptyError
    from sage.services.faithfulness._runner import run_evaluation, run_grounding_delta

    surface = normalize_faithfulness_surface(args.surface)
    cases_path = args.cases_path or faithfulness_cases_path_for_surface(surface)

    try:
        if args.delta:
            run_grounding_delta(cases_path=cases_path)
        else:
            run_evaluation(
                n_samples=args.samples,
                run_ragas=args.ragas,
                ragas_samples=args.ragas_samples,
                cases_path=cases_path,
                outcomes_path=args.outcomes_path,
                manifest_path=args.manifest_path,
            )
    except (
        FaithfulnessCasesEmptyError,
        FaithfulnessCasesManifestError,
        QueryBankSubsetEmptyError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
