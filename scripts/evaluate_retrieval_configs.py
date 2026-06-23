# ruff: noqa: E402,F401,F403,F405
"""
Compare baseline and candidate retrieval configs on judged query-bank subsets.

Run from project root:
    python scripts/evaluate_retrieval_configs.py --comparison-role fit \
        --candidate-min-rating 4 --candidate-aggregation max
    python scripts/evaluate_retrieval_configs.py --comparison-role holdout \
        --candidate-config-path data/retrieval/retrieval_fit.analysis.json
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.data._artifact_io import write_json_object
from sage.data.query_bank import (
    build_query_bank_identity,
    load_eval_cases_from_query_bank,
)
from sage.services.corpus_alignment import (
    CorpusAlignmentError,
    assert_corpus_alignment,
)
from sage.services.evaluation import evaluate_recommendations_with_details
from sage.services.retrieval import recommend
from sage.services.retrieval_eval import *
from sage.services.retrieval_eval import _artifacts as _artifacts_impl
from sage.services.retrieval_eval import _config as _config_impl
from sage.services.retrieval_eval import _evaluation as _evaluation_impl
from sage.services.retrieval_eval import _runner as _runner_impl


def _sync_compatibility_patches() -> None:
    """Keep historical script-level monkeypatches working for tests/tools."""

    _config_impl._resolve_profile_label = _resolve_profile_label
    _config_impl._current_retrieval_config = _current_retrieval_config
    _config_impl._config_from_payload = _config_from_payload
    _config_impl._load_candidate_config_from_artifact = (
        _load_candidate_config_from_artifact
    )
    _config_impl._has_explicit_candidate_overrides = _has_explicit_candidate_overrides

    _evaluation_impl.assert_corpus_alignment = assert_corpus_alignment
    _evaluation_impl.load_eval_cases_from_query_bank = load_eval_cases_from_query_bank
    _evaluation_impl.evaluate_recommendations_with_details = (
        evaluate_recommendations_with_details
    )
    _evaluation_impl.recommend = recommend
    _evaluation_impl._load_corpus_alignment = _load_corpus_alignment
    _evaluation_impl._evaluate_config = _evaluate_config
    _evaluation_impl._evaluate_subset = _evaluate_subset

    _artifacts_impl._candidate_config_source = _candidate_config_source
    _artifacts_impl._metrics_delta = _metrics_delta
    _artifacts_impl._recommend_winner = _recommend_winner
    _artifacts_impl._build_query_slice_breakdowns = _build_query_slice_breakdowns
    _artifacts_impl._build_evaluation_scope = _build_evaluation_scope
    _artifacts_impl._build_summary = _build_summary
    _artifacts_impl._build_subset_payload = _build_subset_payload

    _runner_impl.parse_args = parse_args
    _runner_impl._current_retrieval_config = _current_retrieval_config
    _runner_impl._resolve_candidate_config = _resolve_candidate_config
    _runner_impl.build_query_bank_identity = build_query_bank_identity
    _runner_impl._load_corpus_alignment = _load_corpus_alignment
    _runner_impl._evaluate_subsets = _evaluate_subsets
    _runner_impl._build_artifact = _build_artifact
    _runner_impl.write_json_object = write_json_object
    _runner_impl._log_run_header = _log_run_header
    _runner_impl._log_summary = _log_summary


def main(argv: Sequence[str] | None = None) -> None:
    _sync_compatibility_patches()
    _runner_impl.main(argv)


if __name__ == "__main__":
    main()
