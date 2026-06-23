"""Evidence-gate holdout comparison workflow."""

from __future__ import annotations

from collections.abc import Sequence

from sage.config import get_logger
from sage.data._artifact_io import write_json_object
from sage.data.query_bank import QueryBankSubsetEmptyError, build_query_bank_identity
from sage.services.calibration._holdout_config import (
    _build_parser,
    _resolve_run_config,
)
from sage.services.calibration._holdout_evaluation import (
    HoldoutDependencies,
    _build_base_artifact,
    _build_query_slice_metrics,
    _evaluate_subset,
    _update_evaluation_scope,
)
from sage.services.calibration._holdout_policy import (
    DEFAULT_SUBSETS,
    SubsetEvaluationPolicy,
)
from sage.services.calibration._holdout_reporting import (
    _log_retrieval_readiness,
    _log_subset_policy,
    _print_comparison,
)
from sage.services.calibration._holdout_thresholds import (
    _load_candidate_threshold,
)
from sage.services.calibration._analysis import (
    compare_gate_thresholds,
    current_gate_threshold,
)
from sage.services.calibration._dataset import (
    build_gate_calibration_dataset,
    ensure_calibration_retrieval_ready,
)
from sage.services.calibration._types import GateCalibrationRetrievalError

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_SUBSETS",
    "HoldoutDependencies",
    "_build_parser",
    "_build_query_slice_metrics",
    "default_dependencies",
    "main",
    "run_holdout",
]


def default_dependencies() -> HoldoutDependencies:
    return HoldoutDependencies(
        ensure_retrieval_ready=ensure_calibration_retrieval_ready,
        build_dataset=build_gate_calibration_dataset,
        compare_thresholds=compare_gate_thresholds,
        current_threshold=current_gate_threshold,
        build_query_bank_identity=build_query_bank_identity,
        build_query_slice_metrics=_build_query_slice_metrics,
    )


def run_holdout(
    argv: Sequence[str] | None = None,
    *,
    dependencies: HoldoutDependencies | None = None,
) -> None:
    dependencies = dependencies or default_dependencies()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = _resolve_run_config(args)
        policy = SubsetEvaluationPolicy.from_subsets(config.subsets)
        baseline_threshold = dependencies.current_threshold()
        candidate = _load_candidate_threshold(
            config.analysis_path,
            tokens=config.candidate_tokens,
            chunks=config.candidate_chunks,
            score=config.candidate_score,
        )

        retrieval_info = dependencies.ensure_retrieval_ready()
        _log_retrieval_readiness(retrieval_info)
        logger.info("Candidate threshold source: %s", candidate.source)
        _log_subset_policy(policy)

        results, subset_results = _build_base_artifact(
            config=config,
            baseline_threshold=baseline_threshold,
            candidate=candidate,
            retrieval_info=retrieval_info,
            policy=policy,
            dependencies=dependencies,
        )

        for subset in config.subsets:
            comparison = _evaluate_subset(
                config,
                subset=subset,
                baseline_threshold=baseline_threshold,
                candidate_threshold=candidate.threshold,
                policy=policy,
                dependencies=dependencies,
            )
            subset_results[subset] = comparison
            _print_comparison(subset, comparison)

        _update_evaluation_scope(results, subset_results=subset_results)
        write_json_object(config.output_path, results)
        logger.info("Saved holdout comparison to %s", config.output_path)
    except GateCalibrationRetrievalError as exc:
        raise SystemExit(
            "ERROR: holdout retrieval failed. "
            f"{exc} "
            "If the cluster is flaky, retry without `--strict-retrieval` and "
            "inspect the skipped-query summary."
        ) from exc
    except QueryBankSubsetEmptyError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    run_holdout(argv)
