from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard

from sage.config import (
    FAITHFULNESS_TARGET,
    RESULTS_DIR,
    EVAL_REPORTABLE_MIN_NDCG_AT_10,
)

RUN_FRESHNESS_TOLERANCE_SECONDS = 1.0
_ArtifactValidator = Callable[[dict[str, Any]], list[str]]


@dataclass(frozen=True)
class _EvalArtifactSpec:
    key: str
    label: str
    filename: str
    validator: _ArtifactValidator


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_json_artifact(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None, [f"Missing required artifact `{path.name}`."]
    except json.JSONDecodeError:
        return None, [f"Artifact `{path.name}` is not valid JSON."]
    except OSError as exc:
        return None, [f"Artifact `{path.name}` could not be read: {exc}"]

    if not isinstance(payload, dict):
        return None, [f"Artifact `{path.name}` must be a JSON object."]
    return payload, []


def _artifact_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _fresh_for_run(path: Path, run_started_at: datetime | None) -> bool | None:
    if run_started_at is None:
        return None
    mtime = _artifact_mtime(path)
    if mtime is None:
        return False
    return mtime >= (run_started_at.timestamp() - RUN_FRESHNESS_TOLERANCE_SECONDS)


def _validate_recommendation_artifact(payload: dict[str, Any]) -> list[str]:
    primary = payload.get("primary_metrics")
    if not isinstance(primary, dict):
        return ["Recommendation artifact is missing `primary_metrics`."]

    errors: list[str] = []
    for key, label in (
        ("ndcg_at_10", "NDCG@10"),
        ("hit_at_10", "Hit@10"),
        ("mrr", "MRR"),
    ):
        if not _is_number(primary.get(key)):
            errors.append(f"Recommendation artifact is missing numeric `{label}`.")
    return errors


def _validate_faithfulness_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if _extract_faithfulness_effective_metric(payload)[1] is None:
        errors.append(
            "Faithfulness artifact is missing a usable headline metric "
            "(`multi_metric.claim_level_avg_score`, `ragas.faithfulness_mean`, "
            "or `hhem.mean_score`)."
        )
    evaluation_scope = payload.get("evaluation_scope")
    if not isinstance(evaluation_scope, dict):
        errors.append("Faithfulness artifact is missing `evaluation_scope`.")
    return errors


def _validate_adjusted_faithfulness_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_number(payload.get("adjusted_pass_rate")):
        errors.append(
            "Adjusted faithfulness artifact is missing numeric `adjusted_pass_rate`."
        )
    if not isinstance(payload.get("n_total"), int):
        errors.append("Adjusted faithfulness artifact is missing integer `n_total`.")
    return errors


def _validate_boundary_artifact(payload: dict[str, Any]) -> list[str]:
    boundary_guardrail = payload.get("boundary_guardrail")
    if not isinstance(boundary_guardrail, dict):
        return ["Boundary artifact is missing `boundary_guardrail`."]
    status = boundary_guardrail.get("status")
    if not isinstance(status, str) or not status.strip():
        return ["Boundary artifact is missing `boundary_guardrail.status`."]
    return []


_LOAD_TEST_MIN_GROUNDED_SUCCESS_RATE = 0.70


def _validate_load_test_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    headline = payload.get("headline_metric")
    if isinstance(headline, dict) and _is_number(headline.get("value_ms")):
        pass
    elif _is_number(payload.get("p99_ms")):
        pass
    else:
        errors.append(
            "Load-test artifact is missing a measurable latency headline "
            "(`headline_metric.value_ms` or legacy `p99_ms`)."
        )

    # Guard against runs where the service was broken (e.g. model 404,
    # stale cache serving empty results). A low grounded_success_rate
    # means the load test measured a broken deployment, not a healthy one.
    measured = payload.get("measured") or {}
    api_quality = measured.get("api_quality") or {}
    grounded_rate = api_quality.get("grounded_success_rate")
    min_rate = _LOAD_TEST_MIN_GROUNDED_SUCCESS_RATE
    if _is_number(grounded_rate) and grounded_rate < min_rate:
        errors.append(
            "Load test grounded success rate is too low: "
            f"{grounded_rate:.1%} < {min_rate:.0%}. "
            "The deployed service may be broken — check that "
            "explanations are generating correctly."
        )

    return errors


_EVAL_ARTIFACT_SPECS = (
    _EvalArtifactSpec(
        key="recommendation",
        label="recommendation",
        filename="eval_natural_queries_latest.json",
        validator=_validate_recommendation_artifact,
    ),
    _EvalArtifactSpec(
        key="faithfulness",
        label="faithfulness",
        filename="faithfulness_latest.json",
        validator=_validate_faithfulness_artifact,
    ),
    _EvalArtifactSpec(
        key="adjusted_faithfulness",
        label="adjusted_faithfulness",
        filename="adjusted_faithfulness_latest.json",
        validator=_validate_adjusted_faithfulness_artifact,
    ),
    _EvalArtifactSpec(
        key="boundary",
        label="boundary",
        filename="boundary_behavior_latest.json",
        validator=_validate_boundary_artifact,
    ),
    _EvalArtifactSpec(
        key="load_test",
        label="load_test",
        filename="load_test_latest.json",
        validator=_validate_load_test_artifact,
    ),
)


def _inspect_artifact(
    *,
    label: str,
    path: Path,
    validator: _ArtifactValidator,
    run_started_at: datetime | None,
) -> dict[str, Any]:
    payload, errors = _read_json_artifact(path)
    fresh_for_run = _fresh_for_run(path, run_started_at)
    if payload is not None:
        errors.extend(validator(payload))
    if fresh_for_run is False:
        errors.append(
            f"Artifact `{path.name}` was not refreshed during the current evaluation run."
        )

    return {
        "label": label,
        "path": path,
        "payload": payload,
        "present": payload is not None,
        "fresh_for_run": fresh_for_run,
        "errors": errors,
        "ready": payload is not None and not errors,
    }


def _extract_boundary_status(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    boundary_guardrail = payload.get("boundary_guardrail")
    if isinstance(boundary_guardrail, dict):
        status = boundary_guardrail.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    summary = payload.get("summary")
    if isinstance(summary, dict):
        status = summary.get("boundary_guardrail_status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    return None


def _extract_boundary_reasons(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    boundary_guardrail = payload.get("boundary_guardrail")
    if not isinstance(boundary_guardrail, dict):
        return []
    violations = boundary_guardrail.get("violations")
    if not isinstance(violations, list):
        return []

    reasons: list[str] = []
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        message = violation.get("message")
        if isinstance(message, str) and message.strip():
            reasons.append(message.strip())
    return reasons


def _extract_retrieval_ndcg(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    primary = payload.get("primary_metrics")
    if not isinstance(primary, dict):
        return None
    value = primary.get("ndcg_at_10")
    return float(value) if _is_number(value) else None


def _extract_faithfulness_effective_metric(
    payload: dict[str, Any] | None,
) -> tuple[str | None, float | None]:
    if not isinstance(payload, dict):
        return None, None

    multi_metric = payload.get("multi_metric")
    if isinstance(multi_metric, dict):
        claim_level_avg = multi_metric.get("claim_level_avg_score")
        if _is_number(claim_level_avg):
            return "claim_level_avg_score", float(claim_level_avg)

    ragas = payload.get("ragas")
    if isinstance(ragas, dict):
        ragas_faith = ragas.get("faithfulness_mean")
        if _is_number(ragas_faith):
            return "ragas.faithfulness_mean", float(ragas_faith)

    hhem = payload.get("hhem")
    if isinstance(hhem, dict):
        mean_score = hhem.get("mean_score")
        if _is_number(mean_score):
            return "hhem.mean_score", float(mean_score)

    return None, None


def _faithfulness_target(payload: dict[str, Any] | None) -> float:
    if isinstance(payload, dict):
        target = payload.get("target")
        if _is_number(target):
            return float(target)
    return FAITHFULNESS_TARGET


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return deduped


def build_eval_status(
    *,
    results_dir: str | Path = RESULTS_DIR,
    run_started_at: datetime | None = None,
) -> dict[str, Any]:
    resolved_results_dir = Path(results_dir)
    artifacts = {
        spec.key: _inspect_artifact(
            label=spec.label,
            path=resolved_results_dir / spec.filename,
            validator=spec.validator,
            run_started_at=run_started_at,
        )
        for spec in _EVAL_ARTIFACT_SPECS
    }

    present_count = sum(1 for artifact in artifacts.values() if artifact["present"])
    total_count = len(artifacts)
    if present_count == 0:
        latest_artifacts = "NOT STARTED"
    elif present_count == total_count:
        latest_artifacts = (
            f"COMPLETE  ({present_count}/{total_count} latest artifacts present)"
        )
    else:
        latest_artifacts = (
            f"PARTIAL   ({present_count}/{total_count} latest artifacts present)"
        )

    execution_reasons: list[str] = []
    for artifact in artifacts.values():
        execution_reasons.extend(artifact["errors"])
    execution_reasons = _dedupe_reasons(execution_reasons)

    execution_complete = not execution_reasons and present_count == total_count
    if present_count == 0:
        execution_status = "NOT STARTED"
    elif execution_complete:
        execution_status = "COMPLETE"
    else:
        execution_status = "INCOMPLETE"

    boundary_payload = artifacts["boundary"]["payload"]
    boundary_status = _extract_boundary_status(boundary_payload)
    boundary_reasons = _extract_boundary_reasons(boundary_payload)
    safety_reasons = list(boundary_reasons)
    if artifacts["boundary"]["errors"]:
        safety_reasons.extend(artifacts["boundary"]["errors"])
    safety_reasons = _dedupe_reasons(safety_reasons)

    if boundary_status == "pass" and not artifacts["boundary"]["errors"]:
        safety_green = True
        safety_status = "PASS  [boundary-green]"
    elif boundary_status is None:
        safety_green = False
        safety_status = "NOT AVAILABLE"
    else:
        safety_green = False
        safety_status = boundary_status.upper()

    reportable_reasons: list[str] = []
    if not execution_complete:
        reportable_reasons.extend(execution_reasons)
    if not safety_green:
        reportable_reasons.append(
            "Boundary safety gate is not green, so the run is not reportable."
        )

    retrieval_ndcg = _extract_retrieval_ndcg(artifacts["recommendation"]["payload"])
    if retrieval_ndcg is None:
        reportable_reasons.append(
            "Recommendation artifact is missing headline `NDCG@10`."
        )
    elif retrieval_ndcg < EVAL_REPORTABLE_MIN_NDCG_AT_10:
        reportable_reasons.append(
            "Recommendation quality is below the reportable floor: "
            f"NDCG@10={retrieval_ndcg:.3f} < {EVAL_REPORTABLE_MIN_NDCG_AT_10:.3f}."
        )

    faithfulness_metric_name, faithfulness_metric_value = (
        _extract_faithfulness_effective_metric(artifacts["faithfulness"]["payload"])
    )
    faithfulness_target = _faithfulness_target(artifacts["faithfulness"]["payload"])
    if faithfulness_metric_value is None:
        reportable_reasons.append("Faithfulness headline metric is unavailable.")
    elif faithfulness_metric_value < faithfulness_target:
        reportable_reasons.append(
            "Faithfulness is below target: "
            f"{faithfulness_metric_name}={faithfulness_metric_value:.3f} < {faithfulness_target:.3f}."
        )

    faithfulness_payload = artifacts["faithfulness"]["payload"]
    if isinstance(faithfulness_payload, dict):
        evaluation_scope = faithfulness_payload.get("evaluation_scope")
        if isinstance(evaluation_scope, dict):
            if evaluation_scope.get("sample_limited"):
                reportable_reasons.append(
                    "Faithfulness artifact is sampled and cannot serve as the reportable baseline."
                )
            if evaluation_scope.get("generation_limited"):
                reportable_reasons.append(
                    "Faithfulness evaluation did not cover every selected case."
                )
        else:
            reportable_reasons.append(
                "Faithfulness artifact is missing `evaluation_scope`."
            )

    reportable_reasons = _dedupe_reasons(reportable_reasons)
    reportable_green = not reportable_reasons
    reportable_status = "PASS  [reportable-green]" if reportable_green else "WITHHELD"

    ragas_canonical: dict[str, Any] | None = None
    ragas_only_path = resolved_results_dir / "ragas_only_latest.json"
    if ragas_only_path.exists():
        try:
            with open(ragas_only_path, encoding="utf-8") as _f:
                _payload = json.load(_f)
            ragas_canonical = {
                "faithfulness_mean": _payload.get("faithfulness_mean"),
                "faithfulness_std": _payload.get("faithfulness_std"),
                "n_samples": _payload.get("n_samples"),
                "n_passing": _payload.get("n_passing"),
                "pass_rate": _payload.get("pass_rate"),
                "source_checkpoint": _payload.get("source_checkpoint"),
            }
        except Exception:
            pass

    return {
        "latest_artifacts": latest_artifacts,
        "artifact_statuses": artifacts,
        "execution_complete": execution_complete,
        "execution_status": execution_status,
        "execution_reasons": execution_reasons,
        "safety_green": safety_green,
        "safety_status": safety_status,
        "safety_reasons": safety_reasons,
        "reportable_green": reportable_green,
        "reportable_status": reportable_status,
        "reportable_reasons": reportable_reasons,
        "metrics": {
            "retrieval_ndcg_at_10": retrieval_ndcg,
            "reportable_min_ndcg_at_10": EVAL_REPORTABLE_MIN_NDCG_AT_10,
            "faithfulness_metric_name": faithfulness_metric_name,
            "faithfulness_metric_value": faithfulness_metric_value,
            "faithfulness_target": faithfulness_target,
            "ragas_canonical": ragas_canonical,
        },
    }
