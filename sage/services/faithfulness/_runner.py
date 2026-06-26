"""Orchestrates frozen-case faithfulness evaluation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from sage.config import (
    FAITHFULNESS_TARGET,
    MAX_EVIDENCE,
    RESULTS_DIR,
    get_logger,
    log_banner,
    log_section,
    save_results,
)
from sage.core.freshness_policy import (
    summarize_evidence_guardrail_reports,
    summarize_freshness_guardrail_cases,
)
from sage.data.faithfulness import (
    FAITHFULNESS_CASES_PATH,
    load_faithfulness_cases_manifest,
    load_faithfulness_cases,
    load_frozen_freshness_reference,
    resolve_faithfulness_case_outcomes_path,
    resolve_faithfulness_cases_manifest_path,
)
from sage.core.query_classification import QUERY_SLICE_DESCRIPTIONS
from sage.services.faithfulness._reports import (
    _build_adjusted_results,
    _build_case_diagnostics,
    _build_query_slice_metrics,
    _log_freshness_guardrail,
    _log_query_slice_metrics,
)
from sage.services.faithfulness._scope import (
    DEFAULT_RAGAS_SAMPLES,
    _copy_object_dict,
    _infer_retrieval_policy,
    _load_materialization_coverage,
    _sample_limit_label,
    _select_case_scope,
)
from sage.services.runtime_provenance import build_run_provenance

logger = get_logger(__name__)


def run_evaluation(
    n_samples: int | None,
    run_ragas: bool = False,
    *,
    ragas_samples: int | None = DEFAULT_RAGAS_SAMPLES,
    cases_path: str | Path = FAITHFULNESS_CASES_PATH,
    outcomes_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, object] | None:
    """Run faithfulness evaluation on frozen cases or an explicit sample."""
    from sage.services import get_explanation_services

    resolved_outcomes_path = resolve_faithfulness_case_outcomes_path(
        cases_path,
        outcomes_path=outcomes_path,
    )
    resolved_manifest_path = resolve_faithfulness_cases_manifest_path(
        cases_path,
        manifest_path=manifest_path,
    )
    all_cases = load_faithfulness_cases(
        path=cases_path,
        require_nonempty=True,
    )
    cases, evaluation_scope = _select_case_scope(
        all_cases,
        requested_samples=n_samples,
    )
    run_started_at = datetime.now().astimezone()
    freshness_reference = load_frozen_freshness_reference(
        cases_path=cases_path,
        manifest_path=resolved_manifest_path,
    )
    cases_manifest = load_faithfulness_cases_manifest(
        resolved_manifest_path,
        require_nonempty=True,
    )
    reference_timestamp_ms = freshness_reference["reference_timestamp_ms"]
    coverage_summary = _load_materialization_coverage(
        outcomes_path=resolved_outcomes_path
    )
    retrieval_policy = _infer_retrieval_policy(cases, coverage_summary)

    log_banner(logger, "FAITHFULNESS EVALUATION")
    logger.info(
        "Cases: %d selected / %d available, Target: %s",
        evaluation_scope["selected_case_count"],
        evaluation_scope["available_case_count"],
        FAITHFULNESS_TARGET,
    )
    logger.info(
        "Scope: %s (requested=%s, sample_limited=%s)",
        evaluation_scope["selection_mode"],
        evaluation_scope["requested_samples"],
        evaluation_scope["sample_limited"],
    )
    logger.info("Retrieval profile: %s", retrieval_policy["retrieval_profile"])
    logger.info(
        "Freshness reference: %s (manifest=%s)",
        freshness_reference["reference_date"],
        freshness_reference["manifest_path"],
    )
    if coverage_summary is not None:
        logger.info(
            "Coverage: %.1f%% materialized (%d/%d seed queries)",
            coverage_summary["materialization_rate"] * 100,  # type: ignore[operator]
            coverage_summary["materialized_case_count"],
            coverage_summary["total_queries"],
        )

    log_section(logger, "1. GENERATING EXPLANATIONS")

    explainer, detector = get_explanation_services()
    evaluated_cases = []
    all_explanations = []

    for index, case in enumerate(cases, 1):
        logger.info('[%d/%d] "%s"', index, len(cases), case.query)
        product = case.to_product_score()
        try:
            result = explainer.generate_explanation(
                case.query, product, max_evidence=MAX_EVIDENCE
            )
            evaluated_cases.append(case)
            all_explanations.append(result)
            logger.info("  %s: %s...", product.product_id, result.explanation[:60])
        except Exception:
            logger.exception("  Error generating explanation")

    if not all_explanations:
        logger.warning("No explanations generated")
        return None

    evaluation_scope["evaluated_case_count"] = len(all_explanations)
    evaluation_scope["generation_limited"] = (
        len(all_explanations) < evaluation_scope["selected_case_count"]
    )
    evaluation_scope["evaluated_case_ids"] = [case.case_id for case in evaluated_cases]

    log_section(logger, "2. HHEM HALLUCINATION DETECTION")

    hhem_results = detector.check_batch(
        [(expl.evidence_texts, expl.explanation) for expl in all_explanations]
    )

    for expl, result in zip(all_explanations, hhem_results, strict=True):
        status = "GROUNDED" if not result.is_hallucinated else "HALLUCINATED"
        logger.info("  [%s] %.3f - %s", status, result.score, expl.product_id)

    hhem_scores = [result.score for result in hhem_results]
    n_hallucinated = sum(1 for result in hhem_results if result.is_hallucinated)

    logger.info(
        "HHEM (full-explanation): %d/%d grounded, mean=%.3f",
        len(hhem_results) - n_hallucinated,
        len(hhem_results),
        np.mean(hhem_scores),
    )

    log_section(logger, "3. MULTI-METRIC FAITHFULNESS")

    from sage.services.faithfulness._metrics import compute_multi_metric_faithfulness

    multi_items = [(expl.evidence_texts, expl.explanation) for expl in all_explanations]
    multi_report = compute_multi_metric_faithfulness(multi_items)

    logger.info(
        "Quote verification: %d/%d (%.1f%%)",
        multi_report.quotes_found,
        multi_report.quotes_total,
        multi_report.quote_verification_rate * 100,
    )
    logger.info(
        "Claim-level HHEM:   %.3f avg, %.1f%% pass rate",
        multi_report.claim_level_avg_score,
        multi_report.claim_level_pass_rate * 100,
    )
    logger.info(
        "Full-explanation:   %.3f avg, %.1f%% pass rate (reference only)",
        multi_report.full_explanation_avg_score,
        multi_report.full_explanation_pass_rate * 100,
    )
    adjusted_results = _build_adjusted_results(
        hhem_results,
        [expl.explanation for expl in all_explanations],
        timestamp=datetime.now(),
        n_samples=len(all_explanations),
    )
    logger.info(
        "Adjusted HHEM:    %.3f adjusted, %.1f%% refusal rate",
        adjusted_results["adjusted_pass_rate"],
        adjusted_results["refusal_rate"] * 100,  # type: ignore[operator]
    )
    query_slice_metrics = _build_query_slice_metrics(
        evaluated_cases,
        all_explanations,
        hhem_results,
        reference_timestamp_ms=reference_timestamp_ms,
    )
    case_diagnostics = _build_case_diagnostics(
        evaluated_cases,
        all_explanations,
        hhem_results,
        reference_timestamp_ms=reference_timestamp_ms,
    )
    evidence_guardrails = summarize_evidence_guardrail_reports(
        [row["evidence_guardrails"] for row in case_diagnostics]  # type: ignore[misc]
    )
    freshness_guardrail = summarize_freshness_guardrail_cases(
        [row.get("freshness_guardrail") for row in case_diagnostics]  # type: ignore[misc]
    )
    _log_query_slice_metrics(query_slice_metrics)
    _log_freshness_guardrail(freshness_guardrail)
    run_provenance = build_run_provenance(
        explainer=explainer,
        retrieval_profile=(
            str(retrieval_policy["retrieval_profile"])
            if isinstance(retrieval_policy.get("retrieval_profile"), str)
            else None
        ),
    )
    run_provenance["frozen_case_source"] = {
        "cases_path": str(cases_path),
        "outcomes_path": str(resolved_outcomes_path),
        "manifest_path": str(resolved_manifest_path),
        "retrieval_config": _copy_object_dict(cases_manifest.get("retrieval_config")),
        "gate_config": _copy_object_dict(cases_manifest.get("gate_config")),
    }
    query_bank_identity = _copy_object_dict(cases_manifest.get("query_bank_identity"))

    if not _RAGAS_PROGRESS_PATH.exists():
        _save_ragas_checkpoint(evaluated_cases, all_explanations)
    else:
        logger.info(
            "Skipping checkpoint write — ragas_progress.json exists "
            "(run in progress). Delete it to allow a fresh checkpoint."
        )

    ragas_report = None
    ragas_scope = {
        "enabled": run_ragas,
        "selection_mode": "disabled",
        "selection_policy": None,
        "selection_seed": None,
        "requested_samples": _sample_limit_label(ragas_samples),
        "available_case_count": len(all_explanations),
        "selected_case_count": 0,
        "sample_limited": False,
        "selected_case_ids": [],
    }
    if run_ragas:
        log_section(logger, "4. RAGAS EVALUATION")

        try:
            from sage.services.faithfulness._evaluator import FaithfulnessEvaluator

            ragas_cases, ragas_scope = _select_case_scope(
                evaluated_cases,
                requested_samples=ragas_samples,
            )
            ragas_scope["enabled"] = True
            ragas_scope["available_case_count"] = len(all_explanations)
            ragas_case_ids = {case.case_id for case in ragas_cases}
            ragas_explanations = [
                explanation
                for case, explanation in zip(
                    evaluated_cases,
                    all_explanations,
                    strict=True,
                )
                if case.case_id in ragas_case_ids
            ]
            ragas_scope["selected_case_count"] = len(ragas_explanations)
            ragas_scope["selected_case_ids"] = [case.case_id for case in ragas_cases]
            evaluator = FaithfulnessEvaluator()
            ragas_report = evaluator.evaluate_batch(ragas_explanations)

            logger.info(
                "Faithfulness: %.3f +/- %.3f (scope=%s, n=%d/%d)",
                ragas_report.mean_score,
                ragas_report.std_score,
                ragas_scope["selection_mode"],
                ragas_scope["selected_case_count"],
                ragas_scope["available_case_count"],
            )
            logger.info(
                "Passing: %d/%d", ragas_report.n_passing, ragas_report.n_samples
            )
        except Exception:
            logger.exception("RAGAS evaluation failed")

    timestamp = datetime.now()
    adjusted_results["timestamp"] = timestamp.isoformat()
    results = {
        "timestamp": timestamp.isoformat(),
        "evaluation_run_started_at": run_started_at.isoformat(),
        "n_samples": len(all_explanations),
        "requested_samples": _sample_limit_label(n_samples),
        "retrieval_policy": retrieval_policy,
        "hhem": {
            "mean_score": float(np.mean(hhem_scores)),
            "n_hallucinated": n_hallucinated,
            "hallucination_rate": n_hallucinated / len(hhem_results),
        },
        "multi_metric": {
            "quote_verification_rate": multi_report.quote_verification_rate,
            "quotes_found": multi_report.quotes_found,
            "quotes_total": multi_report.quotes_total,
            "claim_level_pass_rate": multi_report.claim_level_pass_rate,
            "claim_level_avg_score": multi_report.claim_level_avg_score,
            "claim_level_min_score": multi_report.claim_level_min_score,
            "full_explanation_pass_rate": multi_report.full_explanation_pass_rate,
            "full_explanation_avg_score": multi_report.full_explanation_avg_score,
        },
        "target": FAITHFULNESS_TARGET,
        "evaluation_scope": evaluation_scope,
        "adjusted": adjusted_results,
        "run_provenance": run_provenance,
        "evidence_guardrails": evidence_guardrails,
        "freshness_guardrail": freshness_guardrail,
        "evidence_guardrail_methodology": {
            "reference_timestamp_ms": reference_timestamp_ms,
            "reference_date": freshness_reference["reference_date"],
            "reference_source": "faithfulness_cases_manifest",
            "reference_manifest_path": str(freshness_reference["manifest_path"]),
            "negative_review_rule": "rating <= 2.0",
            "old_review_days_threshold": 365,
            "very_old_review_days_threshold": 1095,
        },
        "case_diagnostics": case_diagnostics,
    }
    if query_bank_identity is not None:
        results["query_bank_identity"] = query_bank_identity

    if ragas_report:
        results["ragas"] = {
            "faithfulness_mean": ragas_report.mean_score,
            "faithfulness_std": ragas_report.std_score,
            "per_case": [
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "product_id": case.product_id,
                    "score": r.score,
                    "meets_target": r.meets_target,
                }
                for case, r in zip(
                    ragas_cases,
                    ragas_report.results,
                    strict=True,
                )
            ],
        }
    results["ragas_scope"] = ragas_scope

    if coverage_summary is not None:
        results["coverage"] = {
            "source_query_count": coverage_summary["total_queries"],
            "materialized_case_count": coverage_summary["materialized_case_count"],
            "non_materialized_query_count": coverage_summary[
                "non_materialized_query_count"
            ],
            "queries_with_candidates_count": coverage_summary[
                "queries_with_candidates_count"
            ],
            "insufficient_evidence_count": coverage_summary[
                "insufficient_evidence_count"
            ],
            "no_candidates_retrieved_count": coverage_summary[
                "no_candidates_retrieved_count"
            ],
            "retrieval_error_count": coverage_summary["retrieval_error_count"],
            "materialization_rate": coverage_summary["materialization_rate"],
            "candidate_retrieval_rate": coverage_summary["candidate_retrieval_rate"],
            "gate_pass_rate": coverage_summary["gate_pass_rate"],
            "retrieval_profile": coverage_summary.get("retrieval_profile"),
            "by_retrieval_profile": coverage_summary.get("by_retrieval_profile"),
            "outcome_status_counts": coverage_summary["by_outcome_status"],
            "coverage_note": (
                "Faithfulness metrics are conditional on materialized cases. "
                "Coverage metrics report how much of the full faithfulness_seed "
                "pool reached that evaluation surface."
            ),
        }
        results["evaluation_scope"]["available_materialized_case_count"] = (  # type: ignore[index]
            coverage_summary["materialized_case_count"]
        )
        results["evaluation_scope"]["coverage_sample_limited"] = (  # type: ignore[index]
            results["evaluation_scope"]["selected_case_count"]  # type: ignore[index]
            < coverage_summary["materialized_case_count"]
        )

    if query_slice_metrics:
        results["query_slice_methodology"] = {
            "report_only": True,
            "slice_descriptions": QUERY_SLICE_DESCRIPTIONS,
            "note": (
                "These slices are simple heuristics on query text. They do not "
                "change runtime behavior; they surface whether wins are hiding "
                "regressions on recency-sensitive or complaint-oriented asks. "
                "Each slice now also carries refusal and evidence-trust "
                "diagnostics derived from the actual frozen evidence bundle."
            ),
        }
        results["query_slice_metrics"] = query_slice_metrics

    results["ragas_limitations"] = {
        "metrics_available": ["faithfulness"],
        "metrics_unavailable": {
            "answer_relevancy": (
                "Requires embeddings model; RAGAS doesn't support Anthropic as "
                "embeddings provider"
            ),
            "context_precision": (
                "Requires ground-truth reference answers per query (not available)"
            ),
            "context_recall": (
                "Requires ground-truth reference answers per query (not available)"
            ),
        },
        "primary_metric": "claim_level_hhem",
        "rationale": (
            f"Claim-level HHEM ({multi_report.claim_level_avg_score:.1%}) is more "
            "reliable than full-explanation RAGAS for citation-heavy explanations"
        ),
    }
    adjusted_results["run_provenance"] = run_provenance
    if query_bank_identity is not None:
        adjusted_results["query_bank_identity"] = query_bank_identity

    ts_file = save_results(results, "faithfulness")
    logger.info("Saved: %s", ts_file)
    adjusted_file = save_results(adjusted_results, "adjusted_faithfulness")
    logger.info("Saved: %s", adjusted_file)

    return results


_RAGAS_CHECKPOINT_PATH = Path(RESULTS_DIR) / "ragas_checkpoint.json"


def _save_ragas_checkpoint(
    evaluated_cases: list,
    all_explanations: list,
) -> None:
    checkpoint = {
        "cases": [
            {
                "case_id": case.case_id,
                "query": case.query,
                "product_id": case.product_id,
                "explanation": expl.explanation,
                "evidence_texts": expl.evidence_texts,
            }
            for case, expl in zip(evaluated_cases, all_explanations, strict=True)
        ]
    }
    with open(_RAGAS_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info("Checkpoint saved: %s", _RAGAS_CHECKPOINT_PATH)


_RAGAS_PROGRESS_PATH = Path(RESULTS_DIR) / "ragas_progress.json"


def run_ragas_from_checkpoint(
    checkpoint_path: str | Path | None = None,
    ragas_samples: int | None = DEFAULT_RAGAS_SAMPLES,
) -> None:
    """Run RAGAS only, using explanations saved by a prior run's checkpoint.

    Saves progress after every case — safe to interrupt and resume at any
    point. Re-running reloads completed scores from ragas_progress.json and
    skips already-scored cases.
    """
    from sage.services.faithfulness._evaluator import FaithfulnessEvaluator

    path = (
        Path(checkpoint_path).resolve() if checkpoint_path else _RAGAS_CHECKPOINT_PATH
    )
    from sage.config import DATA_DIR

    if not str(path).startswith(str(DATA_DIR)):
        raise ValueError(
            f"Checkpoint path {path} is outside DATA_DIR ({DATA_DIR}). "
            "Pass a path within the project data directory."
        )
    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)

    cases_data: list[dict] = checkpoint["cases"]
    if ragas_samples is not None and ragas_samples < len(cases_data):
        cases_data = cases_data[:ragas_samples]

    # Load any progress saved by a previous interrupted run.
    completed: dict[str, dict] = {}
    if _RAGAS_PROGRESS_PATH.exists():
        with open(_RAGAS_PROGRESS_PATH, encoding="utf-8") as f:
            completed = json.load(f)

    log_banner(logger, "RAGAS EVALUATION (FROM CHECKPOINT)")
    logger.info("Loaded %d cases from %s", len(cases_data), path)
    logger.info("Target: %.2f", FAITHFULNESS_TARGET)
    if completed:
        logger.info("Resuming: %d/%d already scored", len(completed), len(cases_data))

    # Single event loop for all cases — one AsyncAnthropic client, one
    # connection pool, one TLS handshake reused across all 120 API calls.
    async def _score_all() -> tuple[list[dict], list[float]]:
        evaluator = FaithfulnessEvaluator()
        per_case_inner: list[dict] = []
        scores_inner: list[float] = []
        n = len(cases_data)

        for i, c in enumerate(cases_data, 1):
            case_id = c["case_id"]
            if case_id in completed:
                cached = completed[case_id]
                score = cached["score"]
                meets = cached["meets_target"]
                logger.info("[%d/%d] %.3f (cached) — %s", i, n, score, c["query"])
            else:
                result = await evaluator.evaluate_single_async(
                    c["query"], c["explanation"], c["evidence_texts"]
                )
                score = result.score
                meets = result.meets_target
                completed[case_id] = {
                    "score": score,
                    "query": c["query"],
                    "product_id": c["product_id"],
                    "meets_target": meets,
                }
                with open(_RAGAS_PROGRESS_PATH, "w", encoding="utf-8") as f:
                    json.dump(completed, f, indent=2)
                logger.info("[%d/%d] %.3f — %s", i, n, score, c["query"])

            per_case_inner.append(
                {
                    "case_id": case_id,
                    "query": c["query"],
                    "product_id": c["product_id"],
                    "score": score,
                    "meets_target": meets,
                }
            )
            scores_inner.append(score)

        return per_case_inner, scores_inner

    per_case, scores = asyncio.run(_score_all())

    scores_arr = np.array(scores)
    mean = float(np.mean(scores_arr))
    std = float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0
    n_passing = sum(1 for s in scores if s >= FAITHFULNESS_TARGET)

    logger.info("Faithfulness: %.3f +/- %.3f (n=%d)", mean, std, len(scores))
    logger.info("Passing: %d/%d", n_passing, len(scores))

    results = {
        "faithfulness_mean": mean,
        "faithfulness_std": std,
        "n_samples": len(scores),
        "n_passing": n_passing,
        "pass_rate": n_passing / len(scores) if scores else 0.0,
        "target": FAITHFULNESS_TARGET,
        "source_checkpoint": str(path),
        "per_case": per_case,
    }
    ts_file = save_results(results, "ragas_only")
    logger.info("Saved: %s", ts_file)

    _RAGAS_PROGRESS_PATH.unlink(missing_ok=True)


def run_grounding_delta(*, cases_path: str | Path = FAITHFULNESS_CASES_PATH) -> None:
    """
    Compare HHEM scores WITH vs WITHOUT evidence grounding.

    This is an experimental diagnostic only.

    It is intentionally excluded from the default evaluation workflow
    and reporting surfaces because it does not yet share the same frozen-case
    scope and production explainer path as the main faithfulness benchmark.
    """
    from sage.adapters.llm import get_llm_client
    from sage.services import get_explanation_services

    log_banner(logger, "EXPERIMENTAL GROUNDING DELTA")
    logger.warning(
        "This is an experimental diagnostic, not a canonical evaluation metric."
    )
    logger.info("Comparing hallucination rates WITH vs WITHOUT evidence grounding")

    cases = load_faithfulness_cases(
        path=cases_path,
        limit=10,
        require_nonempty=True,
    )
    _, detector = get_explanation_services()
    llm = get_llm_client()

    with_evidence = []
    without_evidence = []

    from sage.utils import sanitize_query

    for index, case in enumerate(cases, 1):
        logger.info('[%d/%d] "%s"', index, len(cases), case.query)
        evidence_texts = [item.text for item in case.evidence[:MAX_EVIDENCE]]

        if not evidence_texts:
            continue

        safe_query = sanitize_query(case.query)
        system_prompt = "You are a helpful product recommendation assistant."
        grounded_user = f"""Based on customer reviews, explain why this product is good for: "{safe_query}"

EVIDENCE FROM REVIEWS:
{chr(10).join(f"- {text}" for text in evidence_texts[:3])}

Write a brief 2-3 sentence recommendation based ONLY on the evidence above."""

        try:
            grounded_response, _ = llm.generate(system_prompt, grounded_user)
            grounded_hhem = detector.check_explanation(
                evidence_texts, grounded_response
            )
            with_evidence.append(grounded_hhem.score)
            logger.info("  WITH evidence: %.3f", grounded_hhem.score)
        except Exception:
            logger.exception("  Error with grounded generation")
            continue

        ungrounded_user = f"""Recommend a product for: "{safe_query}"

Write a brief 2-3 sentence recommendation. You may make reasonable assumptions about the product."""

        try:
            ungrounded_response, _ = llm.generate(system_prompt, ungrounded_user)
            ungrounded_hhem = detector.check_explanation(
                evidence_texts, ungrounded_response
            )
            without_evidence.append(ungrounded_hhem.score)
            logger.info("  WITHOUT evidence: %.3f", ungrounded_hhem.score)
        except Exception:
            logger.exception("  Error with ungrounded generation")

    log_banner(logger, "EXPERIMENTAL GROUNDING DELTA RESULTS")

    if with_evidence and without_evidence:
        with_mean = np.mean(with_evidence)
        without_mean = np.mean(without_evidence)
        delta = with_mean - without_mean

        logger.info("Samples: %d", min(len(with_evidence), len(without_evidence)))
        logger.info("WITH evidence (grounded):    %.3f mean HHEM", with_mean)
        logger.info("WITHOUT evidence (halluc):   %.3f mean HHEM", without_mean)
        logger.info("Delta (grounding benefit):   +%.3f", delta)
        logger.info(
            "Interpretation: Grounding %s hallucination by %.1f%%",
            "reduces" if delta > 0 else "increases",
            abs(delta) * 100,
        )

        results = {
            "n_samples": min(len(with_evidence), len(without_evidence)),
            "with_evidence_mean": float(with_mean),
            "without_evidence_mean": float(without_mean),
            "delta": float(delta),
            "with_evidence_scores": with_evidence,
            "without_evidence_scores": without_evidence,
        }
        ts_file = save_results(results, "grounding_delta_experimental")
        logger.info("Saved: %s", ts_file)
    else:
        logger.warning("Not enough samples for comparison")
