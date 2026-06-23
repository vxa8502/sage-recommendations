"""Tests for scripts.evaluate_retrieval_configs decision policy semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.evaluate_retrieval_configs as retrieval_script


def _metrics(
    *,
    ndcg_at_10: float,
    hit_at_10: float,
    mrr: float,
    recall_at_10: float,
    precision_at_10: float = 0.0,
) -> dict[str, float]:
    return {
        "ndcg_at_10": ndcg_at_10,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
        "recall_at_10": recall_at_10,
        "precision_at_10": precision_at_10,
    }


def test_parse_args_uses_role_specific_default_outputs():
    fit_args = retrieval_script.parse_args(
        [
            "--comparison-role",
            "fit",
            "--candidate-min-rating",
            "4",
        ]
    )
    holdout_args = retrieval_script.parse_args(
        [
            "--comparison-role",
            "holdout",
            "--candidate-config-path",
            "data/retrieval/retrieval_fit.analysis.json",
        ]
    )

    assert fit_args.output == retrieval_script.DEFAULT_FIT_OUTPUT
    assert fit_args.subsets == list(retrieval_script.DEFAULT_FIT_SUBSETS)
    assert holdout_args.output == retrieval_script.DEFAULT_HOLDOUT_OUTPUT
    assert holdout_args.subsets == list(retrieval_script.DEFAULT_HOLDOUT_SUBSETS)


def test_holdout_retains_baseline_when_ndcg_gain_is_not_material():
    decision = retrieval_script._recommend_winner(
        baseline_metrics=_metrics(
            ndcg_at_10=0.2000,
            hit_at_10=0.3300,
            mrr=0.1500,
            recall_at_10=0.2800,
        ),
        candidate_metrics=_metrics(
            ndcg_at_10=0.2080,
            hit_at_10=0.3400,
            mrr=0.1520,
            recall_at_10=0.2850,
        ),
        comparison_role="holdout",
    )

    assert decision["decision_status"] == "baseline_retained"
    assert decision["recommended_config"] == "baseline"
    assert decision["primary_metric"] == "ndcg_at_10"
    assert decision["primary_metric_delta"] == 0.008
    assert decision["primary_metric_passed"] is False
    assert decision["guardrails_passed"] is True


def test_holdout_retains_baseline_when_guardrail_regresses_too_far():
    decision = retrieval_script._recommend_winner(
        baseline_metrics=_metrics(
            ndcg_at_10=0.2000,
            hit_at_10=0.3300,
            mrr=0.1500,
            recall_at_10=0.2800,
        ),
        candidate_metrics=_metrics(
            ndcg_at_10=0.2140,
            hit_at_10=0.3180,
            mrr=0.1490,
            recall_at_10=0.2790,
        ),
        comparison_role="holdout",
    )

    assert decision["decision_status"] == "baseline_retained"
    assert decision["recommended_config"] == "baseline"
    assert decision["primary_metric_passed"] is True
    assert decision["guardrails_passed"] is False
    assert decision["guardrail_results"]["hit_at_10"]["passed"] is False
    assert decision["guardrail_results"]["mrr"]["passed"] is True


def test_holdout_promotes_candidate_when_primary_metric_and_guardrails_pass():
    decision = retrieval_script._recommend_winner(
        baseline_metrics=_metrics(
            ndcg_at_10=0.2000,
            hit_at_10=0.3300,
            mrr=0.1500,
            recall_at_10=0.2800,
        ),
        candidate_metrics=_metrics(
            ndcg_at_10=0.2125,
            hit_at_10=0.3240,
            mrr=0.1470,
            recall_at_10=0.2740,
        ),
        comparison_role="holdout",
    )

    assert decision["decision_status"] == "candidate_promoted"
    assert decision["recommended_config"] == "candidate"
    assert decision["promotion_eligible"] is True
    assert decision["primary_metric_passed"] is True
    assert decision["guardrails_passed"] is True


def test_fit_decision_uses_non_promotion_status_labels():
    decision = retrieval_script._recommend_winner(
        baseline_metrics=_metrics(
            ndcg_at_10=0.2000,
            hit_at_10=0.3300,
            mrr=0.1500,
            recall_at_10=0.2800,
        ),
        candidate_metrics=_metrics(
            ndcg_at_10=0.2125,
            hit_at_10=0.3240,
            mrr=0.1470,
            recall_at_10=0.2740,
        ),
        comparison_role="fit",
    )

    assert decision["decision_scope"] == "fit_screen"
    assert decision["promotion_eligible"] is False
    assert decision["decision_status"] == "candidate_preferred_for_holdout"


class _FakeReport:
    def __init__(self, metrics: dict[str, float]):
        self._metrics = metrics

    def to_dict(self) -> dict[str, float]:
        return dict(self._metrics)


def test_main_writes_explicit_holdout_policy(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "retrieval_holdout.analysis.json"
    calls = {"count": 0}
    corpus_alignment = {
        "status": "aligned",
        "corpus_fingerprint": "fingerprint-123",
        "collection_points_count": 2,
    }

    monkeypatch.setattr(
        retrieval_script,
        "_current_retrieval_config",
        lambda: retrieval_script.RetrievalConfig(
            aggregation="max",
            min_rating=None,
            retrieval_profile="eval_unfiltered",
        ),
    )
    monkeypatch.setattr(
        retrieval_script,
        "build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 2,
        },
    )
    monkeypatch.setattr(
        retrieval_script,
        "load_eval_cases_from_query_bank",
        lambda *_args, **_kwargs: [
            SimpleNamespace(query_id="qb_001"),
            SimpleNamespace(query_id="qb_002"),
        ],
    )

    def fake_evaluate(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                _FakeReport(
                    _metrics(
                        ndcg_at_10=0.20004,
                        hit_at_10=0.3300,
                        mrr=0.1500,
                        recall_at_10=0.2800,
                        precision_at_10=0.1000,
                    )
                ),
                [
                    {
                        "recommended_product_ids": ["P1"],
                        "query_slice_tags": [],
                        "metrics": {
                            "ndcg": 0.2000,
                            "hit": 0.3300,
                            "mrr": 0.1500,
                            "precision": 0.1000,
                            "recall": 0.2800,
                            "diversity": 0.0,
                            "novelty": 0.0,
                        },
                    }
                ],
            )
        return (
            _FakeReport(
                _metrics(
                    ndcg_at_10=0.21257,
                    hit_at_10=0.3240,
                    mrr=0.1470,
                    recall_at_10=0.2740,
                    precision_at_10=0.1100,
                )
            ),
            [
                {
                    "recommended_product_ids": ["P1"],
                    "query_slice_tags": [],
                    "metrics": {
                        "ndcg": 0.2125,
                        "hit": 0.3240,
                        "mrr": 0.1470,
                        "precision": 0.1100,
                        "recall": 0.2740,
                        "diversity": 0.0,
                        "novelty": 0.0,
                    },
                }
            ],
        )

    monkeypatch.setattr(
        retrieval_script,
        "evaluate_recommendations_with_details",
        fake_evaluate,
    )
    monkeypatch.setattr(
        retrieval_script,
        "assert_corpus_alignment",
        lambda: corpus_alignment,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval_configs.py",
            "--comparison-role",
            "holdout",
            "--output",
            str(output_path),
            "--candidate-min-rating",
            "4",
        ],
    )

    retrieval_script.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    decision_policy = payload["methodology"]["decision_policy"]
    recommendation = payload["summary"]["recommendation"]

    assert payload["corpus_alignment"] == corpus_alignment
    assert payload["summary"]["baseline_metrics"]["ndcg_at_10"] == pytest.approx(
        0.20004
    )
    assert payload["summary"]["candidate_metrics"]["ndcg_at_10"] == pytest.approx(
        0.21257
    )
    assert decision_policy["policy_version"] == "retrieval_ndcg_guardrails_v1"
    assert decision_policy["primary_metric"] == "ndcg_at_10"
    assert decision_policy["minimum_material_improvement"] == 0.01
    assert decision_policy["guardrail_max_regression"] == {
        "mrr": 0.005,
        "hit_at_10": 0.01,
        "recall_at_10": 0.01,
    }
    assert recommendation["decision_status"] == "candidate_promoted"
    assert recommendation["recommended_config"] == "candidate"


def test_main_exits_when_corpus_alignment_fails(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "retrieval_fit.analysis.json"

    monkeypatch.setattr(
        retrieval_script,
        "_current_retrieval_config",
        lambda: retrieval_script.RetrievalConfig(
            aggregation="max",
            min_rating=None,
            retrieval_profile="eval_unfiltered",
        ),
    )
    monkeypatch.setattr(
        retrieval_script,
        "build_query_bank_identity",
        lambda _path: {
            "query_bank_path": "data/query_bank/query_bank.jsonl",
            "query_bank_sha256": "bank-sha",
            "query_bank_row_count": 2,
        },
    )
    monkeypatch.setattr(
        retrieval_script,
        "assert_corpus_alignment",
        lambda: (_ for _ in ()).throw(
            retrieval_script.CorpusAlignmentError("fingerprint mismatch")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval_configs.py",
            "--comparison-role",
            "fit",
            "--output",
            str(output_path),
            "--candidate-min-rating",
            "4",
        ],
    )

    with pytest.raises(SystemExit, match="corpus-aligned Qdrant collection"):
        retrieval_script.main()
