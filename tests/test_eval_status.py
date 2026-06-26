"""Tests for shared evaluation status surfaces."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from sage.cli.eval_status import build_eval_status


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_eval_artifacts(
    root: Path,
    *,
    ndcg_at_10: float = 0.20,
    faithfulness_score: float = 0.90,
    sample_limited: bool = False,
) -> None:
    _write_json(
        root / "eval_natural_queries_latest.json",
        {
            "primary_metrics": {
                "ndcg_at_10": ndcg_at_10,
                "hit_at_10": 0.33,
                "mrr": 0.16,
            },
            "experiments": {
                "baselines": {
                    "random": {"ndcg_at_10": 0.000},
                    "item_knn": {"ndcg_at_10": 0.134},
                }
            },
        },
    )
    _write_json(
        root / "faithfulness_latest.json",
        {
            "multi_metric": {
                "claim_level_avg_score": faithfulness_score,
            },
            "evaluation_scope": {
                "sample_limited": sample_limited,
                "generation_limited": False,
            },
            "target": 0.85,
        },
    )
    _write_json(
        root / "adjusted_faithfulness_latest.json",
        {
            "adjusted_pass_rate": 0.95,
            "n_total": 10,
        },
    )
    _write_json(
        root / "boundary_behavior_latest.json",
        {
            "boundary_guardrail": {
                "status": "pass",
                "violations": [],
            }
        },
    )
    _write_json(
        root / "load_test_latest.json",
        {
            "headline_metric": {
                "value_ms": 320.0,
                "pass": True,
            }
        },
    )


def test_eval_status_passes_when_required_artifacts_are_complete(tmp_path: Path):
    _write_complete_eval_artifacts(tmp_path)

    status = build_eval_status(results_dir=tmp_path)

    assert status["execution_complete"] is True
    assert status["safety_green"] is True
    assert status["reportable_green"] is True
    assert status["reportable_status"] == "PASS  [reportable-green]"


def test_eval_status_withholds_reportable_when_retrieval_floor_is_missed(
    tmp_path: Path,
):
    _write_complete_eval_artifacts(tmp_path, ndcg_at_10=0.05)

    status = build_eval_status(results_dir=tmp_path)

    assert status["execution_complete"] is True
    assert status["reportable_green"] is False
    assert any(
        "NDCG@10=0.050 < 0.100" in reason for reason in status["reportable_reasons"]
    )


def test_eval_status_withholds_reportable_for_sampled_faithfulness_run(
    tmp_path: Path,
):
    _write_complete_eval_artifacts(tmp_path, sample_limited=True)

    status = build_eval_status(results_dir=tmp_path)

    assert status["execution_complete"] is True
    assert status["reportable_green"] is False
    assert any("sampled" in reason.lower() for reason in status["reportable_reasons"])


def test_eval_status_detects_stale_current_run_artifact(tmp_path: Path):
    _write_complete_eval_artifacts(tmp_path)
    stale_path = tmp_path / "load_test_latest.json"
    stale_time = datetime.now().timestamp() - 7200
    os.utime(stale_path, (stale_time, stale_time))

    status = build_eval_status(
        results_dir=tmp_path,
        run_started_at=datetime.now() - timedelta(minutes=5),
    )

    assert status["execution_complete"] is False
    assert any(
        "was not refreshed during the current evaluation run" in reason
        for reason in status["execution_reasons"]
    )
    assert status["reportable_green"] is False
