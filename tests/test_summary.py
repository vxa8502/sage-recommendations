"""Tests for scripts.summary."""

from __future__ import annotations

import json

from scripts import summary


def test_summary_omits_grounding_delta_section(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(summary, "RESULTS_DIR", tmp_path)

    summary.main()

    output = capsys.readouterr().out
    assert "Evaluation Status:" in output
    assert "Latest Artifacts: NOT STARTED" in output
    assert "Execution:       NOT STARTED" in output
    assert "Safety Green:    NOT AVAILABLE" in output
    assert "Reportable:      WITHHELD" in output
    assert "Grounding Delta" not in output
    assert "Explanation Faithfulness:" in output
    assert "Quality Gate (Refusals):" in output


def test_summary_surfaces_boundary_guardrail_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(summary, "RESULTS_DIR", tmp_path)
    (tmp_path / "boundary_behavior_latest.json").write_text(
        json.dumps(
            {
                "summary": {
                    "acceptable_match_rate": 0.6,
                    "acceptable_matches": 15,
                    "total_queries": 25,
                    "refusal_required_false_accept_count": 2,
                    "ambiguous_clarify_rate": 0.5,
                    "boundary_safe_behavior_rate": 0.7,
                    "freshness_sensitive_refusal_rate": 0.0,
                    "freshness_guardrail": {
                        "promotion_status": "blocked",
                        "recency_sensitive_case_count": 3,
                        "coverage_min_recency_sensitive_cases": 3,
                        "safe_rate": 0.6667,
                    },
                },
                "boundary_guardrail": {
                    "status": "fail",
                    "violations": [
                        {
                            "message": (
                                "Ambiguous queries are not producing enough "
                                "clarifications."
                            )
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    summary.main()

    output = capsys.readouterr().out
    assert "Latest Artifacts: PARTIAL" in output
    assert "Execution:       INCOMPLETE" in output
    assert "Safety Green:    FAIL" in output
    assert "Boundary Guard: FAIL" in output
    assert "Fresh Guard:    BLOCKED" in output
    assert "Guardrail Hit:  Ambiguous queries" in output


def test_summary_renders_zero_denominator_freshness_rate_as_unavailable(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(summary, "RESULTS_DIR", tmp_path)
    (tmp_path / "boundary_behavior_latest.json").write_text(
        json.dumps(
            {
                "summary": {
                    "acceptable_match_rate": 1.0,
                    "acceptable_matches": 25,
                    "total_queries": 25,
                    "refusal_required_false_accept_count": 0,
                    "ambiguous_clarify_rate": 1.0,
                    "boundary_safe_behavior_rate": 1.0,
                    "freshness_sensitive_refusal_rate": None,
                    "freshness_guardrail": {
                        "promotion_status": "insufficient_coverage",
                        "recency_sensitive_case_count": 0,
                        "coverage_min_recency_sensitive_cases": 3,
                        "safe_rate": None,
                    },
                },
                "boundary_guardrail": {
                    "status": "insufficient_coverage",
                    "violations": [],
                },
            }
        ),
        encoding="utf-8",
    )

    summary.main()

    output = capsys.readouterr().out
    assert "Fresh Refusal:  unavailable" in output
    assert "Fresh Safe:     unavailable" in output
