"""Tests for the load-test reporting helpers."""

from scripts.load_test import (
    _build_results,
    _evaluate_response_quality,
    _select_headline_metric,
    _summarize_phase,
)


class TestEvaluateResponseQuality:
    def test_grounded_success_requires_grounded_verified_recommendations(self):
        payload = {
            "recommendations": [
                {
                    "explanation": '"Battery lasts all day" [review_1]',
                    "citations_verified": True,
                    "confidence": {"is_grounded": True},
                }
            ]
        }

        result = _evaluate_response_quality(payload, explain=True)

        assert result["grounded_success"] is True
        assert result["refusal_aware_success"] is True

    def test_refusal_counts_only_in_refusal_aware_mode(self):
        payload = {
            "recommendations": [
                {
                    "explanation": "I cannot recommend this product based on the available review evidence.",
                    "citations_verified": None,
                    "confidence": {"is_grounded": False},
                }
            ]
        }

        result = _evaluate_response_quality(payload, explain=True)

        assert result["grounded_success"] is False
        assert result["refusal_aware_success"] is True

    def test_policy_decision_counts_as_refusal_aware_success(self):
        payload = {
            "recommendations": [],
            "policy_decision": {
                "observed_behavior": "clarify",
                "reason_code": "ambiguous_query",
            },
        }

        result = _evaluate_response_quality(payload, explain=True)

        assert result["grounded_success"] is False
        assert result["refusal_aware_success"] is True


class TestSummarizePhase:
    def test_summarizes_cache_and_quality_rates(self):
        samples = [
            {
                "status": 200,
                "client_ms": 90.0,
                "server_ms": 60.0,
                "network_overhead_ms": 30.0,
                "server_timing_header_present": True,
                "cache_result_header_present": True,
                "cache_result": "exact",
                "grounded_success": True,
                "refusal_aware_success": True,
            },
            {
                "status": 200,
                "client_ms": 220.0,
                "server_ms": 180.0,
                "network_overhead_ms": 40.0,
                "server_timing_header_present": True,
                "cache_result_header_present": True,
                "cache_result": "miss",
                "grounded_success": False,
                "refusal_aware_success": True,
            },
            {
                "status": 503,
                "client_ms": None,
                "server_ms": None,
                "network_overhead_ms": None,
                "cache_result": "unknown",
                "grounded_success": None,
                "refusal_aware_success": None,
            },
        ]

        summary = _summarize_phase(samples, explain=True)

        assert summary["successful"] == 2
        assert summary["errors"] == 1
        assert summary["cache_results"] == {"exact": 1, "miss": 1}
        assert summary["cache_hit_rate"] == 0.5
        assert summary["header_presence"]["x_response_time_ms"] == {
            "present": 2,
            "missing": 0,
        }
        assert summary["header_presence"]["x_cache_result"] == {
            "present": 2,
            "missing": 0,
        }
        assert summary["cache_observability"]["available"] is True
        assert summary["api_quality"]["evaluated_requests"] == 2
        assert summary["api_quality"]["grounded_success_rate"] == 0.5
        assert summary["api_quality"]["refusal_aware_success_rate"] == 1.0

    def test_reports_missing_cache_header_observability_explicitly(self):
        samples = [
            {
                "status": 200,
                "client_ms": 80.0,
                "server_ms": 20.0,
                "network_overhead_ms": 60.0,
                "server_timing_header_present": True,
                "cache_result_header_present": False,
                "cache_result": "unknown",
                "grounded_success": True,
                "refusal_aware_success": True,
            }
        ]

        summary = _summarize_phase(samples, explain=True)

        assert summary["cache_results"] == {}
        assert summary["cache_hit_rate"] is None
        assert summary["header_presence"]["x_cache_result"] == {
            "present": 0,
            "missing": 1,
        }
        assert summary["cache_observability"] == {
            "available": False,
            "successful_responses": 1,
            "header_present_responses": 0,
            "missing_header_responses": 1,
            "reason": "X-Cache-Result header absent on successful responses",
        }


class TestSelectHeadlineMetric:
    def test_prefers_server_p95_when_available(self):
        measured_summary = {
            "server_latency_ms": {"p95": 180.0},
            "client_latency_ms": {"p95": 260.0},
        }

        headline = _select_headline_metric(measured_summary, target_ms=500.0)

        assert headline["name"] == "steady_state_server_p95_ms"
        assert headline["value_ms"] == 180.0
        assert headline["pass"] is True

    def test_falls_back_to_client_p95_without_server_metric(self):
        measured_summary = {
            "server_latency_ms": None,
            "client_latency_ms": {"p95": 260.0},
        }

        headline = _select_headline_metric(measured_summary, target_ms=200.0)

        assert headline["name"] == "steady_state_client_p95_ms"
        assert headline["value_ms"] == 260.0
        assert headline["pass"] is False


class TestBuildResults:
    def test_cache_hits_is_none_when_cache_metadata_is_absent(self):
        measured_samples = [
            {
                "status": 200,
                "client_ms": 80.0,
                "server_ms": 20.0,
                "network_overhead_ms": 60.0,
                "server_timing_header_present": True,
                "cache_result_header_present": False,
                "cache_result": "unknown",
                "grounded_success": True,
                "refusal_aware_success": True,
            }
        ]

        results = _build_results(
            warmup_samples=[],
            measured_samples=measured_samples,
            config={"explain": True},
            target_ms=500.0,
        )

        assert results["cache_hits"] is None
        assert results["cache_hit_rate"] is None
        assert results["measured"]["cache_observability"]["available"] is False
