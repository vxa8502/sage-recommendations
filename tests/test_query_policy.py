"""Tests for the deterministic pre-retrieval query policy."""

from __future__ import annotations

import pytest

from sage.data.query_bank.sources.boundary import load_manual_boundary_queries
from sage.services.query_policy import QUERY_POLICY_VERSION, evaluate_query_policy


@pytest.mark.parametrize(
    "query",
    [
        "wireless headphones for working out",
        "monitor for programming",
        "latest usb 3.1 gen 2 hub",
        "phone case for iPhone",
    ],
)
def test_query_policy_allows_in_scope_electronics_queries(query: str):
    decision = evaluate_query_policy(query)

    assert decision.action == "allow"
    assert decision.observed_behavior == "answer"
    assert decision.terminal is False


def test_query_policy_clarifies_empty_queries():
    decision = evaluate_query_policy("  !!!  ")

    assert decision.action == "clarify"
    assert decision.reason_code == "ambiguous_query"
    assert decision.matched_terms == ()
    assert decision.terminal is True


def test_query_policy_decision_serializes_public_contract_explicitly():
    decision = evaluate_query_policy("monitor with the lowest carbon footprint")

    assert decision.to_dict() == {
        "action": "refuse",
        "observed_behavior": "refuse",
        "reason_code": "unsupported_attribute_claim",
        "message": (
            "I cannot provide a reliable recommendation for that claim from "
            "customer review evidence alone."
        ),
        "matched_terms": ["carbon footprint"],
        "terminal": True,
        "policy_version": QUERY_POLICY_VERSION,
    }


def test_query_policy_covers_checked_in_manual_boundary_source():
    for row in load_manual_boundary_queries():
        decision = evaluate_query_policy(row.text)

        if row.evaluation_surface == "runtime_e2e":
            assert decision.action == "allow", row.text
            assert decision.observed_behavior == "answer", row.text
            assert decision.terminal is False, row.text
            continue

        assert decision.terminal is True, row.text
        assert decision.reason_code == row.boundary_type, row.text
        assert decision.message, row.text
        if row.expected_behavior == "hedge_or_refuse":
            assert decision.observed_behavior in {"hedge", "refuse"}, row.text
        else:
            assert decision.observed_behavior == row.expected_behavior, row.text


def test_query_policy_does_not_treat_valid_use_cases_as_ambiguous():
    decision = evaluate_query_policy("headphones for my kid")

    assert decision.action == "allow"
    assert decision.terminal is False


def test_query_policy_dedupes_equivalent_matched_terms():
    decision = evaluate_query_policy("safest webcam for all-day eye comfort")

    assert decision.matched_terms == ("all day eye comfort", "eye comfort", "safest")
    assert decision.to_dict()["matched_terms"] == [
        "all day eye comfort",
        "eye comfort",
        "safest",
    ]


def test_query_policy_reports_overlapping_matches_by_specificity():
    decision = evaluate_query_policy(
        "which gaming mouse has the fewest complaints about double-click complaints"
    )

    assert decision.matched_terms == (
        "double click complaints",
        "fewest complaints",
        "double click",
        "complaints",
    )


def test_query_policy_dedupes_equivalent_negative_problem_terms():
    decision = evaluate_query_policy(
        "which earbuds have the shakiest long-term durability record"
    )

    assert decision.matched_terms == ("shakiest long term durability",)
