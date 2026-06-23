"""Tests for sage.data.query_bank.sources.candidates."""

import json
from contextlib import contextmanager

import pytest

import sage.data.query_bank.sources.candidates as query_candidates_module
from sage.data.query_bank.sources.candidates import (
    QueryCandidate,
    build_esci_query_candidates,
    build_query_bank_rows_from_candidates,
    load_query_candidates,
    save_query_candidates,
)


def test_build_esci_query_candidates_aggregates_and_filters(tmp_path):
    path = tmp_path / "esci.tsv"
    path.write_text(
        "\n".join(
            [
                "query_id\tquery\tproduct_locale\tesci_label\tlarge_version",
                "q1\twireless headphones\tus\tE\t1",
                "q1\twireless headphones\tus\tS\t1",
                "q2\tphone charger\tus\tC\t0",
                "q3\ttablet stand\tuk\tE\t1",
                "q4\t\tus\tE\t1",
            ]
        ),
        encoding="utf-8",
    )

    candidates = build_esci_query_candidates(
        path,
        locale="us",
        require_large_version=True,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.text == "wireless headphones"
    assert candidate.record_count == 2
    assert candidate.labels_observed == ("E", "S")
    assert candidate.locales_observed == ("us",)
    assert candidate.source_file == "esci.tsv"
    assert candidate.source_ref == "esci.tsv:query_id=q1"


def test_query_candidates_round_trip(tmp_path):
    path = tmp_path / "query_candidates.jsonl"
    candidates = [
        QueryCandidate(
            candidate_id="qc_0001",
            text="wireless headphones",
            source_type="amazon_esci",
            domain="electronics",
            source_file="esci.tsv",
            source_ref="esci.tsv:row2",
            locale_hint="us",
            record_count=2,
            labels_observed=("E", "S"),
            locales_observed=("us",),
            notes="candidate note",
        )
    ]

    save_query_candidates(candidates, path)
    loaded = load_query_candidates(path)

    assert loaded == candidates


def test_load_query_candidates_requires_json_object_rows(tmp_path):
    path = tmp_path / "query_candidates.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_query_candidates(path)


def test_load_query_candidates_validates_record_count(tmp_path):
    path = tmp_path / "query_candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "qc_0001",
                "text": "wireless headphones",
                "source_type": "amazon_esci",
                "record_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'record_count' must be >= 1"):
        load_query_candidates(path)


def test_load_query_candidates_validates_string_list_metadata(tmp_path):
    path = tmp_path / "query_candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "qc_0001",
                "text": "wireless headphones",
                "source_type": "amazon_esci",
                "record_count": 1,
                "labels_observed": "E",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'labels_observed' must be a list"):
        load_query_candidates(path)


def test_load_query_candidates_rejects_non_string_optional_text(tmp_path):
    path = tmp_path / "query_candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_id": "qc_0001",
                "text": "wireless headphones",
                "source_type": "amazon_esci",
                "domain": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'domain' must be a string or null"):
        load_query_candidates(path)


def test_build_esci_query_candidates_supports_parquet_sources(tmp_path, monkeypatch):
    path = tmp_path / "esci.parquet"
    path.write_text("placeholder", encoding="utf-8")

    @contextmanager
    def fake_open_parquet_rows(source_path):
        assert source_path == path
        yield query_candidates_module._SourceRows(
            fieldnames=[
                "query_id",
                "query",
                "product_locale",
                "esci_label",
                "small_version",
            ],
            rows=iter(
                [
                    {
                        "query_id": "q1",
                        "query": "wireless headphones",
                        "product_locale": "us",
                        "esci_label": "E",
                        "small_version": 1,
                    },
                    {
                        "query_id": "q1",
                        "query": "wireless headphones",
                        "product_locale": "us",
                        "esci_label": "S",
                        "small_version": 1,
                    },
                    {
                        "query_id": "q2",
                        "query": "tablet stand",
                        "product_locale": "uk",
                        "esci_label": "E",
                        "small_version": 1,
                    },
                ]
            ),
        )

    monkeypatch.setattr(
        query_candidates_module,
        "_open_parquet_rows",
        fake_open_parquet_rows,
    )

    candidates = build_esci_query_candidates(
        path,
        locale="us",
        require_small_version=True,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.text == "wireless headphones"
    assert candidate.record_count == 2
    assert candidate.labels_observed == ("E", "S")
    assert candidate.source_file == "esci.parquet"
    assert candidate.source_ref == "esci.parquet:query_id=q1"


def test_build_esci_query_candidates_rejects_non_binary_numeric_version_flags(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "esci.parquet"
    path.write_text("placeholder", encoding="utf-8")

    @contextmanager
    def fake_open_parquet_rows(source_path):
        assert source_path == path
        yield query_candidates_module._SourceRows(
            fieldnames=["query_id", "query", "small_version"],
            rows=iter(
                [
                    {
                        "query_id": "q1",
                        "query": "wireless headphones",
                        "small_version": 2,
                    },
                    {
                        "query_id": "q2",
                        "query": "tablet stand",
                        "small_version": 1,
                    },
                ]
            ),
        )

    monkeypatch.setattr(
        query_candidates_module,
        "_open_parquet_rows",
        fake_open_parquet_rows,
    )

    candidates = build_esci_query_candidates(
        path,
        locale=None,
        require_small_version=True,
    )

    assert [candidate.text for candidate in candidates] == ["tablet stand"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_records": 0}, "'min_records' must be >= 1"),
        ({"max_queries": 0}, "'max_queries' must be >= 1"),
    ],
)
def test_build_esci_query_candidates_validates_positive_bounds(
    tmp_path, kwargs, message
):
    path = tmp_path / "esci.tsv"

    with pytest.raises(ValueError, match=message):
        build_esci_query_candidates(path, **kwargs)


def test_build_esci_query_candidates_rejects_conflicting_version_filters(tmp_path):
    path = tmp_path / "esci.tsv"

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_esci_query_candidates(
            path,
            require_large_version=True,
            require_small_version=True,
        )


def test_build_query_bank_rows_from_candidates_bootstraps_inactive_rows():
    candidates = [
        QueryCandidate(
            candidate_id="qc_0001",
            text="wireless headphones",
            source_type="amazon_esci",
            domain="electronics",
        )
    ]

    rows = build_query_bank_rows_from_candidates(candidates)

    assert rows == [
        {
            "query_id": "qb_0001",
            "text": "wireless headphones",
            "source_type": "amazon_esci",
            "active": False,
            "source_ref": "qc_0001",
            "domain": "electronics",
            "category": None,
            "intent": None,
            "specificity": None,
            "answerability": None,
            "difficulty": None,
            "subset_tags": [],
            "relevant_items": None,
            "notes": (
                "Bootstrapped from query candidate pool; add category, intent, "
                "subset tags, and relevance judgments before using query-driven workflows."
            ),
            "provenance": {
                "schema_version": "query_provenance_v1",
                "origin_family": "query_candidate_bootstrap",
                "curation_mode": "candidate_bootstrap",
                "upstream_source": {
                    "dataset_name": "amazon_esci",
                    "source_file": None,
                    "source_ref": None,
                    "locale_hint": None,
                },
                "labels_observed": [],
                "selection": {
                    "policy": "query_candidate_bootstrap_v1",
                    "record_count": 1,
                    "included": True,
                },
                "subset_assignment": {
                    "policy": "bootstrap_subset_tags_v1",
                    "assigned_subset_tags": [],
                },
                "candidate_lineage": {
                    "candidate_id": "qc_0001",
                    "source_file": None,
                    "source_ref": None,
                },
            },
        }
    ]
