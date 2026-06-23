"""Tests for sage.data.query_bank."""

import json

import pytest

from sage.data.query_bank import (
    QueryBankEntry,
    QueryProvenance,
    QueryBankSubsetEmptyError,
    load_eval_cases_from_query_bank,
    load_query_bank,
    load_query_bank_manifest,
    load_query_bank_subset,
    save_query_bank_manifest,
    save_query_bank_rows,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_provenance(
    *,
    origin_family: str = "amazon_esci_overlap",
    curation_mode: str = "pure_import",
    source_split: str = "test",
    assigned_subset_tags: list[str] | None = None,
) -> dict:
    subset_tags = assigned_subset_tags or ["retrieval_eval"]
    return {
        "schema_version": "query_provenance_v1",
        "origin_family": origin_family,
        "curation_mode": curation_mode,
        "upstream_source": {
            "dataset_name": "amazon_esci",
            "source_file": "examples.parquet",
            "source_split": source_split,
            "source_query_id": "q1",
            "locale": "us",
            "version": "large",
        },
        "labels_observed": ["E"],
        "selection": {
            "policy": "corpus_overlap_min_relevant_items_v1",
            "included": True,
            "min_relevant_items": 1,
            "overlap_relevant_item_count": 1,
        },
        "subset_assignment": {
            "policy": "normalized_query_sha256_v1",
            "source_split": source_split,
            "assigned_subset_tags": subset_tags,
        },
        "candidate_lineage": None,
    }


class TestLoadQueryBank:
    def test_blank_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        path.write_text("\n", encoding="utf-8")

        assert load_query_bank(path) == []

    def test_loads_valid_rows(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "wireless headphones for working out",
                    "source_type": "amazon_esci",
                    "domain": "electronics",
                    "intent": "use_case",
                    "subset_tags": ["retrieval_eval", "gate_calibration"],
                    "relevant_items": {"ASIN1": 3.0},
                }
            ],
        )

        rows = load_query_bank(path)

        assert rows == [
            QueryBankEntry(
                query_id="qb_001",
                text="wireless headphones for working out",
                source_type="amazon_esci",
                domain="electronics",
                intent="use_case",
                subset_tags=("retrieval_eval", "gate_calibration"),
                relevant_items={"ASIN1": 3.0},
            )
        ]

    def test_loads_structured_provenance_when_present(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "wireless headphones for working out",
                    "source_type": "amazon_esci",
                    "subset_tags": ["retrieval_eval"],
                    "provenance": _make_provenance(),
                }
            ],
        )

        rows = load_query_bank(path)

        assert rows == [
            QueryBankEntry(
                query_id="qb_001",
                text="wireless headphones for working out",
                source_type="amazon_esci",
                subset_tags=("retrieval_eval",),
                provenance=QueryProvenance(
                    schema_version="query_provenance_v1",
                    origin_family="amazon_esci_overlap",
                    curation_mode="pure_import",
                    upstream_source={
                        "dataset_name": "amazon_esci",
                        "source_file": "examples.parquet",
                        "source_split": "test",
                        "source_query_id": "q1",
                        "locale": "us",
                        "version": "large",
                    },
                    labels_observed=("E",),
                    selection={
                        "policy": "corpus_overlap_min_relevant_items_v1",
                        "included": True,
                        "min_relevant_items": 1,
                        "overlap_relevant_item_count": 1,
                    },
                    subset_assignment={
                        "policy": "normalized_query_sha256_v1",
                        "source_split": "test",
                        "assigned_subset_tags": ["retrieval_eval"],
                    },
                    candidate_lineage=None,
                ),
            )
        ]

    def test_filters_inactive_rows_when_requested(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "active row",
                    "source_type": "amazon_esci",
                },
                {
                    "query_id": "qb_002",
                    "text": "inactive row",
                    "source_type": "manual_stress",
                    "active": False,
                },
            ],
        )

        rows = load_query_bank(path, include_inactive=False)

        assert [row.query_id for row in rows] == ["qb_001"]

    def test_duplicate_query_id_raises_error(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "first row",
                    "source_type": "amazon_esci",
                },
                {
                    "query_id": "qb_001",
                    "text": "second row",
                    "source_type": "amazon_esci",
                },
            ],
        )

        with pytest.raises(ValueError, match="Duplicate query_id 'qb_001'"):
            load_query_bank(path)

    def test_invalid_subset_tags_type_raises_error(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "row",
                    "source_type": "amazon_esci",
                    "subset_tags": "retrieval_eval",
                }
            ],
        )

        with pytest.raises(ValueError, match="'subset_tags' must be a list"):
            load_query_bank(path)

    def test_invalid_relevant_items_type_raises_error(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "row",
                    "source_type": "amazon_esci",
                    "relevant_items": ["ASIN1"],
                }
            ],
        )

        with pytest.raises(ValueError, match="'relevant_items' must be a dict"):
            load_query_bank(path)

    @pytest.mark.parametrize(
        ("relevant_items", "message"),
        [
            ({}, "must not be empty"),
            ({"ASIN1": True}, "must be numeric"),
            ({"ASIN1": -1}, "must be >= 0"),
        ],
    )
    def test_invalid_relevant_item_scores_raise_error(
        self, tmp_path, relevant_items, message
    ):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "row",
                    "source_type": "amazon_esci",
                    "relevant_items": relevant_items,
                }
            ],
        )

        with pytest.raises(ValueError, match=message):
            load_query_bank(path)

    def test_invalid_jsonl_reports_line_number(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        path.write_text(
            (
                '{"query_id": "qb_001", "text": "valid row", '
                '"source_type": "amazon_esci"}\n'
                "{bad json}\n"
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="line 2"):
            load_query_bank(path)

    def test_invalid_provenance_type_raises_error(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "row",
                    "source_type": "amazon_esci",
                    "provenance": "not-an-object",
                }
            ],
        )

        with pytest.raises(ValueError, match="'provenance' must be an object"):
            load_query_bank(path)


class TestLoadQueryBankManifest:
    def test_loads_manifest_object(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"dataset_name": "query_bank", "status": "bootstrap"}),
            encoding="utf-8",
        )

        manifest = load_query_bank_manifest(path)

        assert manifest["dataset_name"] == "query_bank"
        assert manifest["status"] == "bootstrap"

    def test_save_then_load_manifest_round_trip(self, tmp_path):
        path = tmp_path / "manifest.json"
        save_query_bank_manifest(
            {
                "dataset_name": "query_bank",
                "stage": "ingestion_data_staging",
                "status": "ready_esci_overlap",
            },
            path,
        )

        manifest = load_query_bank_manifest(path)

        assert manifest["dataset_name"] == "query_bank"
        assert manifest["stage"] == "ingestion_data_staging"
        assert manifest["status"] == "ready_esci_overlap"

    def test_save_query_bank_rows_round_trip(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        row = {
            "query_id": "qb_001",
            "text": "wireless headphones",
            "source_type": "manual_seed",
            "subset_tags": ["retrieval_eval"],
            "relevant_items": {"ASIN1": 1.0},
        }

        save_query_bank_rows([row], path)

        assert load_query_bank(path) == [
            QueryBankEntry(
                query_id="qb_001",
                text="wireless headphones",
                source_type="manual_seed",
                subset_tags=("retrieval_eval",),
                relevant_items={"ASIN1": 1.0},
            )
        ]


class TestQueryBankSubsetHelpers:
    def test_load_query_bank_subset_filters_rows(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "query one",
                    "source_type": "legacy_repo_manual",
                    "subset_tags": ["faithfulness_seed"],
                },
                {
                    "query_id": "qb_002",
                    "text": "query two",
                    "source_type": "legacy_repo_manual",
                    "subset_tags": ["retrieval_eval"],
                    "relevant_items": {"ASIN1": 3.0},
                },
            ],
        )

        subset = load_query_bank_subset(
            "retrieval_eval",
            path=path,
            require_relevant_items=True,
        )

        assert [entry.query_id for entry in subset] == ["qb_002"]

    def test_load_query_bank_subset_require_nonempty_raises(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        path.write_text("", encoding="utf-8")

        with pytest.raises(
            QueryBankSubsetEmptyError,
            match="Required query-bank subset 'retrieval_eval' is empty",
        ):
            load_query_bank_subset("retrieval_eval", path=path, require_nonempty=True)

    def test_load_query_bank_subset_require_nonempty_honors_relevant_items(
        self, tmp_path
    ):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "query one",
                    "source_type": "legacy_repo_manual",
                    "subset_tags": ["retrieval_eval"],
                }
            ],
        )

        with pytest.raises(
            QueryBankSubsetEmptyError,
            match="retrieval_eval' with relevance judgments",
        ):
            load_query_bank_subset(
                "retrieval_eval",
                path=path,
                require_relevant_items=True,
                require_nonempty=True,
            )

    def test_load_eval_cases_from_query_bank(self, tmp_path):
        path = tmp_path / "query_bank.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "query_id": "qb_001",
                    "text": "example retrieval query",
                    "source_type": "manual_seed",
                    "category": "keyboards_mice",
                    "intent": "use_case",
                    "subset_tags": ["retrieval_eval"],
                    "relevant_items": {"ASIN1": 3.0, "ASIN2": 2.0},
                    "provenance": _make_provenance(
                        origin_family="manual_seed",
                        curation_mode="candidate_bootstrap",
                    ),
                }
            ],
        )

        cases = load_eval_cases_from_query_bank("retrieval_eval", path=path)

        assert len(cases) == 1
        assert cases[0].query == "example retrieval query"
        assert cases[0].relevant_items == {"ASIN1": 3.0, "ASIN2": 2.0}
        assert cases[0].query_id == "qb_001"
        assert cases[0].source_type == "manual_seed"
        assert cases[0].category == "keyboards_mice"
        assert cases[0].intent == "use_case"
        assert cases[0].subset_tags == ("retrieval_eval",)
        assert cases[0].query_slice_tags == ()
        assert cases[0].provenance is not None
        assert cases[0].provenance.origin_family == "manual_seed"
        assert cases[0].provenance.curation_mode == "candidate_bootstrap"
        assert cases[0].provenance.source_dataset == "amazon_esci"
        assert cases[0].provenance.source_split == "test"
        assert (
            cases[0].provenance.selection_policy
            == "corpus_overlap_min_relevant_items_v1"
        )
        assert (
            cases[0].provenance.subset_assignment_policy == "normalized_query_sha256_v1"
        )
