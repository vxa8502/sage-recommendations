"""Tests for sage.data.split_leakage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sage.data.split_leakage import (
    build_strong_paraphrase_components,
    build_split_leakage_matrix_audit,
    save_split_leakage_audit,
)


def _row(
    query_id: str,
    text: str,
    subset_tag: str,
    *,
    relevant_items: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "text": text,
        "source_type": "amazon_esci",
        "source_ref": f"examples.parquet:{query_id}",
        "subset_tags": [subset_tag],
        "relevant_items": relevant_items,
    }


def test_build_strong_paraphrase_components_merges_only_strong_pairs():
    entries = [
        {
            "query_id": "q1",
            "text": "home security camera",
            "source_ref": "examples.parquet:q1",
            "relevant_items": {"P1": 3.0, "P2": 2.0},
        },
        {
            "query_id": "q2",
            "text": "cameras for home security",
            "source_ref": "examples.parquet:q2",
            "relevant_items": {"P1": 3.0, "P2": 2.0},
        },
        {
            "query_id": "q3",
            "text": "4k graphics card",
            "source_ref": "examples.parquet:q3",
            "relevant_items": {"P9": 3.0},
        },
        {
            "query_id": "q4",
            "text": "4k gaming monitor",
            "source_ref": "examples.parquet:q4",
            "relevant_items": {"P10": 3.0},
        },
    ]
    embeddings = {
        "q1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "q2": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "q3": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "q4": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    components = build_strong_paraphrase_components(
        entries,
        semantic_embeddings_by_query_id=embeddings,
    )

    by_query_id = components["query_id_to_component"]
    assert components["group_key"] == "strong_paraphrase_component"
    assert components["component_edge_policy"] == (
        "exact_duplicate_or_high_confidence_near_duplicate_v1"
    )
    assert components["strong_edge_count"] == 1
    assert components["high_confidence_edge_count"] == 1
    assert components["multi_query_component_count"] == 1
    assert components["queries_in_multi_query_components"] == 2
    assert by_query_id["q1"]["component_id"] == by_query_id["q2"]["component_id"]
    assert by_query_id["q3"]["component_id"] != by_query_id["q4"]["component_id"]
    assert by_query_id["q3"]["component_size"] == 1
    assert by_query_id["q4"]["component_size"] == 1


def test_build_split_leakage_matrix_audit_summarizes_all_surface_pairs():
    rows = [
        _row(
            "qb_g1",
            "wireless earbuds for gym",
            "gate_calibration",
            relevant_items={"P1": 3.0},
        ),
        _row(
            "qb_g2",
            "mechanical keyboard for coding",
            "gate_calibration",
            relevant_items={"P5": 3.0},
        ),
        _row(
            "qb_r1",
            "wireless earbuds for the gym",
            "retrieval_eval",
            relevant_items={"P1": 3.0},
        ),
        _row(
            "qb_r2",
            "usb c hub for laptop",
            "retrieval_eval",
            relevant_items={"P2": 3.0},
        ),
        _row(
            "qb_f1",
            "usb c hub laptop",
            "faithfulness_seed",
            relevant_items={"P2": 3.0},
        ),
        _row(
            "qb_f2",
            "portable monitor stand",
            "faithfulness_seed",
            relevant_items={"P9": 3.0},
        ),
    ]
    embeddings = {
        "qb_g1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "qb_g2": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "qb_r1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "qb_r2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "qb_f1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "qb_f2": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    audit = build_split_leakage_matrix_audit(
        rows,
        semantic_embeddings_by_query_id=embeddings,
    )

    assert audit["audit_version"] == "cross_surface_query_similarity_v2"
    assert [surface["surface_name"] for surface in audit["surface_catalog"]] == [
        "gate_calibration",
        "retrieval_eval",
        "faithfulness_seed",
    ]
    assert audit["pair_count"] == 3
    assert audit["summary"]["overall_risk_level"] == "moderate"
    assert audit["summary"]["pairs_by_risk_level"] == {
        "moderate": 2,
        "low": 1,
    }
    assert audit["summary"]["aggregate_severity_counts"] == {
        "exact_duplicate": 0,
        "high_confidence_near_duplicate": 2,
        "semantic_watchlist": 0,
    }

    pair_summaries = {pair["pair_id"]: pair for pair in audit["pair_summaries"]}
    assert (
        pair_summaries["gate_calibration__vs__retrieval_eval"]["risk_level"]
        == "moderate"
    )
    assert (
        pair_summaries["retrieval_eval__vs__faithfulness_seed"]["risk_level"]
        == "moderate"
    )
    assert (
        pair_summaries["gate_calibration__vs__faithfulness_seed"]["risk_level"] == "low"
    )
    assert audit["summary"]["worst_pairs"][0]["pair_id"] in {
        "gate_calibration__vs__retrieval_eval",
        "retrieval_eval__vs__faithfulness_seed",
    }


def test_build_split_leakage_matrix_audit_accepts_legacy_subset_tags():
    rows = [
        _row(
            "qb_g1",
            "wireless mouse",
            "gate_calibration",
            relevant_items={"P1": 3.0},
        ),
        _row(
            "qb_r1",
            "usb c dock",
            "retrieval_final_report",
            relevant_items={"P2": 3.0},
        ),
        _row(
            "qb_f1",
            "monitor light bar",
            "faithfulness_final_seed",
            relevant_items={"P3": 3.0},
        ),
    ]
    embeddings = {
        "qb_g1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "qb_r1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "qb_f1": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    audit = build_split_leakage_matrix_audit(
        rows,
        semantic_embeddings_by_query_id=embeddings,
    )

    surface_catalog = {
        surface["surface_name"]: surface for surface in audit["surface_catalog"]
    }
    assert surface_catalog["retrieval_eval"]["subset_tags"] == [
        "retrieval_eval",
        "retrieval_dev_holdout",
        "retrieval_final_report",
    ]
    assert surface_catalog["retrieval_eval"]["query_count"] == 1
    assert surface_catalog["faithfulness_seed"]["query_count"] == 1
    assert audit["summary"]["overall_risk_level"] == "low"
    assert audit["summary"]["aggregate_flagged_pair_count"] == 0


def test_build_split_leakage_matrix_audit_ignores_unmatched_embedding_cache_entries():
    rows = [
        _row("qb_g1", "wireless mouse", "gate_calibration"),
        _row("qb_r1", "usb c dock", "retrieval_eval"),
        _row("qb_f1", "monitor light bar", "faithfulness_seed"),
    ]
    embeddings = {
        "qb_g1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "qb_r1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "qb_f1": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "stale_bad_vector": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    }

    audit = build_split_leakage_matrix_audit(
        rows,
        semantic_embeddings_by_query_id=embeddings,
    )

    assert audit["summary"]["overall_risk_level"] == "low"
    assert audit["summary"]["aggregate_flagged_pair_count"] == 0


def test_build_split_leakage_matrix_audit_rejects_string_boolean_surface_flags():
    with pytest.raises(ValueError, match="'include_in_global_risk' must be a bool"):
        build_split_leakage_matrix_audit(
            [],
            surface_specs=(
                {
                    "surface_name": "gate_calibration",
                    "subset_tags": ("gate_calibration",),
                    "include_in_global_risk": "false",
                },
            ),
        )


def test_build_split_leakage_matrix_audit_validates_saved_pair_limit():
    rows = [
        _row("qb_g1", "wireless earbuds for gym", "gate_calibration"),
        _row("qb_r1", "wireless earbuds for the gym", "retrieval_eval"),
    ]
    embeddings = {
        "qb_g1": np.array([1.0, 0.0], dtype=np.float32),
        "qb_r1": np.array([1.0, 0.0], dtype=np.float32),
    }
    surface_specs = (
        {
            "surface_name": "gate_calibration",
            "subset_tags": ("gate_calibration",),
        },
        {
            "surface_name": "retrieval_eval",
            "subset_tags": ("retrieval_eval",),
        },
    )

    with pytest.raises(ValueError, match="'saved_pair_limit' must be >= 0"):
        build_split_leakage_matrix_audit(
            rows,
            surface_specs=surface_specs,
            semantic_embeddings_by_query_id=embeddings,
            saved_pair_limit=-1,
        )

    with pytest.raises(ValueError, match="'saved_pair_limit' must be an int"):
        build_split_leakage_matrix_audit(
            rows,
            surface_specs=surface_specs,
            semantic_embeddings_by_query_id=embeddings,
            saved_pair_limit=True,
        )

    audit = build_split_leakage_matrix_audit(
        rows,
        surface_specs=surface_specs,
        semantic_embeddings_by_query_id=embeddings,
        saved_pair_limit=0,
    )
    pair_audit = audit["pair_audits"][0]
    assert audit["methodology"]["saved_pair_limit"] == 0
    assert pair_audit["total_flagged_pair_count"] == 1
    assert pair_audit["saved_flagged_pair_count"] == 0
    assert pair_audit["flagged_pairs"] == []


def test_save_split_leakage_audit_writes_pretty_json(tmp_path):
    path = tmp_path / "split_leakage_audit.json"
    payload = {
        "audit_version": "cross_surface_query_similarity_v2",
        "summary": {"risk_level": "low"},
        "flagged_pairs": [],
    }

    saved_path = save_split_leakage_audit(payload, path)

    assert saved_path == path
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
