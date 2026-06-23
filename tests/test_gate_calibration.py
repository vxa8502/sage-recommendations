import pytest

from sage.core import ProductScore, RetrievedChunk
from sage.data.query_bank import QueryBankEntry
from sage.services.calibration._analysis import (
    analyze_gate_thresholds,
    choose_recommended_threshold,
    compare_gate_thresholds,
)
from sage.services.calibration._dataset import build_gate_calibration_dataset
from sage.services.calibration._io import (
    load_gate_calibration_dataset,
    save_gate_calibration_dataset,
)
from sage.services.calibration._types import (
    GateCalibrationRetrievalError,
    GateThreshold,
    GateThresholdMetrics,
)


def _chunk(product_id: str, text: str, score: float, review_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        product_id=product_id,
        rating=4.5,
        review_id=review_id,
    )


def _product(
    product_id: str,
    product_score: float,
    chunk_specs: list[tuple[str, float]],
) -> ProductScore:
    evidence = [
        _chunk(product_id, text, chunk_score, f"{product_id}-{i}")
        for i, (text, chunk_score) in enumerate(chunk_specs, start=1)
    ]
    return ProductScore(
        product_id=product_id,
        score=product_score,
        chunk_count=len(evidence),
        avg_rating=4.5,
        evidence=evidence,
    )


def _entry(
    query_id: str, text: str, relevant_items: dict[str, float]
) -> QueryBankEntry:
    return QueryBankEntry(
        query_id=query_id,
        text=text,
        source_type="amazon_esci",
        subset_tags=("gate_calibration",),
        relevant_items=relevant_items,
    )


def test_build_gate_calibration_dataset_tracks_query_ceiling_and_observations():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0, "B": 2.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93), ("b" * 120, 0.91)]),
            _product("X", 0.84, [("c" * 160, 0.84)]),
        ],
        "portable speaker": [
            _product("C", 0.84, [("e" * 80, 0.84), ("f" * 120, 0.82)]),
            _product("Y", 0.76, [("g" * 40, 0.76)]),
        ],
        "gaming mouse": [
            _product("Z", 0.82, [("h" * 160, 0.82)]),
        ],
    }

    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=2,
    )

    assert len(dataset.queries) == 3
    assert len(dataset.observations) == 5

    q1 = dataset.queries[0]
    assert q1.retrieved_relevant_product_ids == ("A",)
    assert q1.missed_relevant_product_ids == ("B",)
    assert q1.retrieved_relevant_count == 1

    q3 = dataset.queries[2]
    assert q3.retrieved_relevant_count == 0
    assert q3.missed_relevant_product_ids == ("D",)

    obs_a = next(row for row in dataset.observations if row.product_id == "A")
    assert obs_a.is_relevant is True
    assert obs_a.relevance_grade == 3.0
    assert obs_a.chunk_count == 2
    assert obs_a.total_tokens == 50


def test_build_gate_calibration_dataset_ignores_non_positive_relevance_grades():
    entries = [_entry("q1", "budget headphones", {"A": 3.0, "B": 0.0, "C": -1.0})]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93)]),
            _product("X", 0.84, [("b" * 80, 0.84)]),
        ],
    }

    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=2,
    )

    query = dataset.queries[0]
    assert query.relevant_count == 1
    assert query.relevant_grade_mass == 3.0
    assert query.retrieved_relevant_product_ids == ("A",)
    assert query.missed_relevant_product_ids == ()


def test_analyze_gate_thresholds_prefers_precision_with_ceiling_retained():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0, "B": 2.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93), ("b" * 120, 0.91)]),
            _product("X", 0.84, [("c" * 160, 0.84)]),
        ],
        "portable speaker": [
            _product("C", 0.84, [("e" * 80, 0.84), ("f" * 120, 0.82)]),
            _product("Y", 0.76, [("g" * 40, 0.76)]),
        ],
        "gaming mouse": [
            _product("Z", 0.82, [("h" * 160, 0.82)]),
        ],
    }
    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=2,
    )

    analysis = analyze_gate_thresholds(
        dataset,
        token_thresholds=[20],
        chunk_thresholds=[1, 2],
        score_thresholds=[0.7, 0.8],
        query_success_retention=0.95,
        bootstrap_samples=0,
    )

    assert analysis["dataset_summary"]["candidate_hit_rate"] == 0.6667
    assert analysis["selection_policy"]["query_success_ceiling"] == 0.6667
    assert analysis["selection_policy"]["required_query_success_rate"] == 0.6333
    assert analysis["selection_policy"]["candidate_hit_rate_upper_bound"] == 0.6667
    assert analysis["current_threshold"] == {
        "min_tokens": 20,
        "min_chunks": 1,
        "min_score": 0.7,
    }
    assert analysis["recommended_threshold"] == {
        "min_tokens": 20,
        "min_chunks": 2,
        "min_score": 0.7,
    }
    assert analysis["current_metrics"]["precision_at_accept"] == 0.5
    assert analysis["recommended_metrics"]["precision_at_accept"] == 1.0
    assert analysis["recommended_metrics"]["query_success_rate"] == 0.6667


def test_build_gate_calibration_dataset_skips_small_number_of_failed_queries():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93), ("b" * 120, 0.91)])
        ],
        "gaming mouse": [_product("Z", 0.82, [("h" * 160, 0.82)])],
    }

    def _retriever(entry: QueryBankEntry):
        if entry.text == "portable speaker":
            raise RuntimeError("Unexpected Response: 500 (Internal Server Error)")
        return products_by_query[entry.text]

    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=_retriever,
        top_k=2,
        continue_on_retrieval_error=True,
        max_failed_queries=1,
        max_failure_rate=0.5,
    )

    assert dataset.attempted_query_count == 3
    assert len(dataset.queries) == 2
    assert len(dataset.failed_queries) == 1
    assert dataset.failed_queries[0].query_id == "q2"
    assert dataset.failed_queries[0].error_type == "RuntimeError"
    assert "Internal Server Error" in dataset.failed_queries[0].error_message

    analysis = analyze_gate_thresholds(
        dataset,
        token_thresholds=[20],
        chunk_thresholds=[1],
        score_thresholds=[0.7],
        bootstrap_samples=0,
    )
    assert analysis["dataset_summary"]["attempted_query_count"] == 3
    assert analysis["dataset_summary"]["failed_query_count"] == 1
    assert analysis["dataset_summary"]["query_coverage_rate"] == 0.6667


def test_build_gate_calibration_dataset_marks_query_limited_scope():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [_product("A", 0.93, [("a" * 80, 0.93)])],
        "portable speaker": [_product("C", 0.84, [("b" * 80, 0.84)])],
        "gaming mouse": [_product("D", 0.82, [("c" * 80, 0.82)])],
    }

    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=1,
        query_limit=2,
    )

    assert dataset.available_query_count == 3
    assert dataset.attempted_query_count == 2
    assert dataset.requested_query_limit == 2
    assert dataset.sample_limited is True

    analysis = analyze_gate_thresholds(
        dataset,
        token_thresholds=[20],
        chunk_thresholds=[1],
        score_thresholds=[0.7],
        bootstrap_samples=0,
    )
    assert analysis["dataset_summary"]["available_query_count"] == 3
    assert analysis["dataset_summary"]["requested_query_limit"] == 2
    assert analysis["dataset_summary"]["sample_limited"] is True


def test_build_gate_calibration_dataset_aborts_when_failure_budget_exceeded():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]

    def _retriever(_entry: QueryBankEntry):
        raise RuntimeError("Unexpected Response: 500 (Internal Server Error)")

    with pytest.raises(
        GateCalibrationRetrievalError, match="Too many retrieval failures"
    ):
        build_gate_calibration_dataset(
            entries=entries,
            retriever=_retriever,
            continue_on_retrieval_error=True,
            max_failed_queries=1,
            max_failure_rate=0.2,
        )


def test_compare_gate_thresholds_reports_metric_deltas():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0, "B": 2.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93), ("b" * 120, 0.91)]),
            _product("X", 0.84, [("c" * 160, 0.84)]),
        ],
        "portable speaker": [
            _product("C", 0.84, [("e" * 80, 0.84), ("f" * 120, 0.82)]),
            _product("Y", 0.76, [("g" * 40, 0.76)]),
        ],
        "gaming mouse": [
            _product("Z", 0.82, [("h" * 160, 0.82)]),
        ],
    }
    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=2,
    )

    comparison = compare_gate_thresholds(
        dataset,
        GateThreshold(min_tokens=20, min_chunks=1, min_score=0.7),
        GateThreshold(min_tokens=20, min_chunks=2, min_score=0.7),
        baseline_label="current_config",
        candidate_label="candidate_threshold",
    )

    assert comparison["baseline_label"] == "current_config"
    assert comparison["candidate_label"] == "candidate_threshold"
    assert comparison["baseline_metrics"]["precision_at_accept"] == 0.5
    assert comparison["candidate_metrics"]["precision_at_accept"] == 1.0
    assert comparison["metric_deltas"]["precision_at_accept"] == 0.5
    assert comparison["metric_deltas"]["query_success_rate"] == 0.0


def test_analyze_gate_thresholds_reports_grade_mass_delta_vs_current():
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0, "B": 2.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
        _entry("q3", "gaming mouse", {"D": 2.0}),
    ]
    products_by_query = {
        "budget headphones": [
            _product("A", 0.93, [("a" * 80, 0.93), ("b" * 120, 0.91)]),
            _product("X", 0.84, [("c" * 160, 0.84)]),
        ],
        "portable speaker": [
            _product("C", 0.84, [("e" * 80, 0.84), ("f" * 120, 0.82)]),
            _product("Y", 0.76, [("g" * 40, 0.76)]),
        ],
        "gaming mouse": [
            _product("Z", 0.82, [("h" * 160, 0.82)]),
        ],
    }
    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=2,
    )

    analysis = analyze_gate_thresholds(
        dataset,
        token_thresholds=[20],
        chunk_thresholds=[1, 2],
        score_thresholds=[0.7],
        bootstrap_samples=0,
    )

    assert (
        analysis["metric_deltas_vs_current"]["retrieved_relevant_grade_mass_pass_rate"]
        == 0.0
    )


def test_choose_recommended_threshold_uses_raw_query_success_counts():
    rows = [
        GateThresholdMetrics(
            min_tokens=20,
            min_chunks=2,
            min_score=0.7,
            total_queries=10_000,
            total_observations=1_000,
            candidate_hit_queries=10_000,
            accepted_queries=9_500,
            total_retrieved_relevant=1_000,
            total_retrieved_relevant_grade_mass=1_000.0,
            accepted_count=1_000,
            accepted_relevant_count=900,
            accepted_irrelevant_count=100,
            accepted_relevant_grade_mass=900.0,
        ),
        GateThresholdMetrics(
            min_tokens=10,
            min_chunks=1,
            min_score=0.6,
            total_queries=9_999,
            total_observations=950,
            candidate_hit_queries=9_999,
            accepted_queries=9_499,
            total_retrieved_relevant=950,
            total_retrieved_relevant_grade_mass=950.0,
            accepted_count=950,
            accepted_relevant_count=950,
            accepted_irrelevant_count=0,
            accepted_relevant_grade_mass=950.0,
        ),
    ]

    selected = choose_recommended_threshold(rows, query_success_retention=1.0)

    assert selected.min_tokens == 20
    assert selected.min_chunks == 2
    assert selected.min_score == 0.7


def test_save_and_load_gate_calibration_dataset_round_trip(tmp_path):
    entries = [
        _entry("q1", "budget headphones", {"A": 3.0}),
        _entry("q2", "portable speaker", {"C": 3.0}),
    ]
    products_by_query = {
        "budget headphones": [_product("A", 0.93, [("a" * 80, 0.93)])],
        "portable speaker": [_product("C", 0.84, [("b" * 120, 0.84)])],
    }
    dataset = build_gate_calibration_dataset(
        entries=entries,
        retriever=lambda entry: products_by_query[entry.text],
        top_k=1,
        query_limit=1,
    )

    path = tmp_path / "gate_calibration.json"
    save_gate_calibration_dataset(dataset, path)
    loaded = load_gate_calibration_dataset(path)

    assert loaded == dataset
