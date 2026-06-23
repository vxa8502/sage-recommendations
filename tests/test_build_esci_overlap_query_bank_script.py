"""Tests for scripts.build_esci_overlap_query_bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.build_esci_overlap_query_bank as build_script


def _build_config(
    tmp_path: Path, **overrides: Any
) -> build_script.QueryBankBuildConfig:
    values: dict[str, Any] = {
        "examples": tmp_path / "examples.parquet",
        "output": tmp_path / "query_bank.jsonl",
        "manifest_output": tmp_path / "manifest.json",
        "candidate_pool": tmp_path / "query_candidates.jsonl",
        "subset_size": 123,
        "product_id_cache": tmp_path / "product_ids.json",
        "chunk_manifest": None,
        "force_product_id_cache": False,
        "locale": "us",
        "version": "large",
        "min_relevant_items": 1,
        "max_queries": None,
        "label_weights": {"E": 3.0, "S": 2.0},
        "test_splits": build_script.TestSplitAssignmentPolicy(
            retrieval_family_share=0.8,
            retrieval_dev_share=0.75,
            faithfulness_dev_share=1 / 3,
        ),
        "manual_boundary_path": tmp_path / "manual_boundary_queries.jsonl",
        "split_leakage_audit_output": tmp_path / "split_leakage_audit.json",
    }
    values.update(overrides)
    return build_script.QueryBankBuildConfig(**values)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--subset-size", "0"),
        ("--min-relevant-items", "0"),
        ("--max-queries", "-1"),
        ("--test-retrieval-share", "1.1"),
        ("--test-retrieval-dev-share", "-0.1"),
        ("--test-faithfulness-dev-share", "nan"),
        ("--locale", " "),
    ],
)
def test_parser_rejects_invalid_boundary_values(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_script.main([flag, value])


def test_resolve_corpus_product_ids_prefers_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "product_ids.json"
    cache_path.write_text("{}", encoding="utf-8")
    chunk_manifest = tmp_path / "chunks_123.jsonl"
    config = _build_config(
        tmp_path,
        product_id_cache=cache_path,
        chunk_manifest=chunk_manifest,
        force_product_id_cache=False,
    )
    calls: list[str] = []

    def fake_load_corpus_product_ids(path: Path) -> set[str]:
        calls.append(f"load:{path.name}")
        return {"cached"}

    def fail_build_from_chunk_manifest(*_args: object, **_kwargs: object) -> set[str]:
        raise AssertionError("chunk manifest should not be used without force")

    monkeypatch.setattr(
        build_script,
        "load_corpus_product_ids",
        fake_load_corpus_product_ids,
    )
    monkeypatch.setattr(
        build_script,
        "build_corpus_product_id_cache_from_chunk_manifest",
        fail_build_from_chunk_manifest,
    )

    assert build_script._resolve_corpus_product_ids(config) == {"cached"}
    assert calls == ["load:product_ids.json"]


def test_resolve_corpus_product_ids_uses_chunk_manifest_when_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "product_ids.json"
    cache_path.write_text("{}", encoding="utf-8")
    chunk_manifest = tmp_path / "chunks_123.jsonl"
    config = _build_config(
        tmp_path,
        product_id_cache=cache_path,
        chunk_manifest=chunk_manifest,
        force_product_id_cache=True,
    )
    calls: list[tuple[Path, int, Path]] = []

    def fake_build_from_chunk_manifest(
        manifest_path: Path,
        *,
        subset_size: int,
        path: Path,
    ) -> set[str]:
        calls.append((manifest_path, subset_size, path))
        return {"rebuilt"}

    def fail_load_corpus_product_ids(*_args: object, **_kwargs: object) -> set[str]:
        raise AssertionError("existing cache should be ignored when force is set")

    monkeypatch.setattr(
        build_script,
        "load_corpus_product_ids",
        fail_load_corpus_product_ids,
    )
    monkeypatch.setattr(
        build_script,
        "build_corpus_product_id_cache_from_chunk_manifest",
        fake_build_from_chunk_manifest,
    )

    assert build_script._resolve_corpus_product_ids(config) == {"rebuilt"}
    assert calls == [(chunk_manifest, 123, cache_path)]
