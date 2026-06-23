from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from sage.data._artifact_io import (
    iter_jsonl_object_rows,
    load_json_array_file,
    load_optional_json_object_file,
    load_required_json_object_file,
    write_json_object,
    write_jsonl_rows,
)


class ArtifactError(ValueError):
    """Test-only artifact validation error."""


def test_iter_jsonl_object_rows_skips_blank_lines_and_reports_source_lines(
    tmp_path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    rows = list(
        iter_jsonl_object_rows(
            path,
            label="test rows",
            row_description="test",
        )
    )

    assert rows == [({"id": 1}, 1), ({"id": 2}, 3)]


def test_iter_jsonl_object_rows_reports_parse_location(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2, column 2"):
        list(
            iter_jsonl_object_rows(
                path,
                label="test rows",
                row_description="test",
            )
        )


def test_iter_jsonl_object_rows_requires_object_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n["not", "object"]\n', encoding="utf-8")

    rows = iter_jsonl_object_rows(
        path,
        label="test artifact",
        row_description="test artifact",
    )
    assert next(rows) == ({"id": 1}, 1)
    with pytest.raises(ValueError, match="Each test artifact row"):
        next(rows)


def test_load_optional_json_object_file_preserves_missing_file_error(tmp_path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Optional artifact not found"):
        load_optional_json_object_file(path, description="Optional artifact")


def test_load_required_json_object_file_uses_domain_error_for_missing_path(
    tmp_path,
) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(ArtifactError, match="Required artifact not found"):
        load_required_json_object_file(
            path,
            description="Required artifact",
            error_cls=ArtifactError,
        )


def test_required_json_object_loader_rejects_invalid_json_and_non_objects(
    tmp_path,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="Invalid JSON format"):
        load_required_json_object_file(
            invalid_path,
            description="Required artifact",
            error_cls=ArtifactError,
        )

    array_path = tmp_path / "array.json"
    array_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ArtifactError, match="must contain a JSON object"):
        load_required_json_object_file(
            array_path,
            description="Required artifact",
            error_cls=ArtifactError,
        )


def test_required_json_object_loader_rejects_empty_payload(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactError, match="is empty"):
        load_required_json_object_file(
            path,
            description="Required artifact",
            error_cls=ArtifactError,
        )

    loaded = load_optional_json_object_file(
        path,
        description="Optional artifact",
        error_cls=ArtifactError,
    )
    assert loaded == {}


def test_load_json_array_file_validates_shape_and_emptiness(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('[{"id": 1}]', encoding="utf-8")

    assert load_json_array_file(path, description="Rows artifact") == [{"id": 1}]

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ArtifactError, match="is empty"):
        load_json_array_file(
            empty_path,
            description="Required rows artifact",
            error_cls=ArtifactError,
            require_nonempty=True,
        )

    object_path = tmp_path / "object.json"
    object_path.write_text('{"id": 1}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="must contain a JSON array"):
        load_json_array_file(
            object_path,
            description="Rows artifact",
            error_cls=ArtifactError,
        )


def test_write_jsonl_rows_accepts_iterable_mappings(tmp_path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    rows = (MappingProxyType({"b": index, "a": f"row-{index}"}) for index in range(2))

    saved = write_jsonl_rows(path, rows, sort_keys=True)

    assert saved == path
    assert path.read_text(encoding="utf-8").splitlines() == [
        '{"a": "row-0", "b": 0}',
        '{"a": "row-1", "b": 1}',
    ]


def test_write_json_object_accepts_mapping_payload(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"

    saved = write_json_object(
        path,
        MappingProxyType({"status": "ok", "count": 2}),
        sort_keys=True,
    )

    assert saved == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "count": 2,
        "status": "ok",
    }
