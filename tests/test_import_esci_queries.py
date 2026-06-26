"""Tests for scripts.import_esci_queries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.import_esci_queries as import_esci_queries


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--min-records", "0"),
        ("--max-queries", "-1"),
        ("--locale", ""),
        ("--domain", " "),
    ],
)
def test_parser_rejects_invalid_boundary_values(flag, value):
    parser = import_esci_queries.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "data/query_bank/sources/esci.parquet",
                flag,
                value,
            ]
        )


def test_main_imports_candidates_with_single_version_filter(tmp_path: Path):
    source = tmp_path / "esci.tsv"
    source.write_text(
        "\n".join(
            [
                "query_id\tquery\tproduct_locale\tesci_label\tlarge_version",
                "q1\twireless headphones\tus\tE\t1",
                "q1\twireless headphones\tus\tS\t1",
                "q2\tphone charger\tus\tE\t0",
                "q3\ttablet stand\tuk\tE\t1",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "query_candidates.jsonl"

    import_esci_queries.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--locale",
            "US",
            "--version",
            "large",
            "--domain",
            "electronics",
        ]
    )

    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["text"] == "wireless headphones"
    assert rows[0]["domain"] == "electronics"
    assert rows[0]["record_count"] == 2
