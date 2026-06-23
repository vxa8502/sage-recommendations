from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sage.data.query_bank.sources.esci._config import (
    DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
    DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG,
    DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
    DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
)
from sage.data.query_bank import QueryBankSubsetEmptyError, load_query_bank_subset

GATE_CALIBRATION_SUBSET_TAG = "gate_calibration"


@dataclass(frozen=True)
class QueryBankSubsetRequirement:
    subset_tag: str
    require_relevant_items: bool
    purpose: str
    status_key: str | None = None

    @property
    def missing_detail(self) -> str:
        if self.require_relevant_items:
            return "active queries with relevance judgments"
        return "active queries"


CALIBRATION_QUERY_BANK_REQUIREMENTS: tuple[QueryBankSubsetRequirement, ...] = (
    QueryBankSubsetRequirement(
        GATE_CALIBRATION_SUBSET_TAG,
        True,
        "judged calibration coverage",
        "gate_calibration_ready",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,
        True,
        "judged calibration retrieval/gate holdout coverage",
        "retrieval_dev_holdout_ready",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
        True,
        "sealed evaluation retrieval reporting coverage",
        "retrieval_final_report_ready",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_FAITHFULNESS_DEV_SEED_SUBSET_TAG,
        False,
        "dev seed-case materialization coverage",
        "faithfulness_dev_seed_ready",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_FAITHFULNESS_FINAL_SEED_SUBSET_TAG,
        False,
        "sealed final seed-case coverage",
        "faithfulness_final_seed_ready",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        False,
        "boundary guardrail coverage",
        "boundary_eval_ready",
    ),
)

EVAL_QUERY_BANK_REQUIREMENTS: tuple[QueryBankSubsetRequirement, ...] = (
    QueryBankSubsetRequirement(
        DEFAULT_RETRIEVAL_FINAL_REPORT_SUBSET_TAG,
        True,
        "retrieval metrics",
    ),
    QueryBankSubsetRequirement(
        DEFAULT_BOUNDARY_EVAL_SUBSET_TAG,
        False,
        "boundary behavior guardrail checks",
    ),
)


def load_query_bank_requirement(
    requirement: QueryBankSubsetRequirement,
    *,
    path: str | Path,
) -> list[Any]:
    return load_query_bank_subset(
        requirement.subset_tag,
        path=path,
        require_relevant_items=requirement.require_relevant_items,
        require_nonempty=True,
    )


def load_query_bank_requirements(
    requirements: tuple[QueryBankSubsetRequirement, ...],
    *,
    path: str | Path,
) -> tuple[dict[str, list[Any]], list[str]]:
    loaded: dict[str, list[Any]] = {}
    issues: list[str] = []
    for requirement in requirements:
        try:
            loaded[requirement.subset_tag] = load_query_bank_requirement(
                requirement,
                path=path,
            )
        except (FileNotFoundError, QueryBankSubsetEmptyError, ValueError) as exc:
            issues.append(f"{requirement.purpose}: {exc}")
    return loaded, issues
