from __future__ import annotations

from pathlib import Path

from sage.data.faithfulness import (
    FAITHFULNESS_CASES_PATH,
    FAITHFULNESS_DEV_SOURCE_SUBSET,
    FaithfulnessCasesManifestError,
    FaithfulnessCaseOutcomesEmptyError,
    FaithfulnessCasesEmptyError,
    load_frozen_freshness_reference,
    load_faithfulness_case_outcomes,
    load_faithfulness_cases,
    load_faithfulness_cases_manifest,
    resolve_faithfulness_case_outcomes_path,
    resolve_faithfulness_cases_manifest_path,
)
from sage.data.query_bank import (
    QUERY_BANK_PATH,
    QueryBankSubsetEmptyError,
    load_query_bank_subset,
)
from ..query_bank_contracts import (
    EVAL_QUERY_BANK_REQUIREMENTS,
    load_query_bank_requirement,
)


def ensure_eval_query_bank_ready(path: str | Path = QUERY_BANK_PATH) -> None:
    """Fail fast when the canonical eval query bank is not populated."""
    missing: list[str] = []
    for requirement in EVAL_QUERY_BANK_REQUIREMENTS:
        try:
            load_query_bank_requirement(requirement, path=path)
        except QueryBankSubsetEmptyError:
            missing.append(
                f"- {requirement.subset_tag}: {requirement.missing_detail} "
                f"required for {requirement.purpose}"
            )

    if missing:
        raise SystemExit(
            "ERROR: Cannot run `sage eval run` because required query-bank "
            f"subsets are empty in {Path(path)}.\n"
            "Populate the canonical query bank before running the full "
            "evaluation workflow.\n" + "\n".join(missing)
        )


def ensure_boundary_eval_query_bank_ready(
    *,
    subset_tag: str = "boundary_eval",
    path: str | Path = QUERY_BANK_PATH,
) -> None:
    """Fail fast when the canonical boundary slice is not populated."""
    try:
        load_query_bank_subset(
            subset_tag,
            path=path,
            require_nonempty=True,
        )
    except QueryBankSubsetEmptyError as exc:
        raise SystemExit(
            "ERROR: Cannot run `sage eval boundary` because the required "
            f"`{subset_tag}` subset is empty in {Path(path)}.\n"
            "Populate the canonical query bank before running the boundary "
            "guardrail benchmark."
        ) from exc


def ensure_faithfulness_cases_ready(
    path: str | Path = FAITHFULNESS_CASES_PATH,
    outcomes_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> None:
    """Fail fast when frozen calibration faithfulness artifacts are missing."""
    cases_path = Path(path)
    resolved_outcomes_path = resolve_faithfulness_case_outcomes_path(
        cases_path,
        outcomes_path=outcomes_path,
    )
    resolved_manifest_path = resolve_faithfulness_cases_manifest_path(
        cases_path,
        manifest_path=manifest_path,
    )

    try:
        load_faithfulness_cases(path=cases_path, require_nonempty=True)
        load_faithfulness_case_outcomes(
            path=resolved_outcomes_path,
            require_nonempty=True,
        )
        manifest = load_faithfulness_cases_manifest(
            path=resolved_manifest_path,
            require_nonempty=True,
        )
        if manifest.get("surface") == "dev" or (
            manifest.get("source_subset") == FAITHFULNESS_DEV_SOURCE_SUBSET
        ):
            raise FaithfulnessCasesManifestError(
                "Evaluation requires the sealed final faithfulness "
                "surface, but the selected manifest belongs to the dev "
                "explanation surface."
            )
        load_frozen_freshness_reference(
            cases_path=cases_path,
            manifest_path=resolved_manifest_path,
        )
    except (
        FaithfulnessCasesEmptyError,
        FaithfulnessCaseOutcomesEmptyError,
        FaithfulnessCasesManifestError,
        FileNotFoundError,
    ) as exc:
        raise SystemExit(
            "ERROR: Cannot run `sage eval run` because frozen faithfulness cases, "
            "coverage artifacts, and the freeze-time manifest are not ready.\n"
            f"Cases path: {cases_path}\n"
            f"Outcomes path: {resolved_outcomes_path}\n"
            f"Manifest path: {resolved_manifest_path}\n"
            f"Reason: {exc}\n"
            "Materialize calibration explanation cases before running the full "
            "evaluation workflow.\n"
            "Suggested command:\n"
            "  python scripts/materialize_faithfulness_cases.py"
        ) from exc
