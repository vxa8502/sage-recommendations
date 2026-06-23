"""Subset-role policy for evidence-gate holdout runs."""

from __future__ import annotations

from dataclasses import dataclass

from sage.data.query_bank.sources.esci._config import DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG

DEFAULT_SUBSETS = (DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG,)
PROMOTION_HOLDOUT_SUBSETS = frozenset({DEFAULT_RETRIEVAL_DEV_HOLDOUT_SUBSET_TAG})
DIAGNOSTIC_ONLY_SUBSETS = frozenset({"faithfulness_dev_seed"})
PROMOTION_HOLDOUT_ROLE = "promotion_holdout"
DIAGNOSTIC_NON_PROMOTION_ROLE = "diagnostic_non_promotion"
CUSTOM_NON_PROMOTION_ROLE = "custom_non_promotion"


@dataclass(frozen=True, slots=True)
class SubsetRoleDefinition:
    role: str
    promotion_eligible: bool
    note: str


@dataclass(frozen=True, slots=True)
class SubsetEvaluationPolicy:
    evaluated_subsets: tuple[str, ...]
    subset_roles: dict[str, str]

    @classmethod
    def from_subsets(cls, subsets: tuple[str, ...]) -> "SubsetEvaluationPolicy":
        return cls(
            evaluated_subsets=subsets,
            subset_roles={subset: _classify_subset_role(subset) for subset in subsets},
        )

    def role_for(self, subset: str) -> str:
        return self.subset_roles[subset]

    def role_definition_for(self, subset: str) -> SubsetRoleDefinition:
        return SUBSET_ROLE_DEFINITIONS[self.role_for(subset)]

    def subsets_with_role(self, role: str) -> list[str]:
        return [
            subset
            for subset in self.evaluated_subsets
            if self.subset_roles[subset] == role
        ]

    @property
    def promotion_eligible_subsets(self) -> list[str]:
        return self.subsets_with_role(PROMOTION_HOLDOUT_ROLE)

    @property
    def diagnostic_only_subsets(self) -> list[str]:
        return self.subsets_with_role(DIAGNOSTIC_NON_PROMOTION_ROLE)

    @property
    def non_promotion_subsets(self) -> list[str]:
        return [
            subset
            for subset in self.evaluated_subsets
            if self.subset_roles[subset] != PROMOTION_HOLDOUT_ROLE
        ]

    @property
    def custom_non_promotion_subsets(self) -> list[str]:
        return self.subsets_with_role(CUSTOM_NON_PROMOTION_ROLE)

    @property
    def promotion_eligible(self) -> bool:
        return bool(self.promotion_eligible_subsets)

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated_subsets": list(self.evaluated_subsets),
            "promotion_eligible_subsets": self.promotion_eligible_subsets,
            "diagnostic_only_subsets": self.diagnostic_only_subsets,
            "non_promotion_subsets": self.non_promotion_subsets,
            "custom_non_promotion_subsets": self.custom_non_promotion_subsets,
            "subset_roles": dict(self.subset_roles),
            "promotion_eligible": self.promotion_eligible,
            "role_policy": role_policy_payload(),
            "promotion_policy_note": (
                "Only subsets labeled `promotion_holdout` may justify calibration gate "
                "promotion. Diagnostic and custom subsets are observational only."
            ),
        }


SUBSET_ROLE_DEFINITIONS = {
    PROMOTION_HOLDOUT_ROLE: SubsetRoleDefinition(
        role=PROMOTION_HOLDOUT_ROLE,
        promotion_eligible=True,
        note=(
            "This subset is promotion-eligible and may justify a gate decision "
            "during calibration."
        ),
    ),
    DIAGNOSTIC_NON_PROMOTION_ROLE: SubsetRoleDefinition(
        role=DIAGNOSTIC_NON_PROMOTION_ROLE,
        promotion_eligible=False,
        note=(
            "This subset is diagnostic only. It may provide extra context before "
            "case freezing, but it must not justify gate promotion because it is "
            "reserved for later explanation-case materialization."
        ),
    ),
    CUSTOM_NON_PROMOTION_ROLE: SubsetRoleDefinition(
        role=CUSTOM_NON_PROMOTION_ROLE,
        promotion_eligible=False,
        note=(
            "This subset is not part of the canonical calibration promotion surface. "
            "Interpret it manually and do not treat it as promotion-eligible unless "
            "the experiment policy is updated explicitly."
        ),
    ),
}
CANONICAL_ROLE_SUBSETS = {
    PROMOTION_HOLDOUT_ROLE: PROMOTION_HOLDOUT_SUBSETS,
    DIAGNOSTIC_NON_PROMOTION_ROLE: DIAGNOSTIC_ONLY_SUBSETS,
}
SUBSET_ROLE_BY_TAG = {
    subset: role
    for role, role_subsets in CANONICAL_ROLE_SUBSETS.items()
    for subset in role_subsets
}


def _classify_subset_role(subset_tag: str) -> str:
    return SUBSET_ROLE_BY_TAG.get(subset_tag, CUSTOM_NON_PROMOTION_ROLE)


def role_policy_payload() -> dict[str, object]:
    return {
        PROMOTION_HOLDOUT_ROLE: sorted(PROMOTION_HOLDOUT_SUBSETS),
        DIAGNOSTIC_NON_PROMOTION_ROLE: sorted(DIAGNOSTIC_ONLY_SUBSETS),
        CUSTOM_NON_PROMOTION_ROLE: "all other explicitly requested subsets",
    }
