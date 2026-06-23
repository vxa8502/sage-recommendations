"""Typed payload aliases and dataclasses for split leakage audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


JsonObject: TypeAlias = dict[str, Any]
QueryEntry: TypeAlias = JsonObject
ComponentPayload: TypeAlias = JsonObject
FlaggedPair: TypeAlias = JsonObject
PairAuditPayload: TypeAlias = JsonObject
SurfaceSpecInput: TypeAlias = JsonObject
Severity: TypeAlias = Literal[
    "exact_duplicate",
    "high_confidence_near_duplicate",
    "semantic_watchlist",
]
RiskLevel: TypeAlias = Literal["low", "moderate", "high"]
SignalName: TypeAlias = Literal[
    "semantic",
    "token_jaccard",
    "character_trigram_jaccard",
    "relevant_item_coverage",
]


@dataclass(frozen=True)
class SemanticMetadata:
    mode: str
    model_name: str | None


@dataclass(frozen=True)
class PairMetrics:
    semantic_cosine: float
    token_jaccard: float
    character_trigram_jaccard: float
    shared_relevant_item_count: int
    relevant_item_coverage: float
    exact_duplicate: bool


@dataclass(frozen=True)
class PairSignals:
    strong_semantic: bool
    watch_semantic: bool
    strong_token: bool
    watch_token: bool
    strong_trigram: bool
    watch_trigram: bool
    strong_relevant: bool
    watch_relevant: bool

    @property
    def strong_names(self) -> list[SignalName]:
        return _active_signal_names(
            (
                ("semantic", self.strong_semantic),
                ("token_jaccard", self.strong_token),
                ("character_trigram_jaccard", self.strong_trigram),
                ("relevant_item_coverage", self.strong_relevant),
            )
        )

    @property
    def watch_names(self) -> list[SignalName]:
        return _active_signal_names(
            (
                ("semantic", self.watch_semantic),
                ("token_jaccard", self.watch_token),
                ("character_trigram_jaccard", self.watch_trigram),
                ("relevant_item_coverage", self.watch_relevant),
            )
        )


@dataclass(frozen=True)
class PairDecision:
    severity: Severity
    rationale: str


def _active_signal_names(
    signal_flags: tuple[tuple[SignalName, bool], ...],
) -> list[SignalName]:
    return [name for name, active in signal_flags if active]
