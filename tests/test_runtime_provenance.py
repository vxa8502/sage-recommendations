"""Tests for sage.services.runtime_provenance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sage.services import runtime_provenance


def test_build_run_provenance_preserves_explicit_empty_gate_config() -> None:
    provenance = runtime_provenance.build_run_provenance(gate_config={})

    assert provenance["current_gate_config"] == {}


def test_build_run_provenance_normalizes_explicit_retrieval_profile() -> None:
    provenance = runtime_provenance.build_run_provenance(
        retrieval_profile=" Eval Unfiltered "
    )

    assert provenance["retrieval_profile"] == "eval_unfiltered"


def test_build_run_provenance_rejects_invalid_retrieval_profile() -> None:
    with pytest.raises(
        ValueError,
        match="retrieval profile label must contain letters or numbers",
    ):
        runtime_provenance.build_run_provenance(retrieval_profile="!!!")


def test_current_retrieval_config_uses_canonical_live_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_provenance, "RUNTIME_RETRIEVAL_MIN_RATING", 4.0)
    monkeypatch.setattr(
        runtime_provenance,
        "RUNTIME_RETRIEVAL_AGGREGATION",
        "weighted_mean",
    )

    assert runtime_provenance.current_retrieval_config() == {
        "aggregation": "weighted_mean",
        "min_rating": 4.0,
        "retrieval_profile": "rating_gte_4_aggregation_weighted_mean",
    }


def test_current_explainer_identity_keeps_client_pair_together() -> None:
    explainer = SimpleNamespace(
        provider="wrapper-provider",
        model="wrapper-model",
        client=SimpleNamespace(provider="OPENAI", model="client-model"),
    )

    assert runtime_provenance.current_explainer_identity(explainer) == {
        "provider": "openai",
        "model": "client-model",
    }


def test_current_explainer_identity_uses_complete_wrapper_when_client_is_partial() -> None:
    explainer = SimpleNamespace(
        provider="wrapper-provider",
        model="wrapper-model",
        client=SimpleNamespace(provider=None, model="client-model"),
    )

    assert runtime_provenance.current_explainer_identity(explainer) == {
        "provider": "wrapper-provider",
        "model": "wrapper-model",
    }


def test_current_explainer_identity_combines_split_partial_sources() -> None:
    explainer = SimpleNamespace(
        provider="WRAPPER-PROVIDER",
        model=None,
        client=SimpleNamespace(provider=None, model="client-model"),
    )

    assert runtime_provenance.current_explainer_identity(explainer) == {
        "provider": "wrapper-provider",
        "model": "client-model",
    }


def test_current_explainer_identity_defaults_model_from_partial_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_provenance, "OPENAI_MODEL", "gpt-test")
    explainer = SimpleNamespace(
        provider=None,
        model=None,
        client=SimpleNamespace(provider="OPENAI", model=None),
    )

    assert runtime_provenance.current_explainer_identity(explainer) == {
        "provider": "openai",
        "model": "gpt-test",
    }


def test_current_explainer_identity_uses_configured_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_provenance, "LLM_PROVIDER", "OPENAI")
    monkeypatch.setattr(runtime_provenance, "OPENAI_MODEL", "gpt-test")

    assert runtime_provenance.current_explainer_identity() == {
        "provider": "openai",
        "model": "gpt-test",
    }
