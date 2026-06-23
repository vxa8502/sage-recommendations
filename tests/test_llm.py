from collections.abc import Iterator

import pytest

from sage.adapters.llm import (
    LLMClientBase,
    LLMRateLimitError,
    get_llm_client,
    _extract_text_blocks,
    with_rate_limit_retry,
)


class _FakeSDK:
    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class APIConnectionError(Exception):
        pass


class _DummyLLMClient(LLMClientBase):
    def __init__(self) -> None:
        self._init_common(
            model="test-model",
            provider="test-provider",
            temperature=0.0,
            max_tokens=10,
            sdk=_FakeSDK,
            name="Dummy",
            api_errors=(
                _FakeSDK.APITimeoutError,
                _FakeSDK.RateLimitError,
                _FakeSDK.APIConnectionError,
            ),
        )

    def generate(self, system: str, user: str) -> tuple[str, int]:
        raise NotImplementedError

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        raise NotImplementedError


def test_with_rate_limit_retry_retries_typed_rate_limit_errors(monkeypatch) -> None:
    monkeypatch.setattr("sage.adapters.llm.time.sleep", lambda _: None)

    class _RetryingOperation:
        def __init__(self) -> None:
            self.calls = 0

        @with_rate_limit_retry
        def run(self) -> str:
            self.calls += 1
            if self.calls == 1:
                raise LLMRateLimitError("provider said slow down")
            return "ok"

    operation = _RetryingOperation()
    assert operation.run() == "ok"
    assert operation.calls == 2


def test_stream_retry_retries_when_rate_limited_before_first_token(monkeypatch) -> None:
    monkeypatch.setattr("sage.adapters.llm.time.sleep", lambda _: None)
    client = _DummyLLMClient()
    attempts = {"count": 0}

    def token_factory() -> Iterator[str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _FakeSDK.RateLimitError("try again later")
        yield "hello"
        yield " world"

    assert list(client._stream_with_rate_limit_retry(token_factory)) == [
        "hello",
        " world",
    ]
    assert attempts["count"] == 2


def test_stream_retry_does_not_restart_after_tokens_have_been_emitted(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sage.adapters.llm.time.sleep", lambda _: None)
    client = _DummyLLMClient()

    def token_factory() -> Iterator[str]:
        yield "partial"
        raise _FakeSDK.RateLimitError("mid-stream limit")

    with pytest.raises(LLMRateLimitError, match="rate limited"):
        list(client._stream_with_rate_limit_retry(token_factory))


def test_extract_text_blocks_joins_all_text_fragments() -> None:
    blocks = [
        type("TextBlock", (), {"text": "hello"})(),
        type("NonTextBlock", (), {})(),
        type("TextBlock", (), {"text": " world"})(),
    ]

    assert _extract_text_blocks(blocks) == "hello world"


def test_get_llm_client_normalizes_provider_names(monkeypatch) -> None:
    class _SentinelClient:
        pass

    sentinel = _SentinelClient()
    monkeypatch.setattr(
        "sage.adapters.llm.LLM_CLIENT_FACTORIES",
        {
            "anthropic": lambda: sentinel,
            "openai": lambda: object(),
        },
    )

    assert get_llm_client("  ANTHROPIC ") is sentinel
