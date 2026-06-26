"""
LLM client adapters.

Provides unified interface for LLM providers (Anthropic Claude, OpenAI GPT).

Includes exponential backoff with jitter for rate limit handling:
- Initial delay: 1 second
- Max delay: 60 seconds
- Jitter: 0-25% random variation
- Max retries: configurable (default 3 for rate limits)
"""

import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, NoReturn, Protocol, TypeVar
from collections.abc import Callable, Iterator

from sage.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    get_logger,
)
from sage.utils import calculate_exponential_backoff_delay, require_import

logger = get_logger(__name__)

T = TypeVar("T")

# Exponential backoff settings for rate limits
RATE_LIMIT_INITIAL_DELAY = 1.0  # seconds
RATE_LIMIT_MAX_DELAY = 60.0  # seconds
RATE_LIMIT_MAX_RETRIES = 3  # additional retries for rate limits
RATE_LIMIT_JITTER = 0.25  # 25% random jitter
LLM_CLIENT_FACTORIES = {
    PROVIDER_ANTHROPIC: lambda: AnthropicClient(),
    PROVIDER_OPENAI: lambda: OpenAIClient(),
}


class LLMRateLimitError(RuntimeError):
    """Raised when the provider rejects a request due to rate limiting."""


def _log_rate_limit_retry(*, attempt: int, error: Exception) -> None:
    """Log and sleep for a rate-limited retry attempt."""
    delay = calculate_exponential_backoff_delay(
        initial_delay=RATE_LIMIT_INITIAL_DELAY,
        attempt=attempt,
        max_delay=RATE_LIMIT_MAX_DELAY,
        jitter=RATE_LIMIT_JITTER,
    )
    logger.warning(
        "Rate limited (attempt %d/%d), backing off %.1fs: %s",
        attempt + 1,
        RATE_LIMIT_MAX_RETRIES + 1,
        delay,
        error,
    )
    time.sleep(delay)


def _extract_text_blocks(blocks: object) -> str:
    """Join all provider text blocks, ignoring non-text content blocks."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def with_rate_limit_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for retrying on rate limit errors with exponential backoff.

    Wraps LLM generate methods to handle rate limit errors gracefully.
    Uses exponential backoff with jitter to avoid thundering herd.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        last_exception: LLMRateLimitError | None = None

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return func(self, *args, **kwargs)
            except LLMRateLimitError as e:
                last_exception = e

                if attempt < RATE_LIMIT_MAX_RETRIES:
                    _log_rate_limit_retry(attempt=attempt, error=e)
                else:
                    logger.error(
                        "Rate limit persists after %d retries: %s",
                        RATE_LIMIT_MAX_RETRIES + 1,
                        e,
                    )

        # All retries exhausted
        raise last_exception  # type: ignore[misc]

    return wrapper


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """
    Protocol for LLM clients (Anthropic or OpenAI).

    Implementations must provide at least the generate() method.
    Streaming support via generate_stream() is optional.
    """

    provider: str
    model: str

    def generate(self, system: str, user: str) -> tuple[str, int]:
        """
        Generate a response from the LLM.

        Args:
            system: System prompt setting context and instructions.
            user: User prompt with the actual request.

        Returns:
            Tuple of (generated_text, tokens_used).
        """
        ...

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """
        Stream response tokens from the LLM.

        Args:
            system: System prompt setting context and instructions.
            user: User prompt with the actual request.

        Yields:
            Generated tokens as they become available.
        """
        ...


# ---------------------------------------------------------------------------
# Base class with shared logic
# ---------------------------------------------------------------------------


class LLMClientBase(ABC):
    """Base class with shared initialization and error handling."""

    client: Any
    provider: str
    model: str
    temperature: float
    max_tokens: int
    _sdk: Any
    _name: str
    _api_errors: tuple[type[Exception], ...]

    def _init_common(
        self,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
        sdk: Any,
        name: str,
        api_errors: tuple[type[Exception], ...],
    ) -> None:
        """Initialize common attributes."""
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sdk = sdk
        self._name = name
        self._api_errors = api_errors

    def _translate_error(self, exc: Exception) -> NoReturn:
        """Translate SDK-specific API errors to built-in exceptions."""
        if isinstance(exc, self._sdk.APITimeoutError):
            raise TimeoutError(f"{self._name} API request timed out: {exc}") from exc
        if isinstance(exc, self._sdk.RateLimitError):
            raise LLMRateLimitError(f"{self._name} API rate limited: {exc}") from exc
        if isinstance(exc, self._sdk.APIConnectionError):
            raise ConnectionError(
                f"Failed to connect to {self._name} API: {exc}"
            ) from exc
        raise exc

    def _stream_with_rate_limit_retry(
        self,
        token_factory: Callable[[], Iterator[str]],
    ) -> Iterator[str]:
        """Retry rate-limited stream setup before any tokens have been emitted."""
        last_exception: LLMRateLimitError | None = None

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            emitted_any = False
            try:
                for token in token_factory():
                    emitted_any = True
                    yield token
                return
            except self._sdk.RateLimitError as exc:
                translated = LLMRateLimitError(f"{self._name} API rate limited: {exc}")
                if emitted_any or attempt >= RATE_LIMIT_MAX_RETRIES:
                    logger.error(
                        "Rate limit persists after %d retries: %s",
                        RATE_LIMIT_MAX_RETRIES + 1,
                        translated,
                    )
                    raise translated from exc
                last_exception = translated
                _log_rate_limit_retry(attempt=attempt, error=translated)
            except self._api_errors as exc:
                self._translate_error(exc)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"{self._name} stream retry loop exited unexpectedly.")

    @abstractmethod
    def generate(self, system: str, user: str) -> tuple[str, int]:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """Stream response tokens from the LLM."""
        ...


# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClientBase):
    """
    Anthropic Claude client for explanation generation.

    Implements the LLMClient protocol for use with Claude models.
    Supports both synchronous generation and streaming.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = ANTHROPIC_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY from config.
            model: Model ID to use. Defaults to ANTHROPIC_MODEL from config.
            temperature: Sampling temperature. Defaults to LLM_TEMPERATURE.
            max_tokens: Maximum tokens to generate. Defaults to LLM_MAX_TOKENS.
            timeout: Request timeout in seconds. Defaults to LLM_TIMEOUT.
            max_retries: Maximum retry attempts. Defaults to LLM_MAX_RETRIES.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        anthropic = require_import("anthropic")

        self.client = anthropic.Anthropic(
            api_key=api_key or ANTHROPIC_API_KEY,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._init_common(
            model=model,
            provider=PROVIDER_ANTHROPIC,
            temperature=temperature,
            max_tokens=max_tokens,
            sdk=anthropic,
            name="Anthropic",
            api_errors=(
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            ),
        )

    @with_rate_limit_retry
    def generate(self, system: str, user: str) -> tuple[str, int]:
        """
        Generate explanation using Claude.

        Args:
            system: System prompt with instructions.
            user: User prompt with query and evidence.

        Returns:
            Tuple of (generated_text, tokens_used).

        Raises:
            TimeoutError: If API request times out.
            RuntimeError: If rate limited (after retries exhausted).
            ConnectionError: If connection fails.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = _extract_text_blocks(response.content)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens
        except self._api_errors as exc:
            self._translate_error(exc)

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """
        Stream explanation tokens using Claude.

        Args:
            system: System prompt with instructions.
            user: User prompt with query and evidence.

        Yields:
            Generated tokens as they become available.

        Raises:
            TimeoutError: If API request times out.
            RuntimeError: If rate limited.
            ConnectionError: If connection fails.
        """

        def token_factory() -> Iterator[str]:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                yield from stream.text_stream

        yield from self._stream_with_rate_limit_retry(token_factory)


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClientBase):
    """
    OpenAI client for explanation generation.

    Implements the LLMClient protocol for use with GPT models.
    Supports both synchronous generation and streaming.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = OPENAI_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY from config.
            model: Model ID to use. Defaults to OPENAI_MODEL from config.
            temperature: Sampling temperature. Defaults to LLM_TEMPERATURE.
            max_tokens: Maximum tokens to generate. Defaults to LLM_MAX_TOKENS.
            timeout: Request timeout in seconds. Defaults to LLM_TIMEOUT.
            max_retries: Maximum retry attempts. Defaults to LLM_MAX_RETRIES.

        Raises:
            ImportError: If openai package is not installed.
        """
        openai = require_import("openai")
        OpenAI = openai.OpenAI

        self.client = OpenAI(
            api_key=api_key or OPENAI_API_KEY,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._init_common(
            model=model,
            provider=PROVIDER_OPENAI,
            temperature=temperature,
            max_tokens=max_tokens,
            sdk=openai,
            name="OpenAI",
            api_errors=(
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.APIConnectionError,
            ),
        )

    @with_rate_limit_retry
    def generate(self, system: str, user: str) -> tuple[str, int]:
        """
        Generate explanation using GPT.

        Args:
            system: System prompt with instructions.
            user: User prompt with query and evidence.

        Returns:
            Tuple of (generated_text, tokens_used).

        Raises:
            TimeoutError: If API request times out.
            RuntimeError: If rate limited (after retries exhausted).
            ConnectionError: If connection fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return text, tokens
        except self._api_errors as exc:
            self._translate_error(exc)

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """
        Stream explanation tokens using GPT.

        Args:
            system: System prompt with instructions.
            user: User prompt with query and evidence.

        Yields:
            Generated tokens as they become available.

        Raises:
            TimeoutError: If API request times out.
            RuntimeError: If rate limited.
            ConnectionError: If connection fails.
        """

        def token_factory() -> Iterator[str]:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        yield from self._stream_with_rate_limit_retry(token_factory)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_llm_client(provider: str | None = None) -> LLMClient:
    """
    Get the configured LLM client.

    Args:
        provider: LLM provider (PROVIDER_ANTHROPIC or PROVIDER_OPENAI).
            Defaults to LLM_PROVIDER from config.

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    normalized_provider = (provider or LLM_PROVIDER).strip().lower()
    factory = LLM_CLIENT_FACTORIES.get(normalized_provider)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider: {normalized_provider}. "
            f"Use '{PROVIDER_ANTHROPIC}' or '{PROVIDER_OPENAI}'."
        )
    return factory()


__all__ = [
    "LLMClient",
    "LLMClientBase",
    "AnthropicClient",
    "OpenAIClient",
    "get_llm_client",
    "with_rate_limit_retry",
    "LLMRateLimitError",
    "RATE_LIMIT_MAX_RETRIES",
]
