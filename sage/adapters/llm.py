"""
LLM client adapters.

Provides unified interface for LLM providers (Anthropic Claude, OpenAI GPT).

Includes exponential backoff with jitter for rate limit handling:
- Initial delay: 1 second
- Max delay: 60 seconds
- Jitter: 0-25% random variation
- Max retries: configurable (default 3 for rate limits)
"""

import random
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Iterator, NoReturn, Protocol, TypeVar

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
from sage.utils import require_import

logger = get_logger(__name__)

T = TypeVar("T")

# Exponential backoff settings for rate limits
RATE_LIMIT_INITIAL_DELAY = 1.0  # seconds
RATE_LIMIT_MAX_DELAY = 60.0  # seconds
RATE_LIMIT_MAX_RETRIES = 3  # additional retries for rate limits
RATE_LIMIT_JITTER = 0.25  # 25% random jitter


def _calculate_backoff_delay(attempt: int, jitter: float = RATE_LIMIT_JITTER) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current retry attempt (0-indexed).
        jitter: Maximum jitter factor (0.25 = up to 25% variation).

    Returns:
        Delay in seconds.
    """
    base_delay = RATE_LIMIT_INITIAL_DELAY * (2**attempt)
    delay = min(base_delay, RATE_LIMIT_MAX_DELAY)
    # Add random jitter to prevent thundering herd
    jitter_amount = delay * jitter * random.random()
    return delay + jitter_amount


def with_rate_limit_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for retrying on rate limit errors with exponential backoff.

    Wraps LLM generate methods to handle rate limit errors gracefully.
    Uses exponential backoff with jitter to avoid thundering herd.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        last_exception = None

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as e:
                # Check if this is a rate limit error (translated from SDK)
                if "rate limit" not in str(e).lower():
                    raise

                last_exception = e

                if attempt < RATE_LIMIT_MAX_RETRIES:
                    delay = _calculate_backoff_delay(attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d), backing off %.1fs: %s",
                        attempt + 1,
                        RATE_LIMIT_MAX_RETRIES + 1,
                        delay,
                        e,
                    )
                    time.sleep(delay)
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
    model: str
    temperature: float
    max_tokens: int
    _sdk: Any
    _name: str
    _api_errors: tuple[type[Exception], ...]

    def _init_common(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        sdk: Any,
        name: str,
        api_errors: tuple[type[Exception], ...],
    ) -> None:
        """Initialize common attributes."""
        self.model = model
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
            raise RuntimeError(f"{self._name} API rate limited: {exc}") from exc
        if isinstance(exc, self._sdk.APIConnectionError):
            raise ConnectionError(
                f"Failed to connect to {self._name} API: {exc}"
            ) from exc
        raise exc

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
            # Extract text from first TextBlock
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text
                    break
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
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except self._api_errors as exc:
            self._translate_error(exc)


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
        try:
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
        except self._api_errors as exc:
            self._translate_error(exc)


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
    provider = provider.lower().strip() if provider else LLM_PROVIDER

    if provider == PROVIDER_ANTHROPIC:
        return AnthropicClient()
    elif provider == PROVIDER_OPENAI:
        return OpenAIClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Use '{PROVIDER_ANTHROPIC}' or '{PROVIDER_OPENAI}'."
        )


__all__ = [
    "LLMClient",
    "LLMClientBase",
    "AnthropicClient",
    "OpenAIClient",
    "get_llm_client",
    "with_rate_limit_retry",
    "RATE_LIMIT_MAX_RETRIES",
]
