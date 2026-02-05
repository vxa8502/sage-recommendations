"""
LLM client adapters.

Provides unified interface for LLM providers (Anthropic Claude, OpenAI GPT).
"""

from typing import Iterator, NoReturn, Protocol

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
)


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
# Shared error translation
# ---------------------------------------------------------------------------


def _translate_api_error(exc: Exception, sdk, name: str) -> NoReturn:
    """Translate SDK-specific API errors to built-in exceptions.

    Both Anthropic and OpenAI SDKs expose the same three error types.
    This function maps them to standard Python exceptions so callers
    don't need SDK-specific imports.
    """
    if isinstance(exc, sdk.APITimeoutError):
        raise TimeoutError(f"{name} API request timed out: {exc}") from exc
    if isinstance(exc, sdk.RateLimitError):
        raise RuntimeError(f"{name} API rate limited: {exc}") from exc
    if isinstance(exc, sdk.APIConnectionError):
        raise ConnectionError(f"Failed to connect to {name} API: {exc}") from exc
    raise exc


# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------


class AnthropicClient:
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
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(
            api_key=api_key or ANTHROPIC_API_KEY,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sdk = anthropic
        self._name = "Anthropic"
        self._api_errors = (
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        )

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
            RuntimeError: If rate limited.
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
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens
        except self._api_errors as exc:
            _translate_api_error(exc, self._sdk, self._name)

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
            _translate_api_error(exc, self._sdk, self._name)


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------


class OpenAIClient:
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
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.client = OpenAI(
            api_key=api_key or OPENAI_API_KEY,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sdk = openai
        self._name = "OpenAI"
        self._api_errors = (
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIConnectionError,
        )

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
            RuntimeError: If rate limited.
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
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return text, tokens
        except self._api_errors as exc:
            _translate_api_error(exc, self._sdk, self._name)

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
            _translate_api_error(exc, self._sdk, self._name)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_llm_client(provider: str | None = None) -> LLMClient:
    """
    Get the configured LLM client.

    Args:
        provider: LLM provider ("anthropic" or "openai").
            Defaults to LLM_PROVIDER from config.

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    provider = provider or LLM_PROVIDER

    if provider == "anthropic":
        return AnthropicClient()
    elif provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic' or 'openai'.")


__all__ = [
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "get_llm_client",
]
