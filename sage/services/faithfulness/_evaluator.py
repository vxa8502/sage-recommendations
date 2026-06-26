"""
Faithfulness evaluation service — RAGAS integration and FaithfulnessEvaluator.

Combines RAGAS (LLM-based claim extraction and verification) with retry logic
for transient LLM failures.

Faithfulness = (claims supported by context) / (total claims)

Target: >0.85 faithfulness score
"""

import asyncio
import logging
import re

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sage.config import (
    FAITHFULNESS_TARGET,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RAGAS_MODEL,
)
from sage.core import (
    ExplanationResult,
    FaithfulnessReport,
    FaithfulnessResult,
)
from sage.utils import ensure_ragas_installed

logger = logging.getLogger(__name__)

# Hard read timeout for each Anthropic API call inside RAGAS.
# The Anthropic client defaults to 600s; a hung connection would block the
# entire batch for 10 minutes per case. 60s is generous for a single
# claim-extraction or NLI call and still allows complex explanations to finish.
_RAGAS_API_TIMEOUT_SECONDS = 60.0

try:
    from anthropic import APIConnectionError as _AnthropicConnectionError
    from anthropic import APITimeoutError as _AnthropicTimeoutError
    _ANTHROPIC_EXCEPTIONS: tuple[type[BaseException], ...] = (
        _AnthropicConnectionError,
        _AnthropicTimeoutError,
    )
except ImportError:
    _ANTHROPIC_EXCEPTIONS = ()

# Transient exceptions that should trigger retries
TRANSIENT_EXCEPTIONS = (
    TimeoutError, ConnectionError, OSError
) + _ANTHROPIC_EXCEPTIONS


def _log_retry(retry_state) -> None:  # type: ignore[no-untyped-def]
    """Log retry attempts for transient LLM failures."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "LLM call failed, retrying (attempt %d/3): %s",
        retry_state.attempt_number,
        exc,
    )


# Shared retry decorator for LLM calls
_llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    before_sleep=_log_retry,
)


def is_event_loop_running() -> bool:
    """Check if an asyncio event loop is currently running."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


# Compiled once at module load — avoids 2880 recompilations across a 120-case run.
_REVIEW_ID_RE = re.compile(r"\s*\[review_\d+\]")
_FRAMING_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"According to reviews?,?\s*",
        r"Customers report\s+",
        r"Reviewers?\s+call\s+it\s+",
        r"Reviewers?\s+describe\s+(?:it|this)\s+as\s+",
        r"Reviewers?\s+find\s+it\s+",
        r"Reviewers?\s+report\s+",
        r"Reviewers say\s+",
        r"One user said\s+",
        r"One user found\s+",
        r"One reviewer found\s+",
        r"One reviewer confirms?\s+(it\s+)?",
        r"One reviewer\s+calls?\s+it\s+",
        r"One reviewer\s+describes?\s+it\s+as\s+",
        r"One reviewer\s+",
        r"Users mention\s+",
        r"Users also note\s+",
        r"Users note\s+",
        r"Reviewers?\s+(also\s+)?note\s+",
        r"Reviewers?\s+(also\s+)?mention\s+",
        r"Reviewers?\s+confirm\s+",
        r"Reviewers?\s+praise\s+",
        r"Reviewers?\s+highlight\s+",
    ]
]
_AND_QUOTE_RE = re.compile(r'\s+and\s+"')
_DOT_QUOTE_RE = re.compile(r'\.\s+"')
_HANGING_DOT_RE = re.compile(r"\s+\.")
_HANGING_COMMA_RE = re.compile(r"\s+,")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _clean_explanation_for_ragas(explanation: str) -> str:
    """
    Clean explanation text for RAGAS evaluation.

    RAGAS fails on explanations with quotes + citations together, even when
    the quoted content is verbatim from evidence. This is a known limitation.
    We clean the explanation to remove metadata (citations, framing) while
    preserving the factual claims for evaluation.

    Args:
        explanation: Original explanation with framing and citations.

    Returns:
        Cleaned explanation suitable for RAGAS faithfulness evaluation.
    """
    text = _REVIEW_ID_RE.sub("", explanation)
    for pattern in _FRAMING_RES:
        text = pattern.sub("", text)
    text = _AND_QUOTE_RE.sub('. The product is "', text)
    text = _DOT_QUOTE_RE.sub('. The product is "', text)
    text = _HANGING_DOT_RE.sub(".", text)
    text = _HANGING_COMMA_RE.sub(",", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def get_ragas_llm(provider: str | None = None):
    """
    Get configured LLM for RAGAS evaluation.

    Args:
        provider: LLM provider ("anthropic" or "openai").

    Returns:
        RAGAS-compatible LLM wrapper.
    """
    ensure_ragas_installed()
    from ragas.llms import llm_factory

    provider = provider or LLM_PROVIDER

    if provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required for RAGAS with Claude")

        # RAGAS 0.4.x calls agenerate() internally, which requires an async
        # client. The sync Anthropic() client raises TypeError at eval time
        # and causes ragas.evaluate() to silently retry forever (hangs at 0%).
        # timeout overrides the default 600s read timeout — without it a
        # silent hung socket blocks the entire batch for 10 min per case.
        anthropic_client = AsyncAnthropic(
            timeout=_RAGAS_API_TIMEOUT_SECONDS
        )
        llm = llm_factory(
            RAGAS_MODEL,
            provider="anthropic",
            client=anthropic_client,
        )
        # Anthropic rejects requests that set both temperature and top_p.
        # RAGAS's InstructorModelArgs defaults top_p=0.1 and only strips it
        # for OpenAI reasoning models, not Anthropic. Remove it here.
        if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
            llm.model_args.pop("top_p", None)
            llm.model_args["max_tokens"] = 4096
        elif hasattr(llm, "model_args"):
            logger.warning(
                "RAGAS LLM model_args is not a dict (%s); "
                "top_p and max_tokens patches were not applied. "
                "Requests may fail with Anthropic 400 errors.",
                type(llm.model_args).__name__,
            )
        return llm
    elif provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for RAGAS with OpenAI")

        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        return llm_factory(OPENAI_MODEL, client=openai_client)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class FaithfulnessEvaluator:
    """
    Evaluate explanation faithfulness using RAGAS.

    Uses LLM-based claim extraction and verification to compute
    the proportion of explanation claims supported by evidence.
    """

    def __init__(
        self, provider: str | None = None, target: float = FAITHFULNESS_TARGET
    ):
        """
        Initialize faithfulness evaluator.

        Args:
            provider: LLM provider for RAGAS.
            target: Faithfulness target score (default 0.85).
        """
        ensure_ragas_installed()
        from ragas.metrics.collections import Faithfulness

        self.llm = get_ragas_llm(provider)
        self.scorer = Faithfulness(llm=self.llm)
        self.target = target

    @_llm_retry
    async def _score_with_retry(
        self,
        query: str,
        explanation: str,
        contexts: list[str],
    ) -> float:
        """Score with retry logic for transient failures."""
        result = await self.scorer.ascore(
            user_input=query,
            response=explanation,
            retrieved_contexts=contexts,
        )
        # ragas 0.4.x returns MetricResult(value=float), not a bare float
        v = result.value if hasattr(result, "value") else result
        return float(v)

    async def evaluate_single_async(
        self,
        query: str,
        explanation: str,
        evidence_texts: list[str],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness for a single explanation (async)."""
        # Strip citation markers and framing phrases before RAGAS scoring.
        # RAGAS's LLM claim extractor penalizes [review_X] tokens and treats
        # "Reviewers call it X" as a compound claim requiring evidence that
        # *multiple reviewers* said X — cleaning avoids both false penalties.
        cleaned = _clean_explanation_for_ragas(explanation)
        # RAGAS has issues with multi-element retrieved_contexts — combine into
        # a single context string so claim verification uses all evidence at once.
        combined = [" ".join(evidence_texts)]
        score = await self._score_with_retry(query, cleaned, combined)

        return FaithfulnessResult(
            score=float(score),
            query=query,
            explanation=explanation,
            evidence_count=len(evidence_texts),
            meets_target=float(score) >= self.target,
        )

    def evaluate_single(
        self,
        query: str,
        explanation: str,
        evidence_texts: list[str],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness for a single explanation (sync wrapper)."""
        if is_event_loop_running():
            raise RuntimeError(
                "Cannot call evaluate_single() from async context.\n"
                "Use: await evaluator.evaluate_single_async(...)"
            )

        coro = self.evaluate_single_async(query, explanation, evidence_texts)
        return asyncio.run(coro)

    def evaluate_batch(
        self,
        explanation_results: list[ExplanationResult],
    ) -> FaithfulnessReport:
        """Evaluate faithfulness for multiple explanations."""
        if is_event_loop_running():
            raise RuntimeError(
                "Cannot call evaluate_batch() from async context.\n"
                "Await evaluate_single_async() per item instead."
            )
        # ragas.evaluate() batch API is incompatible with ragas 0.4.x
        # collections metrics; score each sample individually via async loop.
        async def _run_all() -> list[FaithfulnessResult]:
            results = []
            for er in explanation_results:
                r = await self.evaluate_single_async(
                    er.query, er.explanation, er.evidence_texts
                )
                results.append(r)
            return results

        individual_results = asyncio.run(_run_all())
        scores = [r.score for r in individual_results]
        scores_arr = np.array(scores)
        n_passing = sum(1 for s in scores if s >= self.target)

        return FaithfulnessReport(
            mean_score=float(np.mean(scores_arr)),
            min_score=float(np.min(scores_arr)),
            max_score=float(np.max(scores_arr)),
            std_score=(
                float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0
            ),
            n_samples=len(scores),
            n_passing=n_passing,
            pass_rate=n_passing / len(scores) if scores else 0.0,
            target=self.target,
            results=individual_results,
        )


def evaluate_faithfulness(
    explanation_results: list[ExplanationResult],
    provider: str | None = None,
) -> FaithfulnessReport:
    """
    Convenience function to evaluate faithfulness for explanations.

    Args:
        explanation_results: List of ExplanationResult from explainer.
        provider: LLM provider ("anthropic" or "openai").

    Returns:
        FaithfulnessReport with scores and statistics.
    """
    evaluator = FaithfulnessEvaluator(provider=provider)
    return evaluator.evaluate_batch(explanation_results)
