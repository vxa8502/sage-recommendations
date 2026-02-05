"""
HHEM (Hughes Hallucination Evaluation Model) adapter.

Wraps Vectara's HHEM model for hallucination detection.

HHEM is a T5-based classifier that detects when generated text
is not factually consistent with provided context.

Input: (premise, hypothesis) pairs where:
- premise = retrieved evidence (concatenated review texts)
- hypothesis = generated explanation to verify

Output: consistency score [0, 1] where:
- score < 0.5 = hallucinated (hypothesis NOT supported by premise)
- score >= 0.5 = consistent (hypothesis IS supported by premise)

Limitations:
- Max input length is 512 tokens. Inputs exceeding this are truncated.
- Template overhead is ~20 tokens, typical explanation ~90 tokens.
- Safe evidence budget: ~400 tokens (~3 chunks at 100 tokens each).
"""

import threading
import warnings

from sage.core import (
    ClaimResult,
    HallucinationResult,
)
from sage.config import (
    HALLUCINATION_THRESHOLD,
    HHEM_DEVICE,
    HHEM_MODEL,
    get_logger,
)

logger = get_logger(__name__)


# HHEM token limits (T5-base tokenizer)
HHEM_MAX_TOKENS = 512
HHEM_TEMPLATE_OVERHEAD = 20  # Tokens used by prompt template


class HallucinationDetector:
    """
    Detect hallucinations in generated explanations using HHEM.

    Uses Vectara's hallucination_evaluation_model to check if explanations
    are grounded in the retrieved evidence.
    """

    def __init__(
        self,
        model_name: str = HHEM_MODEL,
        device: str = HHEM_DEVICE,
        threshold: float = HALLUCINATION_THRESHOLD,
    ):
        """
        Initialize the hallucination detector.

        Args:
            model_name: HuggingFace model name for HHEM.
            device: Device to run on ("cpu" or "cuda").
            threshold: Score below this = hallucination (default 0.5).

        Raises:
            ImportError: If required packages are not installed.
        """
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification
        except ImportError as e:
            raise ImportError(
                f"Required packages missing: {e}. "
                "Install with: pip install transformers huggingface_hub safetensors"
            )

        self.threshold = threshold
        self.device = device
        self._torch = torch

        # Load HHEM config to get prompt template and foundation model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.prompt = config.prompt

        # Load inner T5 model with token classification head
        foundation_config = AutoConfig.from_pretrained(config.foundation)
        self.model = T5ForTokenClassification(foundation_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.foundation)

        # Load HHEM weights
        weights_file = hf_hub_download(model_name, "model.safetensors")
        state_dict = load_file(weights_file)

        # Remove 't5.' prefix from weight keys
        clean_dict = {
            (k[3:] if k.startswith("t5.") else k): v for k, v in state_dict.items()
        }

        # Load weights (strict=False needed due to HHEM weight naming quirks)
        missing, unexpected = self.model.load_state_dict(clean_dict, strict=False)
        if missing:
            warnings.warn(
                f"HHEM: {len(missing)} missing weights (expected for classifier head)"
            )

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device)
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.eval()

    def _format_premise(
        self,
        evidence_texts: list[str],
        hypothesis: str = "",
        prioritize_hypothesis: bool = False,
    ) -> str:
        """
        Join evidence texts into a single premise, truncating to fit
        within the 512-token HHEM limit.

        Uses the real tokenizer to measure token counts precisely.
        Greedily adds chunks in order until the budget is exhausted,
        ensuring HHEM always sees complete (not mid-word-truncated) evidence.

        Args:
            evidence_texts: List of review texts.
            hypothesis: The explanation being checked (needed to compute
                remaining token budget).
            prioritize_hypothesis: If True, reorder evidence so chunks
                containing the hypothesis text come first. Used for
                claim-level checks where a quote may come from a later
                chunk that would otherwise be dropped by the token budget.

        Returns:
            Concatenated premise string that fits within budget.
        """
        if prioritize_hypothesis and hypothesis:
            hyp_lower = hypothesis.lower()
            containing = [t for t in evidence_texts if hyp_lower in t.lower()]
            remaining = [t for t in evidence_texts if hyp_lower not in t.lower()]
            evidence_texts = containing + remaining

        hypothesis_tokens = len(self.tokenizer(hypothesis, add_special_tokens=False).input_ids)
        budget = HHEM_MAX_TOKENS - HHEM_TEMPLATE_OVERHEAD - hypothesis_tokens

        kept = []
        used = 0
        for text in evidence_texts:
            chunk_tokens = len(self.tokenizer(text, add_special_tokens=False).input_ids)
            if used + chunk_tokens > budget:
                break
            kept.append(text)
            used += chunk_tokens

        # Always keep at least the first chunk (even if over budget) so
        # HHEM has something to evaluate.
        if not kept and evidence_texts:
            kept.append(evidence_texts[0])

        return " ".join(kept)

    def _make_result(
        self, score: float, explanation: str, premise_length: int
    ) -> HallucinationResult:
        """Build a HallucinationResult from a score (DRY helper)."""
        return HallucinationResult(
            score=score,
            is_hallucinated=score < self.threshold,
            threshold=self.threshold,
            explanation=explanation,
            premise_length=premise_length,
        )

    def _predict(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """
        Run HHEM prediction on text pairs.

        Args:
            text_pairs: List of (premise, hypothesis) tuples.

        Returns:
            List of consistency scores [0, 1]. Returns 0.0 for all pairs
            on failure (graceful degradation).
        """
        try:
            pair_dicts = [{"text1": pair[0], "text2": pair[1]} for pair in text_pairs]
            inputs = self.tokenizer(
                [self.prompt.format(**pair) for pair in pair_dicts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with self._torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[:, 0, :]  # First token classification
            probs = self._torch.softmax(logits, dim=-1)
            scores = probs[:, 1]  # Probability of class 1 (consistent)

            return [float(s.item()) for s in scores]
        except Exception:
            logger.exception("HHEM prediction failed, returning low-confidence scores")
            return [0.0] * len(text_pairs)

    def check_explanation(
        self,
        evidence_texts: list[str],
        explanation: str,
    ) -> HallucinationResult:
        """
        Check if an explanation is supported by the evidence.

        Evidence is automatically truncated to fit within the 512-token
        HHEM limit using the real tokenizer.

        Args:
            evidence_texts: List of review texts used as context.
            explanation: Generated explanation to check.

        Returns:
            HallucinationResult with score and hallucination flag.
        """
        premise = self._format_premise(evidence_texts, hypothesis=explanation)
        scores = self._predict([(premise, explanation)])
        return self._make_result(scores[0], explanation, len(premise))

    def check_claims(
        self,
        evidence_texts: list[str],
        claims: list[str],
    ) -> list[ClaimResult]:
        """
        Check multiple claims against the evidence.

        Useful for claim-level analysis of explanations. Evidence is
        truncated per-claim to fit within the HHEM token budget.

        Args:
            evidence_texts: List of review texts.
            claims: Individual claims to verify.

        Returns:
            List of ClaimResult objects, one per claim.
        """
        pairs = [
            (self._format_premise(evidence_texts, hypothesis=claim, prioritize_hypothesis=True), claim)
            for claim in claims
        ]
        scores = self._predict(pairs)

        return [
            ClaimResult(claim=claim, score=score, is_hallucinated=score < self.threshold)
            for claim, score in zip(claims, scores)
        ]

    def check_batch(
        self,
        items: list[tuple[list[str], str]],
    ) -> list[HallucinationResult]:
        """
        Check multiple (evidence, explanation) pairs.

        Args:
            items: List of (evidence_texts, explanation) tuples.

        Returns:
            List of HallucinationResult objects.
        """
        pairs = [
            (self._format_premise(evidence_texts, hypothesis=explanation), explanation)
            for evidence_texts, explanation in items
        ]
        scores = self._predict(pairs)

        return [
            self._make_result(score, explanation, len(premise))
            for (premise, explanation), score in zip(pairs, scores)
        ]


# Module-level singleton
_detector: HallucinationDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> HallucinationDetector:
    """Get or create the global hallucination detector (thread-safe singleton)."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = HallucinationDetector()
    return _detector


def check_hallucination(
    evidence_texts: list[str],
    explanation: str,
) -> HallucinationResult:
    """
    Convenience function to check a single explanation.

    Args:
        evidence_texts: List of review texts used as context.
        explanation: Generated explanation to check.

    Returns:
        HallucinationResult with score and hallucination flag.
    """
    detector = get_detector()
    return detector.check_explanation(evidence_texts, explanation)
