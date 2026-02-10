# Chunking Strategy Decisions

## Strategy Overview

| Review Length | Strategy | Rationale |
|--------------|----------|-----------|
| < 200 tokens | No chunking | Most reviews are single-topic |
| 200-500 tokens | Semantic chunking (85th percentile breakpoint) | Preserves topic coherence |
| > 500 tokens | Semantic + sliding window fallback | Handles very long mixed-topic reviews |

Token estimation: ~4 chars/token (typical for English text with WordPiece tokenizers).

---

## Why Semantic Chunking?

### Failure Modes of Naive (Fixed-Window) Chunking

1. **Mid-sentence splits**: "The battery lasts 8 hours but" / "only if you disable WiFi" - the conditional is severed from the claim, causing the LLM to potentially cite "battery lasts 8 hours" without the critical qualifier.

2. **Aspect fragmentation**: A review discussing battery, then screen, then price gets randomly sliced. Retrieval for "battery life" might return a chunk containing "...great battery. The screen however is dim and..." - mixing positive battery sentiment with negative screen sentiment.

3. **Evidence dilution**: When a user asks about "noise cancellation", a chunk containing half a sentence about noise cancellation plus unrelated content about packaging provides weaker evidence than a chunk focused entirely on audio quality.

### How Semantic Chunking Improves Faithfulness

Semantic chunking uses embedding similarity between adjacent sentences to detect natural topic transitions. When a reviewer shifts from "battery life" to "build quality", sentence embeddings show a similarity drop. We split at these drops (below 85th percentile):

1. **Preserves complete arguments**: Claims stay with their evidence and qualifiers
2. **Creates topically coherent chunks**: Each chunk discusses one aspect
3. **Improves HHEM scores**: Hallucination detection works better with tight topics

---

## Worked Example

**Original review (320 tokens):**
> "I bought these headphones for my commute. The noise cancellation is exceptional - it blocks out subway noise completely, even announcements. I tested it on a plane and the engine drone disappeared. However, the comfort is a different story. After 2 hours my ears hurt from the pressure. The headband also feels cheap and creaks when I move."

**Naive chunking (150 tokens/chunk):**
- Chunk 1: "...exceptional - it blocks out subway noise completely, even announcements. I tested it on a"
- Chunk 2: "plane and the engine drone disappeared. However, the comfort is..."

**Semantic chunking:**
- Chunk 1: Complete noise cancellation evidence (subway + plane tests together)
- Chunk 2: Complete comfort critique (ears + headband together)

The semantic version keeps the complete noise cancellation evidence together for stronger grounding.

---

## Mixed Sentiment Handling

**Example:** "Battery life is amazing but the build quality is garbage"

**Does the chunker split this?** No - intentionally.

**Arguments against splitting (why we chose this):**
- Splitting mid-sentence creates grammatically broken chunks
- The "but" contrast is meaningful information
- Faithfulness requires citing what reviewers actually said
- Rating filter (min_rating=4.0) excludes low-rated reviews with mixed sentiment

---

## Edge Cases

### 1. Very Short Reviews (< 50 tokens)
Example: "Works great!" or "Exactly as described"

**Handling:** No chunking. Become single chunks.

**Rationale:** Short reviews are single-topic. Main risk is LLM over-extrapolating from thin evidence, caught by HHEM.

### 2. HTML Artifacts
Example: "Great product!<br /><br />Fast shipping.<br />[[VIDEOID:abc123]]"

**Handling:** `split_sentences()` replaces `<br />` with spaces. Video IDs pass through.

### 3. Mixed Language Content
Example: "Muy bueno! Great product."

**Handling:** Sentence splitter handles basic mixed content. E5-small primarily trained on English, so non-English chunks may have lower retrieval quality.

### 4. Numbers and Specifications
Example: "Battery: 8hrs. Weight: 250g. Price: $49.99"

**Handling:** Kept together as single chunk. Specification lists are valuable evidence.

### 5. Sarcasm and Irony
Example: "Oh yeah, 'great' battery life - lasted 2 whole hours"

**Handling:** Not detected. Dense retrievers encode topic, not sentiment. Rating filter is the defense (sarcastic reviews typically have low ratings).

---

## Implementation Reference

See `sage/core/chunking.py` for implementation.
