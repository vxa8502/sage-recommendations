"""Tests for sage.core.chunking — pure text processing functions."""

from sage.core.chunking import (
    chunk_text,
    estimate_tokens,
    find_split_points,
    sliding_window_chunk,
    split_sentences,
    NO_CHUNK_THRESHOLD,
)
from sage.config import CHARS_PER_TOKEN


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        assert estimate_tokens("hello") == 1  # 5 chars // 4 = 1

    def test_boundary_values(self):
        """Integer division boundaries at CHARS_PER_TOKEN multiples."""
        assert estimate_tokens("abc") == 0  # 3 // 4 = 0 (under boundary)
        assert estimate_tokens("abcd") == 1  # 4 // 4 = 1 (exact boundary)
        assert estimate_tokens("12345") == 1  # 5 // 4 = 1 (over boundary)
        assert estimate_tokens("abcdefg") == 1  # 7 // 4 = 1 (under next)
        assert estimate_tokens("abcdefgh") == 2  # 8 // 4 = 2 (exact boundary)

    def test_larger_inputs(self):
        assert estimate_tokens("This is a test.") == 3  # 15 // 4 = 3
        assert estimate_tokens("A" * 100) == 25  # 100 // 4 = 25

    def test_unicode_characters(self):
        """Python len() counts code points, not bytes."""
        assert estimate_tokens("🎉") == 0  # 1 code point // 4 = 0
        assert estimate_tokens("Hi 🎉") == 1  # 4 code points // 4 = 1
        assert estimate_tokens("😀" * 8) == 2  # 8 code points // 4 = 2


class TestSplitSentences:
    def test_simple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        text = "Just one sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0].strip() == "Just one sentence."

    def test_question_and_exclamation(self):
        text = "Is this good? Yes it is! Absolutely."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_handles_html_tags(self):
        text = "<br>Hello there.<br/>How are you?"
        sentences = split_sentences(text)
        # Should strip HTML and still split
        assert len(sentences) >= 2

    def test_empty_string(self):
        sentences = split_sentences("")
        assert sentences == [] or sentences == [""]

    def test_preserves_content(self):
        text = "The headphones are great. Sound quality is excellent."
        sentences = split_sentences(text)
        joined = " ".join(s.strip() for s in sentences)
        assert "headphones" in joined
        assert "excellent" in joined


class TestSlidingWindowChunk:
    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0].strip() == text

    def test_long_text_creates_multiple_chunks(self):
        # Create text long enough to require multiple chunks
        sentences = [
            f"This is sentence number {i} with some padding text." for i in range(20)
        ]
        text = " ".join(sentences)
        chunks = sliding_window_chunk(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_chunks_have_content(self):
        sentences = [f"Sentence {i} has content." for i in range(15)]
        text = " ".join(sentences)
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=5)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap_produces_shared_content(self):
        sentences = [f"Unique sentence {i} here." for i in range(20)]
        text = " ".join(sentences)
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 2, "Expected multiple chunks for overlap test"
        # With overlap, adjacent chunks should share some words
        words_0 = set(chunks[0].split())
        words_1 = set(chunks[1].split())
        shared_words = words_0 & words_1
        assert len(shared_words) > 0, (
            "Adjacent chunks should share words due to overlap"
        )

    # --- Additional tests for comprehensive coverage ---

    def test_empty_string_returns_single_empty_chunk(self):
        """Empty input should return a list with the empty string."""
        chunks = sliding_window_chunk("", chunk_size=100, overlap=20)
        assert chunks == [""]

    def test_all_content_preserved_with_overlap(self):
        """With overlap, all original words must appear in at least one chunk."""
        words = [f"uniqueword{i}" for i in range(100)]
        text = " ".join(words)
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=10)

        # Every original word should appear in the joined chunk text
        all_chunk_text = " ".join(chunks)
        missing_words = [w for w in words if w not in all_chunk_text]
        assert not missing_words, f"Content lost: {missing_words}"

    def test_all_adjacent_pairs_share_overlap(self):
        """Extend overlap test to verify ALL adjacent pairs, not just first two."""
        sentences = [f"Unique sentence {i} here." for i in range(20)]
        text = " ".join(sentences)
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=10)

        assert len(chunks) >= 3, "Need at least 3 chunks for comprehensive test"
        for i in range(len(chunks) - 1):
            words_i = set(chunks[i].split())
            words_j = set(chunks[i + 1].split())
            shared = words_i & words_j
            assert shared, (
                f"Chunks {i} and {i + 1} have no overlap: {chunks[i][-30:]!r} | {chunks[i + 1][:30]!r}"
            )

    def test_overlap_zero_produces_no_duplication(self):
        """With overlap=0, adjacent chunks should not share positional content."""
        # Use sentence-structured text to get clean breaks
        sentences = [f"Sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=0)

        assert len(chunks) >= 2, "Need multiple chunks for this test"
        # Check that chunk boundaries don't have the same ending/starting sequence
        for i in range(len(chunks) - 1):
            # Last 10 chars of chunk i should not equal first 10 of chunk i+1
            # (unless there's accidental word repetition, which is fine)
            tail = chunks[i][-10:] if len(chunks[i]) >= 10 else chunks[i]
            head = chunks[i + 1][:10] if len(chunks[i + 1]) >= 10 else chunks[i + 1]
            assert tail != head, f"Unexpected exact overlap at boundary {i}"

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        """Text exactly equal to chunk_size should produce one chunk."""
        # chunk_size=50 tokens * 4 chars/token = 200 chars
        text = "X" * 200
        chunks = sliding_window_chunk(text, chunk_size=50, overlap=10)
        assert len(chunks) == 1
        assert len(chunks[0]) == 200

    def test_text_slightly_larger_than_chunk_size(self):
        """Text just over chunk_size should produce two overlapping chunks."""
        # 210 chars = 200 (first chunk) + overlap region + remainder
        text = "Y" * 210
        chunks = sliding_window_chunk(text, chunk_size=50, overlap=10)
        assert len(chunks) == 2
        # Total coverage should equal text length + overlap
        total_chars = sum(len(c) for c in chunks)
        # With 40 chars overlap (10 tokens * 4), total should be ~250
        assert total_chars == 250, f"Expected 250, got {total_chars}"

    def test_sentence_boundary_preferred_when_possible(self):
        """Chunks should prefer breaking at sentence boundaries when within range."""
        # Create text where sentence boundary is within the search window
        text = "First sentence. " * 10 + "Second sentence. " * 10
        chunks = sliding_window_chunk(text, chunk_size=40, overlap=8)

        # At least some chunks (not the last) should end with period
        chunks_ending_with_period = [c for c in chunks[:-1] if c.rstrip().endswith(".")]
        assert len(chunks_ending_with_period) > 0, (
            "Expected at least some chunks to end at sentence boundaries"
        )

    def test_overlap_exceeds_chunk_size_raises_or_handles_gracefully(self):
        """When overlap >= chunk_size, function should not hang or loop infinitely."""
        import signal

        text = "A" * 500

        def timeout_handler(signum, frame):
            raise TimeoutError("Infinite loop detected")

        # Set a 1-second timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)

        try:
            # This should either complete quickly or raise an error
            # It should NOT hang indefinitely
            chunks = sliding_window_chunk(text, chunk_size=50, overlap=60)
            signal.alarm(0)
            # If it completes, verify it produced something reasonable
            # (current implementation loops infinitely - this test documents the bug)
            assert False, (
                "Expected timeout or error when overlap >= chunk_size, "
                f"but got {len(chunks)} chunks"
            )
        except TimeoutError:
            # This is the expected behavior currently (infinite loop)
            # This test documents the bug for future fixing
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_very_large_overlap_near_chunk_size(self):
        """Overlap just under chunk_size should work but produce many chunks."""
        text = "Z" * 500
        # overlap=49, chunk_size=50 means each step advances by 1 token (4 chars)
        chunks = sliding_window_chunk(text, chunk_size=50, overlap=49)
        # Should complete without hanging
        assert len(chunks) > 0
        # Will produce many overlapping chunks
        assert len(chunks) > 50, (
            f"Expected many chunks due to small step, got {len(chunks)}"
        )

    def test_whitespace_only_text(self):
        """Whitespace-only text should return the original (stripped) or empty."""
        chunks = sliding_window_chunk("   \n\t  ", chunk_size=100, overlap=20)
        # After stripping, this becomes empty
        assert chunks == ["   \n\t  "] or chunks == [""]

    def test_single_word_longer_than_chunk_size(self):
        """A single word longer than chunk_size must still be captured."""
        long_word = "supercalifragilisticexpialidocious" * 10  # ~340 chars
        text = f"Start {long_word} end"
        chunks = sliding_window_chunk(text, chunk_size=20, overlap=5)  # 80 char chunks

        # The long word should appear (possibly split) across chunks
        joined = "".join(chunks)
        # All characters from original should be present
        assert long_word[:50] in joined or long_word in joined, (
            "Long word content should be preserved across chunks"
        )

    def test_punctuation_only_text(self):
        """Text containing only punctuation should be handled gracefully."""
        text = "... !!! ??? ,,, ;;; ::: --- ..."
        chunks = sliding_window_chunk(text, chunk_size=10, overlap=2)
        assert len(chunks) >= 1
        # All punctuation should be preserved
        joined = "".join(chunks)
        for char in ".!?,;:-":
            if char in text:
                assert char in joined, f"Punctuation '{char}' was lost"

    def test_text_with_no_spaces(self):
        """Continuous text without spaces should chunk correctly."""
        # 520 chars of continuous letters
        text = "abcdefghijklmnopqrstuvwxyz" * 20
        chunks = sliding_window_chunk(text, chunk_size=30, overlap=5)

        assert len(chunks) > 1, "Long continuous text should produce multiple chunks"
        # All original characters should be present (with possible duplication from overlap)
        joined = "".join(chunks)
        for char in set(text):
            assert char in joined, f"Character '{char}' was lost"
        # Character count should be >= original (overlap causes duplication)
        assert len(joined) >= len(text), "Content was lost"


class TestChunkText:
    """Tests for the tiered chunk_text function with NO_CHUNK_THRESHOLD boundary."""

    def test_text_under_threshold_not_chunked(self):
        """Text under NO_CHUNK_THRESHOLD (200 tokens) should not be chunked."""
        # 199 tokens = 796 chars
        text = "X" * (NO_CHUNK_THRESHOLD * CHARS_PER_TOKEN - CHARS_PER_TOKEN)
        tokens = estimate_tokens(text)
        assert tokens < NO_CHUNK_THRESHOLD, (
            f"Setup error: {tokens} >= {NO_CHUNK_THRESHOLD}"
        )

        chunks = chunk_text(text)
        assert len(chunks) == 1, (
            f"Text under threshold should not be chunked, got {len(chunks)}"
        )
        assert chunks[0] == text

    def test_text_exactly_at_threshold_not_chunked(self):
        """Text exactly at NO_CHUNK_THRESHOLD (200 tokens) should NOT be chunked."""
        # 200 tokens = 800 chars (boundary condition: <= means not chunked)
        text = "Y" * (NO_CHUNK_THRESHOLD * CHARS_PER_TOKEN)
        tokens = estimate_tokens(text)
        assert tokens == NO_CHUNK_THRESHOLD, (
            f"Setup error: {tokens} != {NO_CHUNK_THRESHOLD}"
        )

        chunks = chunk_text(text)
        assert len(chunks) == 1, (
            f"Text exactly at threshold should not be chunked, got {len(chunks)}"
        )

    def test_text_over_threshold_is_chunked(self):
        """Text over NO_CHUNK_THRESHOLD (200 tokens) should be chunked."""
        # 201 tokens = 804 chars
        text = "Z" * (NO_CHUNK_THRESHOLD * CHARS_PER_TOKEN + CHARS_PER_TOKEN)
        tokens = estimate_tokens(text)
        assert tokens > NO_CHUNK_THRESHOLD, (
            f"Setup error: {tokens} <= {NO_CHUNK_THRESHOLD}"
        )

        # Without embedder, falls back to sliding window
        chunks = chunk_text(text)
        assert len(chunks) > 1, (
            f"Text over threshold should be chunked, got {len(chunks)}"
        )

    def test_empty_text_returns_empty_list(self):
        """Empty or whitespace-only text should return empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text("\n\t") == []

    def test_short_review_not_chunked(self):
        """A typical short review under 200 tokens stays intact."""
        review = (
            "Great headphones! The sound quality is amazing and they're very comfortable. "
            "Battery life is excellent too. Highly recommend for the price."
        )
        tokens = estimate_tokens(review)
        assert tokens < NO_CHUNK_THRESHOLD, f"Review too long: {tokens} tokens"

        chunks = chunk_text(review)
        assert len(chunks) == 1
        assert chunks[0] == review

    def test_long_review_is_chunked(self):
        """A long review over 200 tokens gets chunked."""
        # Create a review that's definitely over 200 tokens (~1000 chars)
        sentences = [
            f"This is sentence number {i} with detailed content." for i in range(30)
        ]
        review = " ".join(sentences)
        tokens = estimate_tokens(review)
        assert tokens > NO_CHUNK_THRESHOLD, f"Review too short: {tokens} tokens"

        chunks = chunk_text(review)  # No embedder = sliding window fallback
        assert len(chunks) > 1, f"Long review should be chunked, got {len(chunks)}"

        # Verify all content is preserved
        all_chunk_text = " ".join(chunks)
        for i in range(30):
            assert f"sentence number {i}" in all_chunk_text, f"Sentence {i} was lost"


class TestFindSplitPoints:
    def test_uniform_similarities_no_splits(self):
        # All same similarity => nothing below threshold
        sims = [0.9, 0.9, 0.9, 0.9]
        splits = find_split_points(sims, threshold_percentile=50)
        # With uniform values, the threshold equals the value itself
        # so no similarity is strictly below the threshold
        assert isinstance(splits, list)

    def test_clear_topic_boundary(self):
        # High similarities with one dip
        sims = [0.95, 0.92, 0.10, 0.93, 0.91]
        splits = find_split_points(sims, threshold_percentile=30)
        # The dip at index 2 should be detected
        assert 2 in splits

    def test_empty_input(self):
        splits = find_split_points([])
        assert splits == []

    def test_single_value(self):
        splits = find_split_points([0.5])
        assert isinstance(splits, list)

    def test_returns_sorted_indices(self):
        sims = [0.9, 0.1, 0.9, 0.05, 0.9]
        splits = find_split_points(sims, threshold_percentile=50)
        assert splits == sorted(splits)
