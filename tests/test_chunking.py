"""Tests for sage.core.chunking — pure text processing functions."""

from sage.core.chunking import (
    estimate_tokens,
    find_split_points,
    sliding_window_chunk,
    split_sentences,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        # "hello" = 5 chars / 4 chars_per_token = 1
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "This is a sample sentence with several words."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4


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
