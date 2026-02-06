"""
Data loading and embedding pipeline.

Loads Amazon Reviews, chunks them, generates embeddings, and uploads to
Qdrant for vector search.

Features:
- Caches data locally (parquet) and embeddings (.npy)
- Skips Qdrant upload if collection already populated (use --force to override)
- Always applies 5-core filtering (standard for recommendation systems)

Usage:
    python scripts/pipeline.py
    python scripts/pipeline.py --force
    python scripts/pipeline.py --validate-tokenizer
    python scripts/pipeline.py --test-chunking

Run from project root.
"""

import argparse

import numpy as np

from sage.config import (
    DEV_SUBSET_SIZE,
    DATA_DIR,
    get_logger,
    log_banner,
    log_section,
    log_kv,
)
from sage.data import (
    prepare_data,
    get_review_stats,
    create_temporal_splits,
    verify_temporal_boundaries,
)
from sage.core.chunking import chunk_reviews_batch
from sage.adapters.embeddings import get_embedder
from sage.adapters.vector_store import (
    get_client,
    create_collection,
    upload_chunks,
    get_collection_info,
    create_payload_indexes,
    search,
)

logger = get_logger(__name__)


# ============================================================================
# TOKENIZER VALIDATION (--validate-tokenizer)
# ============================================================================


def run_tokenizer_validation():
    """Validate the chars/token ratio assumption used in chunker.py."""
    from transformers import AutoTokenizer

    log_banner(logger, "TOKENIZER VALIDATION")

    df = prepare_data(subset_size=DEV_SUBSET_SIZE, verbose=False)
    sample = df["text"].dropna().sample(500, random_state=42)

    logger.info("Loaded reviews and sampled 500", extra={"total": len(df)})
    logger.info("Loading E5 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")

    ratios = []
    for text in sample:
        if text and text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 0:
                ratios.append(len(text) / len(tokens))

    ratios = np.array(ratios)

    log_section(logger, "Results")
    log_kv(logger, "Mean chars/token", np.mean(ratios))
    log_kv(logger, "Std", np.std(ratios))
    log_kv(logger, "Current assumption", 4.0)

    status = "VALID" if abs(np.mean(ratios) - 4.0) <= 0.5 else "UPDATE NEEDED"
    logger.info("Validation status: %s", status)


# ============================================================================
# CHUNKING QUALITY TEST (--test-chunking)
# ============================================================================


def run_chunking_test():
    """Test chunking quality on long reviews."""
    import pandas as pd
    from sage.core.chunking import (
        chunk_text,
        split_sentences,
        estimate_tokens,
        NO_CHUNK_THRESHOLD,
    )

    log_banner(logger, "CHUNKING QUALITY TEST", width=70)

    df = prepare_data(subset_size=DEV_SUBSET_SIZE, verbose=False)
    embedder = get_embedder()

    df["tokens"] = df["text"].apply(estimate_tokens)
    long_reviews = df[df["tokens"] > NO_CHUNK_THRESHOLD]

    logger.info("Reviews needing chunking: %d", len(long_reviews))

    sample = long_reviews.sample(min(50, len(long_reviews)), random_state=42)
    results = []

    for idx, (_, row) in enumerate(sample.iterrows()):
        text, tokens, rating = row["text"], row["tokens"], int(row["rating"])
        chunks = chunk_text(text, embedder=embedder)
        sentences = split_sentences(text)

        results.append(
            {
                "tokens": tokens,
                "sentences": len(sentences),
                "chunks": len(chunks),
                "avg_chunk_tokens": np.mean([estimate_tokens(c) for c in chunks]),
            }
        )

        if idx < 5:
            logger.info(
                "Review %d [%d*] (%d tok) -> %d chunks",
                idx + 1,
                rating,
                tokens,
                len(chunks),
            )

    results_df = pd.DataFrame(results)
    log_section(logger, f"Summary ({len(results_df)} reviews)")
    logger.info(
        "Chunks per review: %.2f (median: %.0f)",
        results_df["chunks"].mean(),
        results_df["chunks"].median(),
    )
    logger.info("Avg tokens/chunk: %.0f", results_df["avg_chunk_tokens"].mean())

    expansion = (
        results_df["chunks"] * results_df["avg_chunk_tokens"]
    ).sum() / results_df["tokens"].sum()
    logger.info("Expansion ratio: %.2fx", expansion)


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline(subset_size: int, force: bool):
    """Run the full data pipeline: load, chunk, embed, upload."""
    logger.info("Config", extra={"subset_size": subset_size, "force": force})

    # Load and prepare data
    df = prepare_data(subset_size=subset_size, force=force)
    logger.info("Prepared dataset", extra={"reviews": len(df)})

    # Basic statistics
    stats = get_review_stats(df)
    log_section(logger, "Dataset Stats")
    for key, value in stats.items():
        if key == "rating_dist":
            continue
        if key == "sparsity":
            logger.info("%s: %.4f (%.2f%% sparse)", key, value, value * 100)
        elif isinstance(value, float):
            log_kv(logger, key, value)
        else:
            log_kv(logger, key, value)

    # Create temporal splits for recommendation evaluation
    train_df, val_df, test_df = create_temporal_splits(df)
    verify_temporal_boundaries(train_df, val_df, test_df)

    # Review length analysis
    df["text_length"] = df["text"].str.len()
    df["estimated_tokens"] = df["text_length"] // 4

    needs_chunking = (df["estimated_tokens"] > 200).sum()
    logger.info(
        "Reviews needing chunking (>200 tokens): %d (%.1f%%)",
        needs_chunking,
        needs_chunking / len(df) * 100,
    )

    # Prepare reviews for chunking
    reviews_for_chunking = df.to_dict("records")
    for i, review in enumerate(reviews_for_chunking):
        review["review_id"] = f"review_{i}"
        review["product_id"] = review.get("parent_asin", review.get("asin", ""))

    # Load embedder and Qdrant client
    client = get_client()
    embedder = get_embedder()

    # Chunk reviews with semantic chunking
    logger.info("Chunking %d reviews...", len(reviews_for_chunking))
    chunks = chunk_reviews_batch(reviews_for_chunking, embedder=embedder)
    logger.info(
        "Created %d chunks from %d reviews (expansion: %.2fx)",
        len(chunks),
        len(reviews_for_chunking),
        len(chunks) / len(reviews_for_chunking),
    )

    # Generate embeddings
    chunk_texts = [c.text for c in chunks]
    cache_path = DATA_DIR / f"embeddings_{len(chunks)}.npy"

    logger.info("Embedding %d chunks...", len(chunk_texts))
    embeddings = embedder.embed_passages(
        chunk_texts, cache_path=cache_path, force=force
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Embedding technical validation
    log_section(logger, "Embedding Technical Validation")
    logger.info("Shape: %s (expected: (n, 384))", embeddings.shape)
    assert embeddings.shape[1] == 384, f"Wrong dimensions: {embeddings.shape[1]}"

    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    logger.info("NaN values: %d", nan_count)
    logger.info("Inf values: %d", inf_count)
    assert nan_count == 0 and inf_count == 0, "Found NaN or Inf values"

    norms = np.linalg.norm(embeddings, axis=1)
    logger.info("L2 norms: mean=%.4f, std=%.6f", norms.mean(), norms.std())
    assert np.allclose(norms, 1.0, atol=0.01), "Embeddings not normalized"

    logger.info("Value range: [%.3f, %.3f]", embeddings.min(), embeddings.max())
    logger.info("Technical validation: PASSED")

    # Model smoke test (verifies model is loaded correctly)
    # E5 captures topical/domain similarity, NOT sentiment polarity.
    # "terrible battery life" scores HIGHER with "great battery life" (0.93) than
    # "excellent screen resolution" (0.84) because topic > sentiment for retrieval.
    log_section(logger, "Model Smoke Test")
    logger.info("(Verifying E5 captures domain/topical similarity)")

    test_query = "great battery life"
    in_domain_similar = "long lasting charge"
    in_domain_different = "excellent screen quality"
    out_of_domain = "Shakespeare wrote many plays"

    emb_query = embedder.embed_single_query(test_query)
    emb_in_sim = embedder.embed_single_query(in_domain_similar)
    emb_in_diff = embedder.embed_single_query(in_domain_different)
    emb_out = embedder.embed_single_query(out_of_domain)

    sim_in_similar = float(np.dot(emb_query, emb_in_sim))
    sim_in_different = float(np.dot(emb_query, emb_in_diff))
    sim_out = float(np.dot(emb_query, emb_out))

    logger.info("Query: '%s'", test_query)
    logger.info(
        "  In-domain (same topic):  '%s' = %.3f", in_domain_similar, sim_in_similar
    )
    logger.info(
        "  In-domain (diff topic):  '%s' = %.3f", in_domain_different, sim_in_different
    )
    logger.info("  Out-of-domain:           '%s' = %.3f", out_of_domain, sim_out)

    if sim_in_similar > sim_in_different > sim_out:
        logger.info(
            "Ranking correct: %.3f > %.3f > %.3f",
            sim_in_similar,
            sim_in_different,
            sim_out,
        )
    else:
        logger.warning("Unexpected ranking")

    # Known limitation: Dense retrievers don't handle negation well
    # -------------------------------------------------------------------------
    # E5 (and most dense retrievers) encode topical similarity, not sentiment.
    # A query for "good sound quality" will match "The sound quality is NOT good"
    # because the content words overlap - the model largely ignores negation.
    #
    # MITIGATION: Rating filter (min_rating=4+) removes negative reviews before
    # they reach the user. The embedding retrieval finds topically relevant
    # content; the metadata filter enforces quality/sentiment alignment.
    # -------------------------------------------------------------------------

    # Create collection and upload
    create_collection(client)
    upload_chunks(client, chunks, embeddings)
    create_payload_indexes(client)

    # Verify upload
    info = get_collection_info(client)
    log_section(logger, "Collection Info")
    for key, value in info.items():
        log_kv(logger, key, value)

    # Test search
    log_section(logger, "Testing Search")

    test_queries = [
        "great battery life, lasts all day",
        "poor quality, broke after a week",
        "easy to set up and use",
    ]

    for query in test_queries:
        query_embedding = embedder.embed_single_query(query)
        results = search(client, query_embedding.tolist(), limit=3)

        logger.info("Query: '%s'", query)
        for i, r in enumerate(results):
            logger.info("  %d. [%.0f*] %s...", i + 1, r["rating"], r["text"][:80])

    # Test filtered search (demonstrates negation mitigation)
    log_section(logger, "Filtered Search (4+ stars)")
    query = "good sound quality"
    query_embedding = embedder.embed_single_query(query)
    results = search(client, query_embedding.tolist(), limit=5, min_rating=4.0)

    logger.info("Query: '%s' (min 4 stars)", query)
    for i, r in enumerate(results):
        logger.info("  %d. [%.0f*] %s...", i + 1, r["rating"], r["text"][:80])

    client.close()
    log_banner(logger, "PIPELINE COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument(
        "--force", action="store_true", help="Force recreate collection"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEV_SUBSET_SIZE,
        help="Number of reviews to load initially",
    )
    parser.add_argument(
        "--validate-tokenizer",
        action="store_true",
        help="Run tokenizer validation only",
    )
    parser.add_argument(
        "--test-chunking", action="store_true", help="Run chunking quality test only"
    )
    args = parser.parse_args()

    if args.validate_tokenizer:
        run_tokenizer_validation()
    elif args.test_chunking:
        run_chunking_test()
    else:
        run_pipeline(subset_size=args.subset_size, force=args.force)


if __name__ == "__main__":
    main()
