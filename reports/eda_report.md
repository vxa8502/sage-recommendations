# Exploratory Data Analysis: Production Data

**Source:** Qdrant Cloud (Collection: `sage_reviews`)
**Status:** green
**Generated from live production data**

---

## Dataset Overview

This report analyzes the actual data deployed in production, ensuring all statistics match what the recommendation system uses.

| Metric | Value |
|--------|-------|
| Total Chunks | 423,165 |
| Unique Reviews | 334,282 |
| Unique Products | 21,827 |
| Expansion Ratio | 1.27x |

---

## Rating Distribution

Amazon reviews exhibit a characteristic J-shaped distribution, heavily skewed toward 5-star ratings.

![Rating Distribution](../assets/rating_distribution.png)

| Rating | Count | Percentage |
|--------|-------|------------|
| 1 | 31,924 | 7.5% |
| 2 | 21,301 | 5.0% |
| 3 | 34,078 | 8.1% |
| 4 | 71,153 | 16.8% |
| 5 | 264,709 | 62.6% |

**Key Observations:**
- 5-star ratings: 62.6% of chunks
- 1-star ratings: 7.5% of chunks
- This polarization is typical for e-commerce review data

---

## Chunk Length Analysis

Chunk lengths affect retrieval quality and context window usage.

![Chunk Lengths](../assets/chunk_lengths.png)

**Statistics:**
- Median chunk length: 169 characters (~42 tokens)
- Mean chunk length: 258 characters
- Most chunks fit comfortably within embedding model context

---

## Chunking Distribution

Reviews are chunked based on length: short reviews stay whole, longer reviews are split semantically.

![Chunks per Review](../assets/chunks_per_review.png)

| Metric | Value |
|--------|-------|
| Single-chunk reviews | 303,550 |
| Multi-chunk reviews | 30,732 |
| Expansion ratio | 1.27x |

**Chunking Strategy:**
- Reviews < 200 tokens: No chunking (embedded whole)
- Reviews 200-500 tokens: Semantic chunking
- Reviews > 500 tokens: Semantic + sliding window

---

## Temporal Distribution

Review timestamps enable chronological analysis and temporal evaluation splits.

![Temporal Distribution](../assets/temporal_distribution.png)

---

## Data Quality

The production dataset has been through 5-core filtering (users and items with 5+ interactions) and quality checks:

- All chunks have valid text content
- All ratings are in [1, 5] range
- All product identifiers present
- Deterministic chunk IDs (MD5 hash of review_id + chunk_index)

---

## Summary

This production EDA confirms the deployed data characteristics:

1. **Scale:** 423,165 chunks across 21,827 products
2. **Quality:** 5-core filtered, validated payloads
3. **Distribution:** J-shaped ratings, typical e-commerce pattern
4. **Chunking:** 1.27x expansion from reviews to chunks

The data matches what the recommendation API queries in real-time.

---

*Report generated from Qdrant Cloud. Run `make eda` to regenerate.*
