# Data Card: Sage Electronics Reviews

This document provides context, methodology, and limitations for the dataset
powering Sage's explainable recommendations. For machine-readable statistics,
see `data/eda_stats_latest.json`.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | Amazon Reviews 2023 (McAuley Lab, published 2023) |
| Category | Electronics |
| Collection | `sage_reviews` (Qdrant Cloud) |
| Total Chunks | 423,165 |
| Unique Reviews | 334,282 |
| Unique Products | 21,827 |

**Purpose:** Ground LLM explanations in real customer experiences. Each chunk
serves as retrievable evidence for RAG-based recommendation explanations.

---

## Collection Process

### Source Data

The [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset from
McAuley Lab provides timestamped reviews with ratings, product IDs, and user IDs.
We use the Electronics category for domain focus.

### 5-Core Filtering

We apply 5-core filtering: only users and items with at least 5 interactions
are retained. This ensures:

- **Data density:** Products have enough reviews to generate meaningful explanations
- **User diversity:** Multiple perspectives per product reduce single-reviewer bias
- **Standard practice:** Aligns with collaborative filtering literature conventions

The filtering reduces raw data by ~70% but dramatically improves evidence quality
for the recommendation task.

### Chunking Strategy

Reviews vary from single sentences to multi-paragraph essays. Our chunking
strategy balances context preservation with retrieval granularity:

| Review Length | Strategy | Rationale |
|---------------|----------|-----------|
| <200 tokens | Keep whole | Short reviews lose meaning when split |
| 200-500 tokens | Semantic split | Natural boundaries (sentences, paragraphs) |
| >500 tokens | Sliding window | 150 tokens with 30-token overlap |

This produces an expansion ratio of ~1.27x (423K chunks from 334K reviews),
meaning most reviews remain intact while long reviews contribute multiple
overlapping chunks.

---

## Data Fields

Each chunk in Qdrant contains:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk content (review text or segment) |
| `product_id` | string | Amazon ASIN |
| `review_id` | string | Unique review identifier |
| `rating` | int | 1-5 star rating |
| `timestamp` | int | Unix timestamp (ms) |
| `chunk_index` | int | Position within review (0-indexed) |
| `total_chunks` | int | Total chunks from this review |

Chunk IDs are deterministic MD5 hashes of (review_id, chunk_index) for
reproducible indexing.

---

## Distribution Analysis

### Rating Distribution

The dataset exhibits typical e-commerce positive skew (review-level counts):

| Rating | Reviews | Percentage |
|--------|---------|------------|
| 5-star | 223,251 | 66.8% |
| 4-star | 49,924 | 14.9% |
| 3-star | 23,323 | 7.0% |
| 2-star | 14,461 | 4.3% |
| 1-star | 23,323 | 7.0% |

**Implication:** Negative sentiment is underrepresented. Sage's quality gate
may refuse queries about product problems when insufficient negative reviews
exist, which is the correct behavior (refuse rather than hallucinate).

### Chunk Length

- Median: 169 characters (42 tokens)
- Mean: 258 characters

The median being lower than mean indicates right-skew from long reviews.
Most chunks are concise single-sentence opinions; a minority are detailed
multi-paragraph analyses.

### Temporal Coverage

Reviews span **May 2000 to March 2023** (23 years). The dataset was published
in 2023 but contains historical reviews. Note: timestamps from 2000–2005 may
include data artifacts from Amazon's early review system.

Temporal distribution affects relevance for queries about "current" product
quality (a known limitation).

---

## Known Limitations

### Category Scope

Electronics-only. The system will not provide useful recommendations for
other product categories. Cross-category queries trigger the quality gate.

### Language

English-only. Non-English queries may retrieve irrelevant evidence.

### Recency Bias

Product quality changes over time (firmware updates, manufacturing changes).
Reviews end in March 2023, so products released after that date have no coverage.
Older reviews (2000–2015) may describe discontinued or significantly updated products.

### Positive Sentiment Skew

The 5-core filter amplifies positive bias (engaged users rate higher).
Queries seeking negative sentiment may have sparse evidence.

### Review Authenticity

No verification of review authenticity. Amazon's verified purchase flag
is not currently used for filtering.

---

## Modeling Implications

EDA findings directly influenced architecture decisions:

1. **5-core requirement:** Ensures MIN_EVIDENCE_CHUNKS (2) is achievable for
   most products. Without filtering, many products have 1-2 reviews total.

2. **Chunk size tuning:** 150-token chunks with 30-token overlap balances
   retrieval precision (small chunks match specific queries) with context
   (overlap preserves sentence boundaries).

3. **Quality gate thresholds:** MIN_EVIDENCE_TOKENS (50) and MIN_RETRIEVAL_SCORE
   (0.7) were calibrated against EDA distributions to refuse ~40-50% of
   low-evidence queries rather than hallucinate.

4. **Rating distribution awareness:** The prompt instructs the LLM to
   acknowledge rating skew when summarizing sentiment.

---

## Reproducibility

To regenerate statistics:

```bash
make eda
```

This produces:
- `data/eda_stats_latest.json` - Machine-readable statistics
- `assets/*.png` - Distribution visualizations

This DATA_CARD.md is manually maintained and not overwritten by `make eda`.

---

## Citation

```bibtex
@article{hou2024amazon,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
