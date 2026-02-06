# Exploratory Data Analysis: Amazon Electronics Reviews

**Dataset:** McAuley-Lab/Amazon-Reviews-2023 (Electronics category)
**Subset:** 100,000 raw reviews → 2,635 after 5-core filtering

---

## Dataset Overview

The Amazon Electronics reviews dataset provides rich user feedback data for building recommendation systems. After standard preprocessing and 5-core filtering (requiring users and items to have at least 5 interactions), the dataset exhibits the characteristic sparsity of real-world recommendation scenarios.

| Metric | Raw | After 5-Core |
|--------|-----|--------------|
| Total Reviews | 100,000 | 2,635 |
| Unique Users | 15,322 | 334 |
| Unique Items | 59,429 | 318 |
| Avg Rating | 4.26 | 4.44 |
| Retention | — | 2.6% |

---

## Rating Distribution

Amazon reviews exhibit a well-known J-shaped distribution, heavily skewed toward 5-star ratings. This reflects both genuine satisfaction and selection bias (dissatisfied customers often don't leave reviews).

![Rating Distribution](../data/figures/rating_distribution.png)

**Key Observations:**
- 5-star ratings dominate (65.4% of reviews)
- 1-star reviews form the second largest group (8.0%)
- Middle ratings (2-4 stars) are relatively rare (26.6% combined)
- This polarization is typical for e-commerce review data

**Implications for Modeling:**
- Binary classification (positive/negative) may be more robust than regression
- Rating-weighted aggregation should account for the skewed distribution
- Evidence from 4-5 star reviews carries stronger positive signal

---

## Review Length Analysis

Review length varies significantly and correlates with the chunking strategy for the RAG pipeline. Most reviews are short enough to embed directly without chunking.

![Review Length Distribution](../data/figures/review_lengths.png)

**Length Statistics:**
- Median: 183 characters (~45 tokens)
- Mean: 369 characters (~92 tokens)
- Reviews exceeding 200 tokens: 11.2% (require chunking)

**Chunking Strategy Validation:**
The tiered chunking approach is well-suited to this distribution:
- **Short (<200 tokens):** No chunking needed — majority of reviews
- **Medium (200-500 tokens):** Semantic chunking at topic boundaries
- **Long (>500 tokens):** Semantic + sliding window fallback

---

## Review Length by Rating

Negative reviews tend to be longer than positive ones. Users who are dissatisfied often provide detailed explanations of issues, while satisfied users may simply express approval.

![Review Length by Rating](../data/figures/length_by_rating.png)

**Pattern:**
- 1-star reviews: 187 chars median
- 2-3 star reviews: 258-265 chars median (users explain nuance)
- 4-star reviews: 297 chars median (longest — detailed positive feedback)
- 5-star reviews: 152 chars median (shortest — quick endorsements)

**Implications:**
- Negative reviews provide richer evidence for issue identification
- Positive reviews may require multiple chunks for substantive explanations
- Rating filters (min_rating=4) naturally bias toward shorter evidence

---

## Temporal Distribution

The dataset spans multiple years of reviews, enabling proper temporal train/validation/test splits that prevent data leakage.

![Reviews Over Time](../data/figures/reviews_over_time.png)

**Temporal Split Strategy:**
- **Train (70%):** Oldest reviews — model learns from historical patterns
- **Validation (10%):** Middle period — hyperparameter tuning
- **Test (20%):** Most recent — simulates production deployment

This chronological ordering ensures the model never sees "future" data during training.

---

## User and Item Activity

The long-tail distribution is pronounced: most users write few reviews, and most items receive few reviews. This sparsity is the fundamental challenge recommendation systems address.

![User and Item Distribution](../data/figures/user_item_distribution.png)

**User Activity:**
- Users with only 1 review: 30.1%
- Users with 5+ reviews: 4,991 (32.6%)
- Power user max: 820 reviews

**Item Popularity:**
- Items with only 1 review: 76.0%
- Items with 5+ reviews: 2,434 (4.1%)
- Most reviewed item: 326 reviews

**Cold-Start Implications:**
- Many items have sparse evidence — content-based features are critical
- User cold-start is common — onboarding preferences help
- 5-core filtering ensures minimum evidence density for evaluation

---

## Data Quality Assessment

The raw dataset contains several quality issues addressed during preprocessing.

| Issue | Count | Resolution |
|-------|-------|------------|
| Missing text | 0 | — |
| Empty reviews | 21 | Removed |
| Very short (<10 chars) | 2,512 | Removed |
| Duplicate texts | 5,219 | Kept (valid re-purchases) |
| Invalid ratings | 0 | — |

**Post-Cleaning:**
- All reviews have valid text content
- All ratings are in [1, 5] range
- All user/product identifiers present

---

## Summary

The Amazon Electronics dataset, after 5-core filtering and cleaning, provides a solid foundation for building and evaluating a RAG-based recommendation system:

1. **Scale:** 2,635 reviews across 334 users and 318 items
2. **Sparsity:** 97.5% — realistic for recommendation evaluation
3. **Quality:** Clean text, valid ratings, proper identifiers
4. **Temporal:** Supports chronological train/val/test splits
5. **Content:** Review lengths suit the tiered chunking strategy

The J-shaped rating distribution and long-tail user/item activity are characteristic of real e-commerce data, making this an appropriate benchmark for portfolio demonstration.

---

*Figures generated by `scripts/eda.py` at 300 DPI. Run `make figures` to regenerate.*
