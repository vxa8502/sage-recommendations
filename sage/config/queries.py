"""
Standard evaluation queries.

Separated from main config to keep configuration declarative.
These are test fixtures used by evaluation scripts.

Query organization:
- CORE_QUERIES: Common queries appearing in all evaluations
- STANDARD_QUERIES: Standard product category queries
- EDGE_CASE_QUERIES: Challenging queries for failure analysis
- Derived lists compose these bases for specific use cases
"""

# Core queries - used across all evaluations (5 queries)
CORE_QUERIES = [
    "wireless headphones with noise cancellation",
    "laptop charger for MacBook",
    "USB hub with multiple ports",
    "portable battery pack for travel",
    "bluetooth speaker with good bass",
]

# Standard product queries - common categories (13 queries)
STANDARD_QUERIES = [
    "HDMI cable for 4K TV",
    "external hard drive for backup",
    "webcam for video calls",
    "wireless mouse for laptop",
    "keyboard with backlight",
    "screen protector for phone",
    "phone case with good protection",
    "earbuds for working out",
    "tablet stand for desk",
    "laptop cooling pad",
    "surge protector with USB ports",
    "wireless charging pad",
    "fast charging USB-C cable",
]

# Edge case queries - tests failure modes (9 queries)
EDGE_CASE_QUERIES = [
    "cheap but good quality earbuds",
    "durable phone case that looks nice",
    "fast charging cable that won't break",
    "comfortable headphones for long sessions",
    "quiet keyboard for office",
    "headphones that don't hurt ears",
    "charger that actually works",
    "waterproof speaker for shower",
    "gift for someone who likes music",
]

# Primary evaluation queries - used for general RAGAS/HHEM evaluation
# Combines core + standard + 2 semantic variants
EVALUATION_QUERIES = (
    CORE_QUERIES
    + STANDARD_QUERIES
    + [
        "noise cancelling headphones for travel",
        "portable speaker with good bass",
    ]
)

# Queries for failure analysis - focused on edge cases and challenging queries
ANALYSIS_QUERIES = CORE_QUERIES + EDGE_CASE_QUERIES
