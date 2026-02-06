"""
Standard evaluation queries.

Separated from main config to keep configuration declarative.
These are test fixtures used by evaluation scripts.
"""

# Primary evaluation queries - used for general RAGAS/HHEM evaluation
EVALUATION_QUERIES = [
    # Common product categories (high confidence expected)
    "wireless headphones with noise cancellation",
    "laptop charger compatible with MacBook",
    "USB hub with multiple ports",
    "portable phone charger for travel",
    "bluetooth speaker with good bass",
    "HDMI cable for 4K TV",
    "external hard drive for backup",
    "webcam for video calls",
    "wireless mouse for laptop",
    "keyboard with backlight",
    # Specific attribute queries (medium confidence)
    "screen protector for phone",
    "phone case with good protection",
    "earbuds for working out",
    "tablet stand for desk",
    "laptop cooling pad",
    "surge protector with USB ports",
    "wireless charging pad",
    "fast charging USB-C cable",
    "noise cancelling headphones for travel",
    "portable speaker with good bass",
]

# Queries for failure analysis - focused on edge cases and challenging queries
ANALYSIS_QUERIES = [
    "wireless headphones with noise cancellation",
    "laptop charger for MacBook",
    "USB hub with multiple ports",
    "portable battery pack for travel",
    "bluetooth speaker with good bass",
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

# Queries for end-to-end success rate evaluation - comprehensive coverage
E2E_EVAL_QUERIES = [
    "wireless headphones with noise cancellation",
    "laptop charger for MacBook",
    "USB hub with multiple ports",
    "portable battery pack for travel",
    "bluetooth speaker with good bass",
    "cheap but good quality earbuds",
    "durable phone case that looks nice",
    "fast charging cable that won't break",
    "comfortable headphones for long sessions",
    "quiet keyboard for office",
    "headphones that don't hurt ears",
    "charger that actually works",
    "waterproof speaker for shower",
    "gift for someone who likes music",
    "tablet stand for kitchen",
    "wireless mouse for laptop",
    "HDMI cable for monitor",
    "phone mount for car",
    "screen protector for phone",
    "backup battery for camping",
]
