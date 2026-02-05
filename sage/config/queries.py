"""
Standard evaluation queries.

Separated from main config to keep configuration declarative.
These are test fixtures used by evaluation scripts.
"""

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
