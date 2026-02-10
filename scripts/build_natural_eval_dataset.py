"""
Build natural language evaluation dataset for semantic retrieval.

Creates realistic user queries with manually curated relevance judgments.
Unlike keyword-based evaluation, these queries reflect how users actually
search for products.

Query types:
- Feature-specific: "wireless earbuds with noise cancellation"
- Use-case based: "tablet for kids"
- Budget-conscious: "cheap but good headphones"
- Gift queries: "gift for someone who likes music"
- Problem-solving: "headphones that don't hurt ears"

Run from project root:
    python scripts/build_natural_eval_dataset.py
"""

import json
from collections import Counter

from sage.config import DATA_DIR, get_logger, log_banner, log_section
from sage.core import EvalCase

logger = get_logger(__name__)

EVAL_DIR = DATA_DIR / "eval"


# Natural language queries with relevance judgments
# Relevance: 3.0 = highly relevant, 2.0 = moderately relevant, 1.0 = marginally relevant
NATURAL_QUERIES = [
    # === ECHO / SMART SPEAKER QUERIES ===
    {
        "query": "smart speaker with alexa for my kitchen",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo (111 chunks, highly reviewed)
            "B07H65KP63": 3.0,  # Echo Dot 3rd gen
            "B07KTYJ769": 2.0,  # Echo device
        },
        "category": "echo_devices",
        "intent": "feature_specific",
    },
    {
        "query": "alexa device for playing music",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo with good speaker
            "B010BWYDYA": 2.0,  # Echo/Fire device
            "B0791TX5P5": 2.0,  # Echo with remote
        },
        "category": "echo_devices",
        "intent": "use_case",
    },
    {
        "query": "small alexa for bedroom nightstand",
        "relevant_items": {
            "B07H65KP63": 3.0,  # Echo Dot (small)
            "B07KTYJ769": 2.0,  # Smart plug with echo
        },
        "category": "echo_devices",
        "intent": "use_case",
    },
    {
        "query": "voice assistant to control smart home",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo
            "B07H65KP63": 3.0,  # Echo Dot
            "B07VXXBTX4": 2.0,  # Echo Show
        },
        "category": "echo_devices",
        "intent": "feature_specific",
    },
    # === FIRE TABLET QUERIES ===
    {
        "query": "tablet for reading books and light browsing",
        "relevant_items": {
            "B010BWYDYA": 3.0,  # Fire tablet for reading
            "B01MTF2Z37": 3.0,  # Fire 10
            "B0051VVOB2": 2.0,  # Fire tablet with apps
        },
        "category": "fire_tablets",
        "intent": "use_case",
    },
    {
        "query": "cheap tablet for kids",
        "relevant_items": {
            "B01MTF2Z37": 3.0,  # Fire tablet (affordable)
            "B0051VVOB2": 2.0,  # Fire tablet
            "B011BRUOMO": 2.0,  # Fire tablet
        },
        "category": "fire_tablets",
        "intent": "budget",
    },
    {
        "query": "amazon tablet with good screen",
        "relevant_items": {
            "B01MTF2Z37": 3.0,  # Fire 10 better screen
            "B0BDSZ2KT3": 2.0,  # Fire tablet
        },
        "category": "fire_tablets",
        "intent": "feature_specific",
    },
    {
        "query": "tablet for streaming movies",
        "relevant_items": {
            "B01MTF2Z37": 3.0,  # Fire 10
            "B010BWYDYA": 2.0,  # Fire tablet
            "B0051VVOB2": 2.0,  # Fire tablet
        },
        "category": "fire_tablets",
        "intent": "use_case",
    },
    # === FIRE TV / STREAMING QUERIES ===
    {
        "query": "streaming device for my tv",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick 4K
            "B00CX5P8FC": 2.0,  # Fire TV
            "B0C3NMZ2C7": 3.0,  # Fire TV newer
        },
        "category": "fire_tv",
        "intent": "general",
    },
    {
        "query": "fire stick with 4k support",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick 4K
            "B09B36FJVT": 3.0,  # Fire TV 4K
        },
        "category": "fire_tv",
        "intent": "feature_specific",
    },
    {
        "query": "device to watch netflix and prime video",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick
            "B0C3NMZ2C7": 3.0,  # Fire TV
            "B00CX5P8FC": 2.0,  # Fire TV
        },
        "category": "fire_tv",
        "intent": "use_case",
    },
    {
        "query": "cord cutting streaming solution",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick
            "B01LXJA5JD": 2.0,  # Streaming related
        },
        "category": "fire_tv",
        "intent": "use_case",
    },
    # === SMART HOME QUERIES ===
    {
        "query": "smart plug to control lights with alexa",
        "relevant_items": {
            "B07KTYJ769": 3.0,  # Smart plug works with Echo
            "B0BN74ZJDK": 3.0,  # Smart plug
            "B07W36WN5X": 2.0,  # Smart light
        },
        "category": "smart_home",
        "intent": "feature_specific",
    },
    {
        "query": "smart light bulb for bedroom",
        "relevant_items": {
            "B07W36WN5X": 3.0,  # Smart light
            "B0935ZDCYD": 2.0,  # Smart home device
        },
        "category": "smart_home",
        "intent": "use_case",
    },
    {
        "query": "home security camera",
        "relevant_items": {
            "B08739SDH3": 2.0,  # Camera/security (low rating though)
            "B07VXXBTX4": 2.0,  # Ring doorbell mentioned
        },
        "category": "smart_home",
        "intent": "general",
    },
    {
        "query": "easy to set up smart home device",
        "relevant_items": {
            "B07KTYJ769": 3.0,  # Smart plug easy setup
            "B0BN74ZJDK": 2.0,  # Smart plug
            "B07W36WN5X": 2.0,  # Smart light
        },
        "category": "smart_home",
        "intent": "feature_specific",
    },
    # === STORAGE QUERIES ===
    # NOTE: Products verified via review text (Session 78 ground truth audit)
    {
        "query": "sd card for camera",
        "relevant_items": {
            "B071R715MZ": 3.0,  # SD card (verified: "tablet loaded with topo maps")
            "B00PFQPX5Q": 3.0,  # SD card (verified: "raspberry pi")
            "B01H3LM75O": 2.0,  # SD card (verified: "record 4k video")
        },
        "category": "storage",
        "intent": "use_case",
    },
    {
        "query": "external hard drive for backup",
        "relevant_items": {
            "B008J0Z9TA": 3.0,  # Hard drive (verified: "Fast reliable")
            "B09Q7YPZPJ": 2.0,  # Storage device
            "B07P9V8GSH": 2.0,  # Storage (verified: "Raspberry Pi")
        },
        "category": "storage",
        "intent": "use_case",
    },
    {
        "query": "fast micro sd card for phone",
        "relevant_items": {
            "B071R715MZ": 3.0,  # MicroSD (verified: tablet storage)
            "B009RWAYTE": 3.0,  # MicroSD (verified: "Galaxy phone")
            "B00IVPU786": 2.0,  # MicroSD (verified: "Works as expected")
        },
        "category": "storage",
        "intent": "feature_specific",
    },
    {
        "query": "reliable storage for important files",
        "relevant_items": {
            "B008J0Z9TA": 3.0,  # Hard drive (verified: "Fast reliable")
            "B071R715MZ": 2.0,  # SD card (verified)
            "B01H3LM75O": 2.0,  # SD card (verified: "4k video")
        },
        "category": "storage",
        "intent": "feature_specific",
    },
    # === HEADPHONES / AUDIO QUERIES ===
    {
        "query": "wireless headphones for working out",
        "relevant_items": {
            "B07S764D9V": 3.0,  # Durable affordable headphones
            "B0778PCW73": 3.0,  # Wireless headphones for yard work
            "B00PKTU83U": 2.0,  # Headphones
        },
        "category": "headphones_audio",
        "intent": "use_case",
    },
    {
        "query": "comfortable headphones for long listening",
        "relevant_items": {
            "B00PKTU83U": 3.0,  # Comfortable headphones
            "B07S764D9V": 2.0,  # Headphones
            "B0778PCW73": 2.0,  # Wireless headphones
        },
        "category": "headphones_audio",
        "intent": "feature_specific",
    },
    {
        "query": "budget bluetooth headphones",
        "relevant_items": {
            "B07S764D9V": 3.0,  # Affordable headphones
            "B07G7VT8KQ": 2.0,  # Bluetooth audio
            "B004OVECU0": 2.0,  # Headphones
        },
        "category": "headphones_audio",
        "intent": "budget",
    },
    {
        "query": "headphones with good sound quality",
        "relevant_items": {
            "B07S764D9V": 3.0,  # Good reviews on sound
            "B0778PCW73": 2.0,  # Wireless headphones
            "B00PKTU83U": 2.0,  # Headphones
        },
        "category": "headphones_audio",
        "intent": "feature_specific",
    },
    {
        "query": "earbuds for phone calls",
        "relevant_items": {
            "B07S764D9V": 2.0,  # Headphones
            "B07G7VT8KQ": 2.0,  # Audio device
        },
        "category": "headphones_audio",
        "intent": "use_case",
    },
    # === CABLES / ADAPTERS QUERIES ===
    # NOTE: Products verified via review text (Session 78 ground truth audit)
    {
        "query": "usb c charging cable",
        "relevant_items": {
            "B07QDNWC7X": 3.0,  # USB-C cable (verified: "heavy-duty USB-C cable")
            "B08DK8BP1J": 2.0,  # USB-C cable (verified: "USB2 cables")
            "B0BGS23YKX": 2.0,  # USB-C cable (verified: "braided cords")
        },
        "category": "cables_adapters",
        "intent": "general",
    },
    {
        "query": "hdmi cable for tv",
        "relevant_items": {
            "B085YG25ZS": 3.0,  # HDMI cable (verified: "8K HDMI Cable")
            "B075KP9XHS": 2.0,  # HDMI cable (verified: "quality of the cable")
            "B06XGHMDJ1": 2.0,  # HDMI cable (verified: "picture quality on TV")
        },
        "category": "cables_adapters",
        "intent": "use_case",
    },
    {
        "query": "fast phone charger",
        "relevant_items": {
            "B0C3KV52B7": 3.0,  # Charger (verified: "Charges quickly")
            "B01LWQ933G": 2.0,  # Charger (verified: "many devices plugged-in")
            "B0B96VM41C": 2.0,  # Charger (verified: "power cable for iPad")
        },
        "category": "cables_adapters",
        "intent": "feature_specific",
    },
    {
        "query": "durable charging cable that lasts",
        "relevant_items": {
            "B00SUX3XQC": 3.0,  # Cable (verified: "Nicely made cable")
            "B0C1FRBK4K": 2.0,  # Cable (verified: "Anker brand")
            "B09CTFY414": 2.0,  # Cable (verified: "very nice" charging cable)
        },
        "category": "cables_adapters",
        "intent": "feature_specific",
    },
    # === KEYBOARD / MOUSE QUERIES ===
    # NOTE: Products verified via review text (Session 78 ground truth audit)
    {
        "query": "wireless keyboard for computer",
        "relevant_items": {
            "B001IOHAVO": 3.0,  # Keyboard (verified: keyboard reviews)
            "B003VAGXZC": 2.0,  # Keyboard (verified: "keyboard combo")
            "B00QRPPCMI": 2.0,  # Keyboard (verified: "keyboard and mouse")
        },
        "category": "keyboards_mice",
        "intent": "general",
    },
    {
        "query": "quiet keyboard for office",
        "relevant_items": {
            "B0BK3LYMR2": 3.0,  # Keyboard (verified: "super quiet")
            "B004MF11MU": 2.0,  # Keyboard (verified: keyboard reviews)
            "B095JX15XF": 2.0,  # Keyboard (verified: "keyboards that are used")
        },
        "category": "keyboards_mice",
        "intent": "use_case",
    },
    {
        "query": "compact keyboard for small desk",
        "relevant_items": {
            "B07J21Y82K": 3.0,  # Keyboard (verified: "Great little keyboard")
            "B00MUTWLW4": 2.0,  # Keyboard (verified: keyboard slot)
            "B0BXYXD1Q2": 2.0,  # Keyboard (verified: "keyboard works well")
        },
        "category": "keyboards_mice",
        "intent": "feature_specific",
    },
    # === GIFT QUERIES ===
    {
        "query": "gift for someone who likes music",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo for music
            "B07S764D9V": 2.0,  # Headphones
            "B0778PCW73": 2.0,  # Wireless headphones
        },
        "category": "gifts",
        "intent": "gift",
    },
    {
        "query": "tech gift under 50 dollars",
        "relevant_items": {
            "B07H65KP63": 3.0,  # Echo Dot (affordable)
            "B07GZFM1ZM": 2.0,  # Fire TV Stick
            "B07S764D9V": 2.0,  # Budget headphones
        },
        "category": "gifts",
        "intent": "gift_budget",
    },
    {
        "query": "gift for elderly parent",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo easy to use
            "B07H65KP63": 3.0,  # Echo Dot
            "B010BWYDYA": 2.0,  # Simple tablet
        },
        "category": "gifts",
        "intent": "gift",
    },
    {
        "query": "christmas gift for tech lover",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick 4K
            "B01K8B8YA8": 3.0,  # Echo
            "B01MTF2Z37": 2.0,  # Fire tablet
        },
        "category": "gifts",
        "intent": "gift",
    },
    # === PROBLEM-SOLVING QUERIES ===
    {
        "query": "headphones that dont hurt ears",
        "relevant_items": {
            "B00PKTU83U": 3.0,  # Comfortable headphones
            "B0778PCW73": 2.0,  # Wireless headphones
        },
        "category": "headphones_audio",
        "intent": "problem_solving",
    },
    {
        "query": "tablet that doesnt freeze",
        "relevant_items": {
            "B01MTF2Z37": 3.0,  # Fire 10 runs smooth
            "B0051VVOB2": 2.0,  # Fire tablet
        },
        "category": "fire_tablets",
        "intent": "problem_solving",
    },
    {
        "query": "reliable wifi streaming device",
        "relevant_items": {
            "B07GZFM1ZM": 3.0,  # Fire TV Stick reliable
            "B0C3NMZ2C7": 3.0,  # Fire TV
        },
        "category": "fire_tv",
        "intent": "problem_solving",
    },
    # === COMPARISON / BEST QUERIES ===
    {
        "query": "best value fire tablet",
        "relevant_items": {
            "B01MTF2Z37": 3.0,  # Fire 10 best value
            "B010BWYDYA": 2.0,  # Fire tablet
        },
        "category": "fire_tablets",
        "intent": "comparison",
    },
    {
        "query": "best alexa device for sound",
        "relevant_items": {
            "B01K8B8YA8": 3.0,  # Echo tower better speaker
            "B0791TX5P5": 2.0,  # Echo with sound
        },
        "category": "echo_devices",
        "intent": "comparison",
    },
    {
        "query": "most reliable smart plug",
        "relevant_items": {
            "B0BN74ZJDK": 3.0,  # Smart plug high rating
            "B07KTYJ769": 2.0,  # Smart plug
        },
        "category": "smart_home",
        "intent": "comparison",
    },
]


def build_natural_eval_cases() -> list[EvalCase]:
    """Convert natural queries to EvalCase objects."""
    cases = []
    for item in NATURAL_QUERIES:
        cases.append(
            EvalCase(
                query=item["query"],
                relevant_items=item["relevant_items"],
                user_id=None,  # No user for natural queries
            )
        )
    return cases


def save_natural_eval_cases(
    cases: list[EvalCase], filename: str = "eval_natural_queries.json"
):
    """Save evaluation cases with metadata."""
    EVAL_DIR.mkdir(exist_ok=True)
    filepath = EVAL_DIR / filename

    # Include metadata for analysis
    data = []
    for i, item in enumerate(NATURAL_QUERIES):
        data.append(
            {
                "query": item["query"],
                "relevant_items": item["relevant_items"],
                "category": item.get("category", "unknown"),
                "intent": item.get("intent", "general"),
            }
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d natural language eval cases to: %s", len(data), filepath)
    return filepath


def analyze_dataset():
    """Print dataset statistics."""
    log_banner(logger, "NATURAL LANGUAGE EVALUATION DATASET")

    logger.info("Total queries: %d", len(NATURAL_QUERIES))

    # By category
    categories = Counter(q["category"] for q in NATURAL_QUERIES)
    logger.info("Queries by category:")
    for cat, count in categories.most_common():
        logger.info("  %s: %d", cat, count)

    # By intent
    intents = Counter(q["intent"] for q in NATURAL_QUERIES)
    logger.info("Queries by intent type:")
    for intent, count in intents.most_common():
        logger.info("  %s: %d", intent, count)

    # Relevance stats
    total_relevant = sum(len(q["relevant_items"]) for q in NATURAL_QUERIES)
    avg_relevant = total_relevant / len(NATURAL_QUERIES)
    logger.info("Avg relevant items per query: %.1f", avg_relevant)

    # Unique products
    all_products = set()
    for q in NATURAL_QUERIES:
        all_products.update(q["relevant_items"].keys())
    logger.info("Unique products in eval set: %d", len(all_products))

    # Sample queries
    log_section(logger, "SAMPLE QUERIES")
    for q in NATURAL_QUERIES[:5]:
        logger.info('Query: "%s"', q["query"])
        logger.info("  Category: %s | Intent: %s", q["category"], q["intent"])
        logger.info("  Relevant: %d products", len(q["relevant_items"]))


if __name__ == "__main__":
    analyze_dataset()

    cases = build_natural_eval_cases()
    save_natural_eval_cases(cases)

    log_banner(logger, "DATASET READY FOR EVALUATION")
    logger.info("Usage:")
    logger.info("  from sage.data import load_eval_cases")
    logger.info("  cases = load_eval_cases('eval_natural_queries.json')")
