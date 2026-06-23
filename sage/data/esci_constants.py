"""Lightweight shared constants for Amazon ESCI source handling."""

from __future__ import annotations

DEFAULT_ESCI_LOCALE = "us"
DEFAULT_ESCI_VERSION = "large"
ESCI_VERSION_CHOICES = ("large", "small", "all")


def require_esci_version(value: str) -> str:
    """Validate an ESCI version selector."""
    if value not in ESCI_VERSION_CHOICES:
        choices = ", ".join(f"{choice!r}" for choice in ESCI_VERSION_CHOICES)
        raise ValueError(f"version must be one of: {choices}")
    return value
