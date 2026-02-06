"""
Structured logging configuration for Sage.

Provides consistent logging across all scripts with support for:
- Console output (human-readable, with colors)
- JSON format (machine-parseable, for production)
- Configurable log levels via environment variable

Usage:
    from sage.config.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing query", extra={"query": query, "k": 10})
"""

import logging
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("SAGE_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("SAGE_LOG_FORMAT", "console")  # "console" or "json"

# Standard LogRecord attributes to ignore when extracting user-specified extras.
# These are built-in attributes from logging.LogRecord plus taskName from asyncio.
_STANDARD_LOG_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "message",
        "asctime",
        "taskName",
    }
)


# ---------------------------------------------------------------------------
# Custom Formatter (Console)
# ---------------------------------------------------------------------------


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter with visual hierarchy."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Check if we're in a TTY (supports colors)
        use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Format level with optional color
        level = record.levelname
        if use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:<8}{reset}"
        else:
            level_str = f"{level:<8}"

        # Format message
        message = record.getMessage()

        # Add extra fields if present (only user-specified ones)
        extras = []
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                extras.append(f"{key}={value}")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        return f"{timestamp} {level_str} {message}{extra_str}"


# ---------------------------------------------------------------------------
# Custom Formatter (JSON)
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Machine-parseable JSON formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields (only user-specified ones)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

_configured = False


def configure_logging() -> None:
    """Configure root logger with appropriate handler and formatter."""
    global _configured
    if _configured:
        return

    # Get root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Set formatter based on config
    if LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())

    root.addHandler(handler)

    # Quiet down noisy third-party loggers
    for noisy_logger in [
        "httpx",
        "httpcore",
        "urllib3",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "qdrant_client",
        "anthropic",
        "openai",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Convenience functions for visual output
# ---------------------------------------------------------------------------


def log_banner(
    logger: logging.Logger, title: str, char: str = "=", width: int = 60
) -> None:
    """Log a visual banner for section headers."""
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_section(
    logger: logging.Logger, title: str, char: str = "-", width: int = 60
) -> None:
    """Log a section divider."""
    logger.info("")
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_kv(logger: logging.Logger, key: str, value: Any, indent: int = 2) -> None:
    """Log a key-value pair with consistent formatting."""
    prefix = " " * indent
    if isinstance(value, float):
        logger.info(f"{prefix}{key}: {value:.4f}")
    elif isinstance(value, int):
        logger.info(f"{prefix}{key}: {value:,}")
    else:
        logger.info(f"{prefix}{key}: {value}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "get_logger",
    "configure_logging",
    "log_banner",
    "log_section",
    "log_kv",
    "LOG_LEVEL",
    "LOG_FORMAT",
]
