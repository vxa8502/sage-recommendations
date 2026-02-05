"""
Server entry point.

Usage:
    python -m sage.api.run
    python -m sage.api.run --port 8000

For auto-reload during development, use uvicorn directly:
    uvicorn sage.api.app:create_app --factory --reload --port 8000
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from sage.api.app import create_app
from sage.config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Sage API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--port", type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port (defaults to PORT env var, then 8000)",
    )
    args = parser.parse_args()

    configure_logging()

    app = create_app()
    # Single worker: E5 + HHEM models consume ~500MB. Multiple workers
    # would duplicate model memory. Scale horizontally via container replicas.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
