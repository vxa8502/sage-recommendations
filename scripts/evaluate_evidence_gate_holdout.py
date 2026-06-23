#! /usr/bin/env python
# ruff: noqa: E402
"""
Compare evidence-gate thresholds on untouched holdout query-bank subsets.

Typical use:
    .venv/bin/python scripts/evaluate_evidence_gate_holdout.py
    .venv/bin/python scripts/evaluate_evidence_gate_holdout.py --subsets faithfulness_dev_seed
    .venv/bin/python scripts/evaluate_evidence_gate_holdout.py --candidate-tokens 40 --candidate-chunks 1 --candidate-score 0.6
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sage.services.calibration._holdout import main


if __name__ == "__main__":
    main()
