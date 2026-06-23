"""
Public API for cross-surface query leakage audits.

The canonical ingestion workflow treats `gate_calibration`, `retrieval_eval`,
and `faithfulness_seed` as distinct experimental surfaces. The package modules
separate the concerns behind that audit: pair scoring, component grouping,
surface selection, matrix aggregation, and persistence.
"""

from __future__ import annotations

from sage.data.split_leakage._components import build_strong_paraphrase_components
from sage.data.split_leakage._config import (
    DEFAULT_MATRIX_SURFACE_SPECS,
    DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION,
    DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY,
    DEFAULT_SPLIT_LEAKAGE_AUDIT_VERSION,
    SPLIT_LEAKAGE_AUDIT_PATH,
)
from sage.data.split_leakage._io import save_split_leakage_audit
from sage.data.split_leakage._matrix import build_split_leakage_matrix_audit


__all__ = [
    "DEFAULT_PARAPHRASE_COMPONENT_EDGE_POLICY_VERSION",
    "DEFAULT_PARAPHRASE_COMPONENT_GROUP_KEY",
    "DEFAULT_MATRIX_SURFACE_SPECS",
    "DEFAULT_SPLIT_LEAKAGE_AUDIT_VERSION",
    "SPLIT_LEAKAGE_AUDIT_PATH",
    "build_strong_paraphrase_components",
    "build_split_leakage_matrix_audit",
    "save_split_leakage_audit",
]
