"""Tests for the sage.data package boundary."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_python(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_import_sage_data_is_lightweight():
    output = _run_python(
        """
        import json
        import sys

        import sage.data

        roots = {
            root: sum(
                1
                for name in sys.modules
                if name == root or name.startswith(root + ".")
            )
            for root in ("pandas", "numpy", "requests", "tqdm")
        }
        print(json.dumps(roots, sort_keys=True))
        """
    )

    assert json.loads(output) == {
        "numpy": 0,
        "pandas": 0,
        "requests": 0,
        "tqdm": 0,
    }


def test_data_root_has_no_convenience_exports():
    output = _run_python(
        """
        import sage.data

        print(sage.data.__all__)
        print("load_eval_cases" in dir(sage.data))
        print("prepare_data" in dir(sage.data))
        print(hasattr(sage.data, "load_eval_cases"))
        """
    )

    assert output.splitlines() == [
        "[]",
        "False",
        "False",
        "False",
    ]


def test_public_helpers_live_on_owning_modules():
    output = _run_python(
        """
        from sage.data.eval import load_eval_cases
        from sage.data.loader import prepare_data
        from sage.data.query_bank.sources.boundary import (
            DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION,
            EVALUATION_SURFACE_RUNTIME_E2E,
        )

        print(load_eval_cases.__module__)
        print(prepare_data.__module__)
        print(DEFAULT_MANUAL_BOUNDARY_SELECTION_POLICY_VERSION)
        print(EVALUATION_SURFACE_RUNTIME_E2E)
        """
    )

    assert output.splitlines() == [
        "sage.data.eval",
        "sage.data.loader",
        "required_boundary_slice_v2",
        "runtime_e2e",
    ]
