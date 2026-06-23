"""Data package namespace.

Import public helpers from their owning modules, for example:

- `sage.data.loader` for review loading and split helpers
- `sage.data.eval` for legacy evaluation case loading
- `sage.data.query_bank` for canonical query-bank contracts
- `sage.data.faithfulness` for frozen explanation artifacts

The package root intentionally avoids convenience re-exports so lightweight
imports stay cheap and public ownership does not drift across modules.
"""

from __future__ import annotations

__all__: list[str] = []
