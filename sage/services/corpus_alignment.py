from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from sage.adapters.vector_store import (
    get_client,
    get_collection_info,
    get_corpus_anchor,
    upsert_corpus_anchor,
)
from sage.config import (
    COLLECTION_NAME,
    DATA_DIR,
    PROJECT_ROOT,
    QDRANT_SYSTEM_COLLECTION_NAME,
)
from sage.data.corpus_anchor import CorpusAnchorError, load_corpus_anchor


DEFAULT_CORPUS_ANCHOR_PATH = DATA_DIR / "indexed_product_ids.json"
FORCE_REPAIR_HINT = (
    "rerun with --force only if you are deliberately repairing stale metadata"
)


class CorpusAlignmentError(RuntimeError):
    """Raised when the live Qdrant collection is not aligned to the staged corpus."""


@dataclass(frozen=True)
class _AlignmentContext:
    client: object
    anchor_path: Path
    local_anchor: dict[str, Any]
    collection_info: Mapping[str, Any]
    points_count: int


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _as_strict_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _require_local_chunk_count(
    local_anchor: Mapping[str, Any],
    *,
    anchor_path: Path,
    for_stamping: bool,
) -> int:
    local_chunk_count = _as_strict_int(local_anchor.get("chunk_count"))
    if local_chunk_count is not None:
        return local_chunk_count
    if for_stamping:
        raise CorpusAlignmentError(
            f"Refusing to stamp {_display_path(anchor_path)} because it is "
            "missing an integer 'chunk_count'. Use the ingestion anchor emitted from "
            f"the indexed chunk set, or {FORCE_REPAIR_HINT}."
        )
    raise CorpusAlignmentError(
        f"Corpus anchor at {_display_path(anchor_path)} is missing an integer "
        "'chunk_count', so the live collection size cannot be verified."
    )


def _require_collection_points_count(
    collection_info: Mapping[str, Any],
    *,
    collection_name: str,
    for_stamping: bool,
) -> int:
    points_count = _as_strict_int(collection_info.get("points_count"))
    if points_count is not None:
        return points_count
    if for_stamping:
        raise CorpusAlignmentError(
            "Refusing to stamp remote corpus metadata because Qdrant collection "
            f"{collection_name!r} did not report a usable points_count."
        )
    raise CorpusAlignmentError(
        f"Configured Qdrant collection {collection_name!r} did not report a usable points_count."
    )


def _assert_chunk_count_matches_points_count(
    *,
    local_chunk_count: int,
    points_count: int,
    collection_name: str,
    anchor_path: Path,
    for_stamping: bool,
) -> None:
    if local_chunk_count == points_count:
        return
    if for_stamping:
        raise CorpusAlignmentError(
            "Refusing to stamp the remote corpus anchor because the live Qdrant "
            f"collection has points_count={points_count}, but "
            f"{_display_path(anchor_path)} reports chunk_count={local_chunk_count}. "
            f"Reindex the cluster or {FORCE_REPAIR_HINT}."
        )
    raise CorpusAlignmentError(
        f"Qdrant collection {collection_name!r} is misaligned with "
        f"{_display_path(anchor_path)}: points_count={points_count} "
        f"but chunk_count={local_chunk_count}."
    )


def _load_alignment_context(
    *,
    anchor_path: str | Path,
    client: object | None,
    collection_name: str,
    for_stamping: bool,
) -> _AlignmentContext:
    resolved_anchor_path = Path(anchor_path)
    try:
        local_anchor = load_corpus_anchor(resolved_anchor_path)
    except FileNotFoundError as exc:
        raise CorpusAlignmentError(
            f"Corpus anchor not found at {_display_path(resolved_anchor_path)}."
        ) from exc
    except CorpusAnchorError as exc:
        raise CorpusAlignmentError(
            f"Corpus anchor at {_display_path(resolved_anchor_path)} is invalid: {exc}"
        ) from exc
    except OSError as exc:
        raise CorpusAlignmentError(
            "Corpus anchor at "
            f"{_display_path(resolved_anchor_path)} could not be read: {exc}"
        ) from exc

    active_client = client if client is not None else get_client()
    collection_info = get_collection_info(
        active_client, collection_name=collection_name
    )
    points_count = _require_collection_points_count(
        collection_info,
        collection_name=collection_name,
        for_stamping=for_stamping,
    )
    return _AlignmentContext(
        client=active_client,
        anchor_path=resolved_anchor_path,
        local_anchor=local_anchor,
        collection_info=collection_info,
        points_count=points_count,
    )


def _assert_local_count_matches_collection(
    context: _AlignmentContext,
    *,
    collection_name: str,
    for_stamping: bool,
) -> int:
    local_chunk_count = _require_local_chunk_count(
        context.local_anchor,
        anchor_path=context.anchor_path,
        for_stamping=for_stamping,
    )
    _assert_chunk_count_matches_points_count(
        local_chunk_count=local_chunk_count,
        points_count=context.points_count,
        collection_name=collection_name,
        anchor_path=context.anchor_path,
        for_stamping=for_stamping,
    )
    return local_chunk_count


def _common_status_payload(
    *,
    context: _AlignmentContext,
    collection_name: str,
    metadata_collection_name: str,
    chunk_count: int | None,
) -> dict[str, Any]:
    return {
        "collection_name": collection_name,
        "metadata_collection_name": metadata_collection_name,
        "local_anchor_path": _display_path(context.anchor_path),
        "corpus_fingerprint": context.local_anchor["corpus_fingerprint"],
        "chunk_count": chunk_count,
        "collection_points_count": context.points_count,
    }


def stamp_corpus_anchor(
    *,
    anchor_path: str | Path = DEFAULT_CORPUS_ANCHOR_PATH,
    client: object | None = None,
    collection_name: str = COLLECTION_NAME,
    metadata_collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
    force: bool = False,
) -> dict[str, Any]:
    """Stamp the staged local corpus anchor into the configured Qdrant metadata store."""
    context = _load_alignment_context(
        anchor_path=anchor_path,
        client=client,
        collection_name=collection_name,
        for_stamping=True,
    )
    if not force:
        _assert_local_count_matches_collection(
            context,
            collection_name=collection_name,
            for_stamping=True,
        )

    remote_payload = upsert_corpus_anchor(
        context.client,
        context.local_anchor,
        collection_name=collection_name,
        metadata_collection_name=metadata_collection_name,
        collection_points_count=context.points_count,
    )
    return {
        "status": "stamped",
        **_common_status_payload(
            context=context,
            collection_name=collection_name,
            metadata_collection_name=metadata_collection_name,
            chunk_count=context.local_anchor.get("chunk_count"),
        ),
        "stamped_at": remote_payload.get("stamped_at"),
    }


def assert_corpus_alignment(
    *,
    anchor_path: str | Path = DEFAULT_CORPUS_ANCHOR_PATH,
    client: object | None = None,
    collection_name: str = COLLECTION_NAME,
    metadata_collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
    require_remote_anchor: bool = True,
) -> dict[str, Any]:
    """Fail closed unless the staged local corpus anchor matches the live Qdrant state."""
    context = _load_alignment_context(
        anchor_path=anchor_path,
        client=client,
        collection_name=collection_name,
        for_stamping=False,
    )
    local_chunk_count = _assert_local_count_matches_collection(
        context,
        collection_name=collection_name,
        for_stamping=False,
    )

    remote_payload = get_corpus_anchor(
        context.client,
        collection_name=collection_name,
        metadata_collection_name=metadata_collection_name,
    )
    remote_anchor = (
        remote_payload.get("anchor") if isinstance(remote_payload, Mapping) else None
    )
    if not isinstance(remote_anchor, Mapping):
        remote_anchor = None

    if require_remote_anchor and remote_anchor is None:
        raise CorpusAlignmentError(
            f"No remote corpus anchor is stamped for {collection_name!r} in "
            f"{metadata_collection_name!r}. Run `sage qdrant stamp-anchor` after "
            "verifying the cluster contains the staged corpus."
        )

    remote_fingerprint = None
    if remote_anchor is not None:
        remote_fingerprint = remote_anchor.get("corpus_fingerprint")
        local_fingerprint = context.local_anchor.get("corpus_fingerprint")
        if remote_fingerprint != local_fingerprint:
            raise CorpusAlignmentError(
                f"Corpus fingerprint mismatch for {collection_name!r}: local="
                f"{local_fingerprint} remote={remote_fingerprint}. The live cluster "
                "does not match the staged ingestion corpus."
            )

    return {
        "status": "aligned",
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **_common_status_payload(
            context=context,
            collection_name=collection_name,
            metadata_collection_name=metadata_collection_name,
            chunk_count=local_chunk_count,
        ),
        "dataset_category": context.local_anchor["dataset_category"],
        "subset_size": context.local_anchor["subset_size"],
        "review_count": context.local_anchor.get("review_count"),
        "product_count": context.local_anchor["product_count"],
        "product_ids_sha256": context.local_anchor["product_ids_sha256"],
        "collection_status": str(context.collection_info.get("status")),
        "remote_anchor_present": remote_anchor is not None,
        "remote_stamped_at": (
            remote_payload.get("stamped_at")
            if isinstance(remote_payload, Mapping)
            else None
        ),
        "remote_corpus_fingerprint": remote_fingerprint,
    }


def get_corpus_alignment_status(
    *,
    anchor_path: str | Path = DEFAULT_CORPUS_ANCHOR_PATH,
    client: object | None = None,
    collection_name: str = COLLECTION_NAME,
    metadata_collection_name: str = QDRANT_SYSTEM_COLLECTION_NAME,
    require_remote_anchor: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Return a non-throwing alignment status payload for CLI/status surfaces."""
    try:
        proof = assert_corpus_alignment(
            anchor_path=anchor_path,
            client=client,
            collection_name=collection_name,
            metadata_collection_name=metadata_collection_name,
            require_remote_anchor=require_remote_anchor,
        )
    except CorpusAlignmentError as exc:
        return False, {"status": "misaligned", "error": str(exc)}
    except Exception as exc:
        return False, {"status": "error", "error": str(exc)}
    return True, proof
