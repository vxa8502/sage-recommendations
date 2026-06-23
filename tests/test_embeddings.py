from types import SimpleNamespace
import json

import numpy as np

from sage.adapters.embeddings import E5Embedder


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.calls = 0
        self.tokenizer = _FakeTokenizer()

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert show_progress_bar in {False, True}
        assert normalize_embeddings is True
        self.calls += 1
        return np.array(
            [[float(sum(ord(char) for char in text))] for text in texts],
            dtype=float,
        )


def _patch_sentence_transformers(monkeypatch) -> None:
    monkeypatch.setattr(
        "sage.adapters.embeddings.require_import",
        lambda *args, **kwargs: SimpleNamespace(
            SentenceTransformer=_FakeSentenceTransformer
        ),
    )


def test_embed_passages_reuses_cache_with_matching_metadata(
    tmp_path, monkeypatch
) -> None:
    _patch_sentence_transformers(monkeypatch)
    embedder = E5Embedder(model_name="test-e5")
    cache_path = tmp_path / "embeddings.npy"

    first = embedder.embed_passages(
        ["alpha", "beta"],
        cache_path=cache_path,
        show_progress=False,
    )
    second = embedder.embed_passages(
        ["alpha", "beta"],
        cache_path=cache_path,
        show_progress=False,
    )

    assert embedder.model.calls == 1
    assert np.array_equal(first, second)
    assert (tmp_path / "embeddings.npy.meta.json").exists()


def test_embed_passages_invalidates_cache_when_texts_change_but_count_matches(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_sentence_transformers(monkeypatch)
    embedder = E5Embedder(model_name="test-e5")
    cache_path = tmp_path / "embeddings.npy"

    first = embedder.embed_passages(
        ["alpha", "beta"],
        cache_path=cache_path,
        show_progress=False,
    )
    second = embedder.embed_passages(
        ["gamma", "delta"],
        cache_path=cache_path,
        show_progress=False,
    )

    assert embedder.model.calls == 2
    assert not np.array_equal(first, second)


def test_embed_passages_ignores_corrupt_metadata_and_reembeds(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_sentence_transformers(monkeypatch)
    embedder = E5Embedder(model_name="test-e5")
    cache_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.npy.meta.json"

    first = embedder.embed_passages(
        ["alpha", "beta"],
        cache_path=cache_path,
        show_progress=False,
    )
    metadata_path.write_text("{not valid json", encoding="utf-8")

    second = embedder.embed_passages(
        ["alpha", "beta"],
        cache_path=cache_path,
        show_progress=False,
    )

    assert embedder.model.calls == 2
    assert np.array_equal(first, second)
    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert repaired["model_name"] == "test-e5"
