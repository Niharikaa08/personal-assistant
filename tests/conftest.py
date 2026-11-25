"""Shared test setup: keep unit tests fast and offline."""
import pytest

import assistant.rag as rag


@pytest.fixture(autouse=True)
def _no_dense_model_download(monkeypatch):
    """Unit tests use the lexical path so they never download the embedding model."""
    monkeypatch.setattr(rag, "_EMBEDDER", False)


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    import assistant.llm as llm
    llm._MODEL_CACHE.clear()
    yield
    llm._MODEL_CACHE.clear()
