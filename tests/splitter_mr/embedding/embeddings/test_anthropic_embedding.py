import builtins
import sys
from types import ModuleType
from typing import Any, Dict, List

import pytest

from splitter_mr.embedding.embeddings.anthropic_embedding import AnthropicEmbedding


class _FakeEmbedResult:
    def __init__(self, embeddings: List[List[float]]):
        self.embeddings = embeddings


class _FakeVoyageClient:
    """
    A minimal fake voyageai.Client that records the last embed call.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls: List[Dict[str, Any]] = []

    def embed(self, inputs: List[str], model: str, **kwargs):
        self.calls.append({"inputs": inputs, "model": model, "kwargs": kwargs})
        # Default: return a matching number of dummy vectors
        return _FakeEmbedResult([[0.1, 0.2]] * len(inputs))


def _install_fake_voyageai(monkeypatch, client_factory=None):
    """
    Inject a fake 'voyageai' module into sys.modules so the class can import it.
    """
    fake_module = ModuleType("voyageai")

    # Allow tests to inject custom client factories (e.g., to return different results)
    def _client(api_key: str):
        if client_factory:
            return client_factory(api_key)
        return _FakeVoyageClient(api_key)

    fake_module.Client = _client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voyageai", fake_module)
    return fake_module


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    yield


# ---- Test cases ---- #

# -------------------
# initialization tests
# -------------------


def test_init_uses_env_key_and_requires_multimodal_extra(monkeypatch):
    # Provide env key
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key-123")

    # Ensure the 'multimodal' extra check passes by making voyageai importable
    _install_fake_voyageai(monkeypatch)

    emb = AnthropicEmbedding(model_name="voyage-3.5")
    assert getattr(emb, "model_name") == "voyage-3.5"
    client = emb.get_client()
    # Our fake client stores the key
    assert isinstance(client, _FakeVoyageClient)
    assert client.api_key == "env-key-123"


def test_init_with_explicit_key(monkeypatch):
    _install_fake_voyageai(monkeypatch)

    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="explicit-key")
    client = emb.get_client()
    assert isinstance(client, _FakeVoyageClient)
    assert client.api_key == "explicit-key"


def test_init_raises_when_no_key_and_env_missing(monkeypatch):
    _install_fake_voyageai(monkeypatch)
    with pytest.raises(ValueError) as ei:
        AnthropicEmbedding(model_name="voyage-3.5")
    assert "VOYAGE_API_KEY" in str(ei.value)


def test_init_raises_when_multimodal_extra_missing(monkeypatch):
    """
    Simulate that importing 'voyageai' fails inside _require_extra('multimodal', 'voyageai').
    The class should raise ImportError with a helpful message.
    """
    # Remove voyageai from sys.modules if present
    monkeypatch.setitem(sys.modules, "voyageai", None)
    monkeypatch.delitem(sys.modules, "voyageai", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "voyageai":
            raise ImportError("No module named 'voyageai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as ei:
        AnthropicEmbedding(model_name="voyage-3.5", api_key="k")
    # Helpful installation hint
    assert "pip install splitter-mr[multimodal]" in str(ei.value)


def test_get_client_returns_underlying_client(monkeypatch):
    _install_fake_voyageai(monkeypatch)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    assert isinstance(emb.get_client(), _FakeVoyageClient)


# -------------------
# embed_text tests
# -------------------


def test_embed_text_success_and_defaults_input_type_document(monkeypatch):
    client_holder = {"client": None}

    def factory(api_key: str):
        client = _FakeVoyageClient(api_key)
        client_holder["client"] = client
        return client

    _install_fake_voyageai(monkeypatch, client_factory=factory)

    emb = AnthropicEmbedding(
        model_name="voyage-3.5", api_key="x", default_input_type="document"
    )
    vec = emb.embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 2

    # Verify the call captured default input_type
    calls = client_holder["client"].calls
    assert len(calls) == 1
    assert calls[0]["inputs"] == ["hello world"]
    assert calls[0]["model"] == "voyage-3.5"
    assert calls[0]["kwargs"].get("input_type") == "document"


def test_embed_text_respects_override_input_type_query(monkeypatch):
    client_holder = {"client": None}

    def factory(api_key: str):
        client = _FakeVoyageClient(api_key)
        client_holder["client"] = client
        return client

    _install_fake_voyageai(monkeypatch, client_factory=factory)

    emb = AnthropicEmbedding(
        model_name="voyage-3.5", api_key="x", default_input_type="document"
    )
    _ = emb.embed_text("what is RAG?", input_type="query")

    calls = client_holder["client"].calls
    assert calls[0]["kwargs"].get("input_type") == "query"


def test_embed_text_raises_on_empty_string(monkeypatch):
    _install_fake_voyageai(monkeypatch)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    with pytest.raises(ValueError):
        emb.embed_text("   ")


def test_embed_text_malformed_response_no_embeddings(monkeypatch):
    class _BadClient(_FakeVoyageClient):
        def embed(self, inputs, model, **kwargs):
            return object()  # no .embeddings attribute

    def factory(api_key: str):
        return _BadClient(api_key)

    _install_fake_voyageai(monkeypatch, client_factory=factory)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    with pytest.raises(RuntimeError) as ei:
        emb.embed_text("hello")
    assert "malformed embeddings response" in str(ei.value)


def test_embed_text_malformed_vector_shape(monkeypatch):
    class _BadClient(_FakeVoyageClient):
        def embed(self, inputs, model, **kwargs):
            # embeddings exists but is empty / wrong
            return _FakeEmbedResult([])

    def factory(api_key: str):
        return _BadClient(api_key)

    _install_fake_voyageai(monkeypatch, client_factory=factory)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    with pytest.raises(RuntimeError) as ei:
        emb.embed_text("hello")
    assert "malformed embeddings response" in str(
        ei.value
    ) or "invalid embedding vector" in str(ei.value)


# -------------------
# embed_documents tests
# -------------------


def test_embed_documents_success_and_count(monkeypatch):
    client_holder = {"client": None}

    def factory(api_key: str):
        client = _FakeVoyageClient(api_key)
        client_holder["client"] = client
        return client

    _install_fake_voyageai(monkeypatch, client_factory=factory)

    emb = AnthropicEmbedding(
        model_name="voyage-3.5", api_key="x", default_input_type="document"
    )
    texts = ["A", "B", "C"]
    vecs = emb.embed_documents(texts)
    assert isinstance(vecs, list)
    assert len(vecs) == 3
    assert all(len(v) == 2 for v in vecs)

    # Verify call received the list and default input_type
    calls = client_holder["client"].calls
    assert len(calls) == 1
    assert calls[0]["inputs"] == texts
    assert calls[0]["kwargs"].get("input_type") == "document"


def test_embed_documents_respects_override_input_type_query(monkeypatch):
    client_holder = {"client": None}

    def factory(api_key: str):
        client = _FakeVoyageClient(api_key)
        client_holder["client"] = client
        return client

    _install_fake_voyageai(monkeypatch, client_factory=factory)
    emb = AnthropicEmbedding(
        model_name="voyage-3.5", api_key="x", default_input_type="document"
    )
    _ = emb.embed_documents(["q1", "q2"], input_type="query")

    calls = client_holder["client"].calls
    assert calls[0]["kwargs"].get("input_type") == "query"


def test_embed_documents_validates_inputs(monkeypatch):
    _install_fake_voyageai(monkeypatch)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")

    with pytest.raises(ValueError):
        emb.embed_documents([])

    with pytest.raises(ValueError):
        emb.embed_documents(["valid", "   "])

    with pytest.raises(ValueError):
        emb.embed_documents(["valid", None])  # type: ignore[list-item]


def test_embed_documents_malformed_response_count_mismatch(monkeypatch):
    class _MismatchClient(_FakeVoyageClient):
        def embed(self, inputs, model, **kwargs):
            # Return fewer embeddings than inputs to trigger mismatch
            return _FakeEmbedResult([[0.1, 0.2]] * (len(inputs) - 1))

    def factory(api_key: str):
        return _MismatchClient(api_key)

    _install_fake_voyageai(monkeypatch, client_factory=factory)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    with pytest.raises(RuntimeError) as ei:
        emb.embed_documents(["one", "two", "three"])
    assert "embeddings for 3 inputs" in str(ei.value)


def test_embed_documents_malformed_response_no_embeddings(monkeypatch):
    class _BadClient(_FakeVoyageClient):
        def embed(self, inputs, model, **kwargs):
            return object()  # no .embeddings

    def factory(api_key: str):
        return _BadClient(api_key)

    _install_fake_voyageai(monkeypatch, client_factory=factory)
    emb = AnthropicEmbedding(model_name="voyage-3.5", api_key="x")
    with pytest.raises(RuntimeError) as ei:
        emb.embed_documents(["x", "y"])
    assert "malformed embeddings response" in str(ei.value)
