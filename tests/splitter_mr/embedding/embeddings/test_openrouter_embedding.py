import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from splitter_mr.embedding.embeddings.openrouter_embedding import OpenRouterEmbedding
from splitter_mr.schema import OPENAI_EMBEDDING_MAX_TOKENS


class _FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any):
        self.calls.append(kwargs)
        inp = kwargs["input"]
        if isinstance(inp, list):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in inp]
            )
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeEncoder:
    def encode(self, text: str):
        return list(range(len(text)))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def mod(monkeypatch):
    import importlib

    m = importlib.import_module("splitter_mr.embedding.embeddings.openrouter_embedding")

    fake_client = _FakeEmbeddingsClient()
    captured: dict = {}

    def factory(**kwargs):
        captured["kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(m, "OpenAI", factory)

    state = {"last_model_name": None}

    def fake_encoding_for_model(name: str):
        state["last_model_name"] = name
        return _FakeEncoder()

    monkeypatch.setattr(m.tiktoken, "encoding_for_model", fake_encoding_for_model)

    m._fake_client = fake_client
    m._encoding_state = state
    m._openai_captured = captured
    return m


def test_init_with_explicit_api_key(mod):
    emb = OpenRouterEmbedding(
        model_name="openai/text-embedding-3-large", api_key="sk-test"
    )
    assert emb.model_name == "openai/text-embedding-3-large"
    assert emb.get_client() is mod._fake_client
    assert mod._openai_captured["kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


def test_init_reads_api_key_from_env(monkeypatch, mod):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-small")
    assert emb.get_client() is mod._fake_client


def test_init_raises_if_missing_api_key():
    with pytest.raises(ValueError) as e:
        OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key=None)
    assert "OPENROUTER_API_KEY" in str(e.value)


def test_init_attribution_headers(mod):
    OpenRouterEmbedding(
        api_key="sk",
        site_url="https://example.com",
        app_name="Splitter MR",
    )
    headers = mod._openai_captured["kwargs"]["default_headers"]
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-OpenRouter-Title"] == "Splitter MR"


def test_embed_text_happy_path(mod):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    vec = emb.embed_text("hello world", user="unit-test")

    assert vec == [0.1, 0.2, 0.3]
    last = mod._fake_client.calls[-1]
    assert last["model"] == "openai/text-embedding-3-large"
    assert last["input"] == "hello world"
    assert last["user"] == "unit-test"


@pytest.mark.parametrize("bad", ["", None])
def test_embed_text_rejects_empty_or_none_input(bad):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    with pytest.raises(ValueError):
        emb.embed_text(bad)


def test_embed_text_raises_when_tokens_exceed_limit(mod):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    too_long = "x" * (OPENAI_EMBEDDING_MAX_TOKENS + 1)
    with pytest.raises(ValueError) as e:
        emb.embed_text(too_long)
    assert "exceeds maximum" in str(e.value).lower()


def test_get_encoder_fallback_on_exception(monkeypatch, mod):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    monkeypatch.setattr(
        mod.tiktoken,
        "encoding_for_model",
        lambda name: (_ for _ in ()).throw(Exception("fail")),
    )
    fallback_called = {}

    class DummyEncoding:
        def encode(self, text):
            return [1, 2, 3]

    def fake_get_encoding(name):
        fallback_called["name"] = name
        return DummyEncoding()

    monkeypatch.setattr(mod.tiktoken, "get_encoding", fake_get_encoding)
    emb._get_encoder()
    assert fallback_called["name"] == mod.OPENAI_EMBEDDING_MODEL_FALLBACK


def test_embed_documents_happy_path(mod):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    out = emb.embed_documents(["hello", "world"], foo="bar")
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    last = mod._fake_client.calls[-1]
    assert last["input"] == ["hello", "world"]
    assert last["foo"] == "bar"


def test_embed_documents_rejects_empty_list(mod):
    emb = OpenRouterEmbedding(model_name="openai/text-embedding-3-large", api_key="sk")
    with pytest.raises(ValueError):
        emb.embed_documents([])
