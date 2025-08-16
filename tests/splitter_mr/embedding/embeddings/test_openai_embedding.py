import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from splitter_mr.embedding.embeddings.openai_embedding import OpenAIEmbedding
from splitter_mr.schema import OPENAI_EMBEDDING_MAX_TOKENS

# --------- Helpers & Fixtures --------------------------------


class _FakeEmbeddingsClient:
    """Mimics OpenAI client with .embeddings.create(...), recording calls."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any):
        self.calls.append(kwargs)
        # Return shape compatible with response.data[0].embedding
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeEncoder:
    """Simple fake tokenizer: each character -> one token (easy to test limits)."""

    def encode(self, text: str):
        return list(range(len(text)))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure OPENAI_API_KEY starts unset for each test unless set explicitly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def mod(monkeypatch):
    """
    Import the module under test and patch its OpenAI constructor and tiktoken.
    Exposes the fake client and last model name used by tiktoken.
    """
    import importlib

    m = importlib.import_module("splitter_mr.embedding.embeddings.openai_embedding")

    fake_client = _FakeEmbeddingsClient()
    # Patch OpenAI(...) to return our fake client instead of the real SDK client
    monkeypatch.setattr(m, "OpenAI", lambda **kwargs: fake_client)

    state = {"last_model_name": None}

    def fake_encoding_for_model(name: str):
        state["last_model_name"] = name
        return _FakeEncoder()

    monkeypatch.setattr(m.tiktoken, "encoding_for_model", fake_encoding_for_model)

    # Expose patched artifacts for assertions
    m._fake_client = fake_client
    m._encoding_state = state
    return m


# ----------- Test cases --------------------------------


def test_init_with_explicit_api_key_does_not_require_env(mod):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk-test")
    assert emb.model_name == "text-embedding-3-large"
    assert emb.get_client() is mod._fake_client


def test_init_reads_api_key_from_env_when_not_provided(monkeypatch, mod):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    emb = OpenAIEmbedding(model_name="text-embedding-3-small")
    assert emb.model_name == "text-embedding-3-small"
    assert emb.get_client() is mod._fake_client


def test_init_raises_if_missing_api_key():
    with pytest.raises(ValueError) as e:
        OpenAIEmbedding(model_name="text-embedding-3-large", api_key=None)
    assert "OPENAI_API_KEY" in str(e.value)


# ----------------------------- tests: get_client -------------------------------


def test_get_client_returns_client_instance(mod):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk")
    assert emb.get_client() is mod._fake_client


# ----------------------------- tests: embed_text -------------------------------


def test_embed_text_happy_path_calls_sdk_and_returns_embedding(mod):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk")
    vec = emb.embed_text("hello world", user="unit-test", trace_id="abc123")

    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert vec == [0.1, 0.2, 0.3]

    # Assert parameter forwarding and correct fields
    assert mod._fake_client.calls, "No embeddings.create calls recorded"
    last = mod._fake_client.calls[-1]
    assert last["model"] == "text-embedding-3-large"
    assert last["input"] == "hello world"
    assert last["user"] == "unit-test"
    assert last["trace_id"] == "abc123"


@pytest.mark.parametrize("bad", ["", None])  # type: ignore[list-item]
def test_embed_text_rejects_empty_or_none_input(bad):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk")
    with pytest.raises(ValueError):
        emb.embed_text(bad)  # type: ignore[arg-type]


def test_embed_text_raises_when_tokens_exceed_limit(mod):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk")
    too_long = "x" * (
        OPENAI_EMBEDDING_MAX_TOKENS + 1
    )  # 1 char = 1 token in fake encoder
    with pytest.raises(ValueError) as e:
        emb.embed_text(too_long)
    assert "exceeds maximum" in str(e.value).lower()


def test_tokenizer_called_with_model_name(mod):
    emb = OpenAIEmbedding(model_name="text-embedding-3-large", api_key="sk")
    emb.embed_text("ok")
    # ensure tiktoken.encoding_for_model was invoked with the model name
    assert mod._encoding_state["last_model_name"] == "text-embedding-3-large"
