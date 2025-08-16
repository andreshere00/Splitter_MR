import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from splitter_mr.embedding.embeddings.azure_openai_embedding import AzureOpenAIEmbedding
from splitter_mr.schema import OPENAI_EMBEDDING_MAX_TOKENS

# --------- Helpers & Fixtures --------------------------------


class _FakeEmbeddingsClient:
    """Mimics the AzureOpenAI client with .embeddings.create(...)"""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.embeddings = types.SimpleNamespace(create=self._create)  # bind method

    def _create(self, **kwargs: Any):
        self.calls.append(kwargs)
        # Return an object with .data[0].embedding
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _FakeEncoder:
    """Simple fake tokenizer encoder that treats each character as a token."""

    def encode(self, text: str):
        # Each char is one "token" → deterministic & easy to exceed limits in tests
        return list(range(len(text)))


@pytest.fixture
def mod(monkeypatch):
    """
    Provide a handle to the module under test to patch its AzureOpenAI and tiktoken.
    """
    import importlib

    m = importlib.import_module(
        "splitter_mr.embedding.embeddings.azure_openai_embedding"
    )

    # patch AzureOpenAI constructor to return our fake client
    fake_client = _FakeEmbeddingsClient()
    monkeypatch.setattr(m, "AzureOpenAI", lambda **kwargs: fake_client)

    # patch tiktoken.encoding_for_model to return our fake encoder,
    # and capture the last model name for assertions
    state = {"last_model_name": None}

    def fake_encoding_for_model(name: str):
        state["last_model_name"] = name
        return _FakeEncoder()

    monkeypatch.setattr(m.tiktoken, "encoding_for_model", fake_encoding_for_model)

    # expose things for tests
    m._fake_client = fake_client
    m._encoding_state = state
    return m


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure relevant env vars start unset for each test, unless set explicitly."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)


# ----------- Test cases --------------------------------


def test_init_with_explicit_params_does_not_require_env(monkeypatch, mod):
    emb = AzureOpenAIEmbedding(
        model_name="ignored",
        api_key="k",
        azure_endpoint="https://example.azure.com",
        azure_deployment="dep-123",
        api_version="2025-04-14-preview",
    )
    assert emb.model_name == "dep-123"
    assert emb.get_client() is mod._fake_client  # the patched client


def test_init_reads_env_when_params_none(monkeypatch, mod):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep-env")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")

    emb = AzureOpenAIEmbedding()
    assert emb.model_name == "dep-env"
    assert emb.get_client() is mod._fake_client


def test_init_uses_model_name_as_fallback_for_deployment(monkeypatch, mod):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint.azure.com")
    # NOTE: no AZURE_OPENAI_DEPLOYMENT here

    emb = AzureOpenAIEmbedding(model_name="dep-from-model")
    assert emb.model_name == "dep-from-model"


def test_init_raises_if_missing_api_key(monkeypatch):
    # No env, no param
    with pytest.raises(ValueError) as e:
        AzureOpenAIEmbedding(
            model_name="anything",
            azure_endpoint="https://endpoint",
            azure_deployment="dep",
        )
    assert "API key" in str(e.value)


def test_init_raises_if_missing_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    with pytest.raises(ValueError) as e:
        AzureOpenAIEmbedding(
            model_name="anything",
            azure_deployment="dep",
        )
    assert "endpoint" in str(e.value).lower()


def test_init_raises_if_missing_deployment_and_model_name(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint")
    with pytest.raises(ValueError) as e:
        AzureOpenAIEmbedding()
    assert "deployment" in str(e.value).lower()


# ----------------------------- tests: get_client -------------------------------


def test_get_client_returns_client_instance(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    assert emb.get_client() is mod._fake_client


# ----------------------------- tests: embedding --------------------------------


def test_embed_text_happy_path_calls_sdk_and_returns_embedding(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )

    vec = emb.embed_text("hello", user="unit-test", trace_id="abc")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)
    assert vec == [0.1, 0.2, 0.3]

    # parameter forwarding + model/input correctness
    assert mod._fake_client.calls, "No SDK calls were recorded"
    last = mod._fake_client.calls[-1]
    assert last["model"] == "dep"
    assert last["input"] == "hello"
    assert last["user"] == "unit-test"
    assert last["trace_id"] == "abc"


@pytest.mark.parametrize("bad", ["", None])  # type: ignore[list-item]
def test_embed_text_rejects_empty_or_none(bad, mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    with pytest.raises(ValueError):
        emb.embed_text(bad)  # type: ignore[arg-type]


def test_embed_text_raises_when_tokens_exceed_limit(monkeypatch, mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    too_long = "x" * (OPENAI_EMBEDDING_MAX_TOKENS + 1)
    with pytest.raises(ValueError) as e:
        emb.embed_text(too_long)
    assert "exceeds maximum" in str(e.value).lower()


def test_tokenizer_called_with_model_name(monkeypatch, mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    emb.embed_text("ok")  # triggers _validate_token_length -> encoding_for_model(...)
    assert mod._encoding_state["last_model_name"] == "dep"
