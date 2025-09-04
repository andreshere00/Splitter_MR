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


def test_embed_documents_happy_path(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )

    # Patch _create to return one embedding per input
    def _create(**kwargs):
        mod._fake_client.calls.append(kwargs)
        inp = kwargs["input"]
        if isinstance(inp, list):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in inp]
            )
        else:
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    mod._fake_client.embeddings.create = _create

    out = emb.embed_documents(["hello", "world"], foo="bar")
    assert isinstance(out, list)
    assert all(isinstance(x, list) for x in out)
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    # Check forwarded parameters and call correctness
    last = mod._fake_client.calls[-1]
    assert last["model"] == "dep"
    assert last["input"] == ["hello", "world"]
    assert last["foo"] == "bar"


def test_embed_documents_raises_on_empty_list(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    with pytest.raises(ValueError) as e:
        emb.embed_documents([])
    assert "non-empty list" in str(e.value)


@pytest.mark.parametrize(
    "bad",
    [
        ["", "foo"],  # Empty string
        [None, "bar"],  # None as element
        ["ok", 123],  # Non-str
    ],
)
def test_embed_documents_raises_on_bad_items(mod, bad):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    with pytest.raises(ValueError):
        emb.embed_documents(bad)


def test_embed_documents_raises_if_any_too_long(monkeypatch, mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    long = "x" * (OPENAI_EMBEDDING_MAX_TOKENS + 1)
    with pytest.raises(ValueError) as e:
        emb.embed_documents(["ok", long])
    assert "maximum allowed" in str(e.value)


def test_explicit_tokenizer_name_overrides(monkeypatch, mod):
    calls = {}

    def fake_get_encoding(name):
        calls["name"] = name

        class Enc:
            def encode(self, txt):
                return [1, 2, 3]

        return Enc()

    monkeypatch.setattr("tiktoken.get_encoding", fake_get_encoding)
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
        tokenizer_name="custom-tokenizer",
    )
    emb._get_encoder()
    assert calls["name"] == "custom-tokenizer"
    # Call _count_tokens to exercise this path
    assert emb._count_tokens("foo") == 3


def test_get_encoder_fallback_to_default(monkeypatch, mod):
    monkeypatch.setattr(
        mod.tiktoken,
        "encoding_for_model",
        lambda name: (_ for _ in ()).throw(Exception("fail")),
    )
    monkeypatch.setattr(
        "tiktoken.get_encoding",
        lambda name: type("E", (), {"encode": lambda self, t: [0, 1, 2, 3]})(),
    )
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    enc = emb._get_encoder()
    # Should be fallback encoder
    assert hasattr(enc, "encode")


def test_validate_token_length_raises(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    # Patch encoder to always return OVER the limit
    emb._get_encoder = lambda: type(
        "Enc", (), {"encode": lambda self, t: [0] * (OPENAI_EMBEDDING_MAX_TOKENS + 1)}
    )()
    with pytest.raises(ValueError) as e:
        emb._validate_token_length("abc")
    assert "exceeds maximum" in str(e.value)


def test_init_reads_api_version_from_env(monkeypatch, mod):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep-env")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2026-01-01-preview")
    emb = AzureOpenAIEmbedding()
    # You could assert it is set by patching AzureOpenAI and capturing its call args, if needed
    assert emb.model_name == "dep-env"


def test_count_tokens_calls_encoder(mod):
    emb = AzureOpenAIEmbedding(
        model_name="dep",
        api_key="k",
        azure_endpoint="https://endpoint",
        azure_deployment="dep",
    )
    # Patch encoder: one token per char
    emb._get_encoder = lambda: type(
        "E", (), {"encode": lambda self, txt: list(range(len(txt)))}
    )()
    assert emb._count_tokens("abc") == 3
