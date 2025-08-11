from typing import Any, Dict, List

import pytest

from splitter_mr.embedding.base_embedding import BaseEmbedding

# ---- Fixtures & Helpers ---------------------------------------------------------


class _DummyClient:
    """Minimal client that records the last call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _GoodEmbedding(BaseEmbedding):
    """Concrete implementation used only for testing BaseEmbedding’s contract."""

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name is required")
        self.model_name = model_name
        self._client = _DummyClient()

    def get_client(self) -> _DummyClient:
        return self._client

    def embed_text(self, text: str, **parameters: Dict[str, Any]) -> List[float]:
        if text is None or text == "":
            raise ValueError("`text` must be a non-empty string.")
        # Simulate a backend failure if requested
        if parameters.get("raise_runtime_error"):
            raise RuntimeError("Backend failed")
        # Simulate unexpected shape if requested
        if parameters.get("force_bad_shape"):
            # In a real impl you'd detect this and raise; we mirror that here:
            raise RuntimeError("Unexpected response shape")

        # Record the call (to verify parameter forwarding)
        self._client.record(text=text, parameters=parameters)

        # Return a deterministic embedding-like vector for testing
        # Example: basic character-code features
        length = len(text)
        ascii_sum = sum(ord(c) for c in text)
        ascii_avg = ascii_sum / length
        return [float(length), float(ascii_sum), float(ascii_avg)]


class _MissingEmbedMethod(BaseEmbedding):
    """Negative test: subclass that does NOT implement embed_text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client = _DummyClient()

    def get_client(self) -> _DummyClient:
        return self._client


# ---- Tests -------------------------------------------------------------------


def test_base_embedding_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEmbedding("any-model")  # type: ignore[abstract]


def test_subclass_missing_abstract_method_cannot_instantiate():
    with pytest.raises(TypeError):
        _MissingEmbedMethod("model-x")  # embed_text not implemented


def test_good_embedding_initializes_and_exposes_client():
    emb = _GoodEmbedding("model-ok")
    client = emb.get_client()
    assert client is not None
    assert isinstance(client, _DummyClient)


@pytest.mark.parametrize("bad_text", ["", None])  # type: ignore[list-item]
def test_embed_text_raises_value_error_on_empty_or_none(bad_text):
    emb = _GoodEmbedding("model-ok")
    with pytest.raises(ValueError):
        emb.embed_text(bad_text)  # type: ignore[arg-type]


def test_embed_text_returns_float_vector_and_is_deterministic():
    emb = _GoodEmbedding("model-ok")
    v1 = emb.embed_text("hello world")
    v2 = emb.embed_text("hello world")
    assert isinstance(v1, list)
    assert all(isinstance(x, float) for x in v1)
    assert len(v1) > 0
    assert v1 == v2  # deterministic for the same input in this test double


def test_embed_text_parameter_forwarding_is_recorded():
    emb = _GoodEmbedding("model-ok")
    params = {"user": "unit-test", "trace_id": "abc123"}
    _ = emb.embed_text("ping", **params)
    # Verify our dummy client saw the parameters
    assert emb.get_client().calls, "Client did not record any calls"
    last = emb.get_client().calls[-1]
    assert last["parameters"]["user"] == "unit-test"
    assert last["parameters"]["trace_id"] == "abc123"
    assert last["text"] == "ping"


def test_embed_text_raises_runtime_error_on_backend_failure():
    emb = _GoodEmbedding("model-ok")
    with pytest.raises(RuntimeError):
        emb.embed_text("anything", raise_runtime_error=True)


def test_embed_text_raises_runtime_error_on_unexpected_shape():
    emb = _GoodEmbedding("model-ok")
    with pytest.raises(RuntimeError):
        emb.embed_text("anything", force_bad_shape=True)


def test_constructor_validates_model_name():
    with pytest.raises(ValueError):
        _GoodEmbedding("")  # must provide a model name
