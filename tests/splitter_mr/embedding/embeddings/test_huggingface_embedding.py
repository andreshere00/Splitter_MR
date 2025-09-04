import sys
import types

import numpy as np
import pytest

from splitter_mr.embedding.embeddings.huggingface_embedding import HuggingFaceEmbedding

# ---- Fixtures, helpers & dummies ---- #


class DummyTorchTensor:
    """A tiny torch-like tensor that supports .detach().cpu().float().tolist()."""

    __is_dummytensor__ = True

    def __init__(self, data):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def tolist(self):
        return self._data


class DummyTorch:
    """A torch-like shim so code paths guarded by `torch is not None` run."""

    @staticmethod
    def is_tensor(x):
        return getattr(x, "__is_dummytensor__", False)


class DummySentenceTransformer:
    """
    Minimal shim for sentence_transformers.SentenceTransformer:
    - Captures init args
    - Provides get_max_seq_length/tokenize
    - encode(...) behavior is monkeypatched per test
    """

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self._device = device
        self._max_len = 8  # default, can monkeypatch per test
        self._last_kwargs = None  # tests can inspect what was forwarded to encode
        # default encode result (tests usually monkeypatch this method anyway)
        self._default_dim = 4

    def get_max_seq_length(self):
        return self._max_len

    def tokenize(self, texts):
        # texts is expected to be a list[str]
        # Use a naive word count as token count: len(text.split())
        lens = [len(t.split()) for t in texts]
        # Return the same shapes sentence-transformers typically returns
        # For simplicity, return list of lists with dummy token ids
        return {"input_ids": [[1] * L for L in lens]}

    # The encode method gets monkeypatched in tests to return different shapes.
    def encode(self, inputs, **kwargs):
        self._last_kwargs = kwargs
        if isinstance(inputs, str):
            return np.ones(self._default_dim, dtype=np.float32)
        # list[str]
        return np.ones((len(inputs), self._default_dim), dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_sentence_transformers(monkeypatch):
    """
    Provide a dummy `sentence_transformers` module so the production code's
    `from sentence_transformers import SentenceTransformer` inside __init__
    imports our dummy class.
    """
    dummy_mod = types.ModuleType("sentence_transformers")
    dummy_mod.SentenceTransformer = DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_mod)


# ---- Test cases ---- #


def test_init_success_and_get_client():
    emb = HuggingFaceEmbedding(model_name="dummy/model", device="mps")
    assert emb.model_name == "dummy/model"
    # The model instance is our DummySentenceTransformer (imported at runtime)
    assert emb.get_client().__class__.__name__ == "DummySentenceTransformer"
    # Device string is passed through to the SentenceTransformer dummy
    assert emb.get_client()._device == "mps"


def test_init_failure(monkeypatch):
    class RaisingST:
        def __init__(self, *a, **k):
            raise Exception("boom")

    # Patch the import target (the dummy `sentence_transformers` module)
    monkeypatch.setattr(
        sys.modules["sentence_transformers"],
        "SentenceTransformer",
        RaisingST,
        raising=True,
    )
    with pytest.raises(ValueError, match="Failed to load SentenceTransformer"):
        HuggingFaceEmbedding("anything")


def test_embed_text_numpy_1d(monkeypatch):
    emb = HuggingFaceEmbedding()

    # Return a 1D numpy vector
    def fake_encode(text, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)

    out = emb.embed_text(
        "hello", convert_to_tensor=True, batch_size=3
    )  # convert_to_tensor should be forced False
    assert isinstance(out, list) and all(isinstance(x, float) for x in out)
    # Ensure request kwargs were set/overridden
    assert emb.model._last_kwargs["convert_to_tensor"] is False
    # normalize defaults True unless overridden
    assert emb.model._last_kwargs["normalize_embeddings"] is True
    assert emb.model._last_kwargs["batch_size"] == 3


def test_embed_text_override_normalize(monkeypatch):
    emb = HuggingFaceEmbedding(normalize=False)

    def fake_encode(text, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.array([1, 2, 3], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    _ = emb.embed_text("hi", normalize_embeddings=True)  # override default False
    assert emb.model._last_kwargs["normalize_embeddings"] is True


def test_embed_text_torch_like(monkeypatch):
    emb = HuggingFaceEmbedding()

    # Return a torch-like 1D vector
    def fake_encode(text, **kwargs):
        return DummyTorchTensor([0.5, 0.6, 0.7])

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_text("hello")
    assert out == [0.5, 0.6, 0.7]


def test_embed_text_tuple_output(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(text, **kwargs):
        return (1, 2.0, 3)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_text("ok")
    assert out == [1.0, 2.0, 3.0]


def test_embed_text_unexpected_output_raises(monkeypatch):
    emb = HuggingFaceEmbedding()

    class Weird:
        pass

    def fake_encode(text, **kwargs):
        return Weird()

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    with pytest.raises(RuntimeError, match="Unexpected embedding output type"):
        emb.embed_text("weird")


def test_embed_text_empty_raises():
    emb = HuggingFaceEmbedding()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        emb.embed_text("")


def test_embed_documents_numpy_2d(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(texts, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)

    out = emb.embed_documents(
        ["a", "b"], convert_to_tensor=True, show_progress_bar=False
    )
    assert out == [[0.0, 1.0], [2.0, 3.0]]
    assert emb.model._last_kwargs["convert_to_tensor"] is False
    assert emb.model._last_kwargs["normalize_embeddings"] is True
    assert emb.model._last_kwargs["show_progress_bar"] is False


def test_embed_documents_list_of_lists(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(texts, **kwargs):
        return [[1, 2], [3.5, 4]]

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_documents(["x", "y"])
    assert out == [[1.0, 2.0], [3.5, 4.0]]


def test_embed_documents_flat_list_single_vector(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(texts, **kwargs):
        # Some ST versions return a flat list when input has length 1
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_documents(["single"])
    assert out == [[0.1, 0.2, 0.3]]


def test_embed_documents_torch_like(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(texts, **kwargs):
        return DummyTorchTensor([[1, 2], [3, 4]])

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_documents(["a", "b"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_documents_input_validation():
    emb = HuggingFaceEmbedding()
    with pytest.raises(ValueError, match="must be a non-empty list"):
        emb.embed_documents([])

    with pytest.raises(ValueError, match="non-empty strings"):
        emb.embed_documents(["ok", ""])


def test_enforce_max_length_single_ok(monkeypatch):
    emb = HuggingFaceEmbedding(enforce_max_length=True)

    # Max len = 8 (Dummy default). Text with <= 8 tokens should pass.
    def fake_encode(text, **kwargs):
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_text("one two three four")  # 4 tokens
    assert isinstance(out, list)


def test_enforce_max_length_single_too_long():
    emb = HuggingFaceEmbedding(enforce_max_length=True)
    # 9 tokens > default 8
    long_text = "a b c d e f g h i"
    with pytest.raises(ValueError, match="exceeds model max sequence length"):
        emb.embed_text(long_text)


def test_enforce_max_length_batch_too_long():
    emb = HuggingFaceEmbedding(enforce_max_length=True)
    texts = ["one two three", "a b c d e f g h i"]  # second is 9 tokens

    with pytest.raises(ValueError, match="exceeds model max sequence length"):
        emb.embed_documents(texts)


def test_length_check_fallback_to_attribute(monkeypatch):
    # Make get_max_seq_length raise, but set attribute max_seq_length = 5
    emb = HuggingFaceEmbedding(enforce_max_length=True)
    monkeypatch.setattr(
        emb.model,
        "get_max_seq_length",
        lambda: (_ for _ in ()).throw(Exception("no attr")),
        raising=True,
    )
    monkeypatch.setattr(emb.model, "max_seq_length", 5, raising=False)

    # 6 tokens should violate 5
    with pytest.raises(ValueError, match="exceeds model max sequence length"):
        emb.embed_text("a b c d e f")


def test_length_check_skips_when_unknown(monkeypatch):
    # Force both token count and max_seq_length lookup to be unknown
    emb = HuggingFaceEmbedding(enforce_max_length=True)

    def no_tokenize(_):
        raise Exception("no tokenize")

    monkeypatch.setattr(emb.model, "tokenize", no_tokenize, raising=True)
    monkeypatch.setattr(emb.model, "get_max_seq_length", lambda: None, raising=True)
    if hasattr(emb.model, "max_seq_length"):
        monkeypatch.delattr(emb.model, "max_seq_length", raising=False)

    # Should not raise since it cannot establish a hard limit
    def fake_encode(text, **kwargs):
        return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_text("this can be arbitrarily long without checks")
    assert out == [1.0, 2.0]


def test_embed_text_string_numbers_coerced(monkeypatch):
    emb = HuggingFaceEmbedding()

    class FakeVec:
        def __iter__(self):
            return iter(["1.1", "2.2", "3.3"])

    def fake_encode(text, **kwargs):
        return FakeVec()

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_text("ok")
    assert out == [1.1, 2.2, 3.3]


def test_embed_documents_returns_flat_list_if_output_is_flat(monkeypatch):
    emb = HuggingFaceEmbedding()

    # Simulate model.encode returning [1.5, 2.5, 3.5] for a single input string
    def fake_encode(texts, **kwargs):
        return [1.5, 2.5, 3.5]

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_documents(["foo"])
    assert out == [[1.5, 2.5, 3.5]]


def test_embed_documents_raises_on_weird_output(monkeypatch):
    emb = HuggingFaceEmbedding()

    class Weird:
        pass

    def fake_encode(texts, **kwargs):
        return Weird()

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    with pytest.raises(RuntimeError, match="Unexpected batch embedding output type"):
        emb.embed_documents(["foo", "bar"])


@pytest.mark.parametrize("bad", [None, 123, 1.1, {}, [], object()])
def test_embed_text_raises_on_non_str(bad):
    emb = HuggingFaceEmbedding()
    with pytest.raises(ValueError, match="must be a non-empty string"):
        emb.embed_text(bad)


def test_max_seq_length_returns_none_if_all_fails(monkeypatch):
    emb = HuggingFaceEmbedding()
    # Patch get_max_seq_length to raise and model has no max_seq_length attr
    monkeypatch.setattr(
        emb.model,
        "get_max_seq_length",
        lambda: (_ for _ in ()).throw(Exception("fail")),
        raising=True,
    )
    if hasattr(emb.model, "max_seq_length"):
        monkeypatch.delattr(emb.model, "max_seq_length", raising=False)
    assert emb._max_seq_length() is None


def test_count_tokens_returns_none_on_failure(monkeypatch):
    emb = HuggingFaceEmbedding()
    monkeypatch.setattr(
        emb.model,
        "tokenize",
        lambda _: (_ for _ in ()).throw(Exception("fail")),
        raising=True,
    )
    assert emb._count_tokens("irrelevant") is None


def test_embed_documents_explicit_normalize_embeddings(monkeypatch):
    emb = HuggingFaceEmbedding(normalize=True)  # default True

    def fake_encode(texts, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    out = emb.embed_documents(["a", "b"], normalize_embeddings=False)
    assert out == [[1.0, 1.0], [1.0, 1.0]]
    assert emb.model._last_kwargs["normalize_embeddings"] is False


def test_embed_text_always_sets_convert_to_tensor_false(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(text, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.array([9, 8, 7], dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    _ = emb.embed_text("hello", convert_to_tensor=True)
    assert emb.model._last_kwargs["convert_to_tensor"] is False


def test_embed_documents_forwards_all_parameters(monkeypatch):
    emb = HuggingFaceEmbedding()

    def fake_encode(texts, **kwargs):
        emb.model._last_kwargs = kwargs
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(emb.model, "encode", fake_encode, raising=True)
    _ = emb.embed_documents(["a", "b"], batch_size=22, show_progress_bar=False)
    assert emb.model._last_kwargs["batch_size"] == 22
    assert emb.model._last_kwargs["show_progress_bar"] is False
