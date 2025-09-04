import sys as _sys
import types as _types

import pytest as _pytest

import splitter_mr.embedding.embeddings as embedding_pkg


def _make_dummy_embeddings(**attrs):
    mod = _types.SimpleNamespace(**attrs)
    mod.__name__ = "splitter_mr.model.embedding.embeddings"
    return mod


def test_embedding___all___contains_expected_names():
    assert set(embedding_pkg.__all__) == {
        "BaseEmbedding",
        "AzureOpenAIEmbedding",
        "OpenAIEmbedding",
        "HuggingFaceEmbedding",
        "GeminiEmbedding",
        "AnthropicEmbedding",
    }


def test_embedding___dir___returns_sorted_all():
    assert embedding_pkg.__dir__() == sorted(embedding_pkg.__all__)


def test_embedding_base_class_is_exposed_without_lazy_import():
    _sys.modules.pop("splitter_mr.model.embedding.embeddings", None)
    assert hasattr(embedding_pkg, "BaseEmbedding")
    _ = embedding_pkg.BaseEmbedding
    assert "splitter_mr.model.embedding.embeddings" not in _sys.modules


@_pytest.mark.parametrize(
    "name",
    [
        "AzureOpenAIEmbedding",
        "OpenAIEmbedding",
        "HuggingFaceEmbedding",
        "GeminiEmbedding",
        "AnthropicEmbedding",
    ],
)
def test_embedding___getattr___delegates_to_embeddings(monkeypatch, name):
    sentinel = object()
    dummy = _make_dummy_embeddings(**{name: sentinel})
    monkeypatch.dict(
        _sys.modules, {"splitter_mr.model.embedding.embeddings": dummy}, clear=False
    )

    obj = getattr(embedding_pkg, name)
    assert obj is sentinel


def test_embedding___getattr___repeated_access_returns_same_object(monkeypatch):
    sentinel = object()
    dummy = _make_dummy_embeddings(OpenAIEmbedding=sentinel)
    monkeypatch.dict(
        _sys.modules, {"splitter_mr.model.embedding.embeddings": dummy}, clear=False
    )

    a = embedding_pkg.OpenAIEmbedding
    b = embedding_pkg.OpenAIEmbedding
    assert a is b is sentinel


def test_embedding___getattr___unknown_name_raises_attributeerror():
    with _pytest.raises(AttributeError) as exc:
        getattr(embedding_pkg, "DoesNotExist")
    assert f"module {embedding_pkg.__name__!r} has no attribute 'DoesNotExist'" in str(
        exc.value
    )


@_pytest.mark.parametrize(
    "name",
    [
        "AzureOpenAIEmbedding",
        "OpenAIEmbedding",
        "HuggingFaceEmbedding",
        "GeminiEmbedding",
        "AnthropicEmbedding",
    ],
)
def test_embedding_import_succeeds_but_class_missing_raises_attributeerror(
    monkeypatch, name
):
    dummy = _make_dummy_embeddings()
    monkeypatch.dict(
        _sys.modules, {"splitter_mr.model.embedding.embeddings": dummy}, clear=False
    )

    with _pytest.raises(AttributeError) as exc:
        getattr(embedding_pkg, name)

    msg = str(exc.value)
    assert "has no attribute" in msg
    assert name in msg
