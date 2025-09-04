import types as _types

import pytest as _pytest

import splitter_mr.embedding.embeddings as embedding


def _mk_module(**attrs):
    return _types.SimpleNamespace(**attrs)


def test_embedding___all___contains_expected_names():
    expected = {
        "AzureOpenAIEmbedding",
        "OpenAIEmbedding",
        "GeminiEmbedding",
        "HuggingFaceEmbedding",
        "AnthropicEmbedding",
    }
    assert set(embedding.__all__) == expected


def test_embedding___dir___returns_sorted_all():
    assert embedding.__dir__() == sorted(embedding.__all__)


@_pytest.mark.parametrize(
    "name, module_path, class_name",
    [
        ("AzureOpenAIEmbedding", ".azure_openai_embedding", "AzureOpenAIEmbedding"),
        ("OpenAIEmbedding", ".openai_embedding", "OpenAIEmbedding"),
        ("GeminiEmbedding", ".gemini_embedding", "GeminiEmbedding"),
        ("HuggingFaceEmbedding", ".huggingface_embedding", "HuggingFaceEmbedding"),
        ("AnthropicEmbedding", ".anthropic_embedding", "AnthropicEmbedding"),
    ],
)
def test_embedding_lazy_getattr_success(monkeypatch, name, module_path, class_name):
    captured = {}

    def fake_import(mod_path, package=None):
        captured["args"] = (mod_path, package)
        Dummy = type(class_name, (), {})
        return _mk_module(**{class_name: Dummy})

    monkeypatch.setattr(embedding.importlib, "import_module", fake_import)

    obj = getattr(embedding, name)
    assert isinstance(obj, type)
    assert obj.__name__ == class_name
    assert captured["args"][0] == module_path
    assert captured["args"][1] == embedding.__name__


@_pytest.mark.parametrize(
    "name",
    [
        "AzureOpenAIEmbedding",
        "OpenAIEmbedding",
        "GeminiEmbedding",
        "HuggingFaceEmbedding",
        "AnthropicEmbedding",
    ],
)
def test_embedding_lazy_getattr_missing_optional_dep_raises_multimodal_hint(
    monkeypatch, name
):
    def fake_import(*a, **kw):
        raise ModuleNotFoundError("simulated missing dependency")

    monkeypatch.setattr(embedding.importlib, "import_module", fake_import)

    with _pytest.raises(ModuleNotFoundError) as exc:
        getattr(embedding, name)

    msg = str(exc.value)
    assert "requires the 'multimodal' extra" in msg
    assert "pip install 'splitter-mr[multimodal]'" in msg


@_pytest.mark.parametrize(
    "name, class_name",
    [
        ("AzureOpenAIEmbedding", "AzureOpenAIEmbedding"),
        ("OpenAIEmbedding", "OpenAIEmbedding"),
        ("GeminiEmbedding", "GeminiEmbedding"),
        ("HuggingFaceEmbedding", "HuggingFaceEmbedding"),
        ("AnthropicEmbedding", "AnthropicEmbedding"),
    ],
)
def test_embedding_import_succeeds_but_class_missing_raises_attributeerror(
    monkeypatch, name, class_name
):
    def fake_import(*a, **kw):
        return _mk_module()

    monkeypatch.setattr(embedding.importlib, "import_module", fake_import)

    with _pytest.raises(AttributeError) as exc:
        getattr(embedding, name)

    msg = str(exc.value)
    assert "has no attribute" in msg
    assert class_name in msg


def test_embedding_unknown_name_raises_attributeerror():
    with _pytest.raises(AttributeError) as exc:
        embedding.DoesNotExist
    msg = str(exc.value)
    assert f"module {embedding.__name__!r} has no attribute 'DoesNotExist'" in msg


def test_embedding_repeated_access_triggers_import_each_time(monkeypatch):
    calls = {"n": 0}

    def fake_import(mod_path, package=None):
        calls["n"] += 1
        Dummy = type("OpenAIEmbedding", (), {"_marker": calls["n"]})
        return _mk_module(OpenAIEmbedding=Dummy)

    monkeypatch.setattr(embedding.importlib, "import_module", fake_import)

    c1 = embedding.OpenAIEmbedding
    c2 = embedding.OpenAIEmbedding
    assert calls["n"] == 2
    assert c1 is not c2
