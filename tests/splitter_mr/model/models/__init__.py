import types

import pytest

import splitter_mr.model.models as vision


def _make_module(**attrs):
    return types.SimpleNamespace(**attrs)


def test___all___contains_expected_names():
    expected = {
        "AzureOpenAIVisionModel",
        "OpenAIVisionModel",
        "HuggingFaceVisionModel",
        "GrokVisionModel",
        "GeminiVisionModel",
        "AnthropicVisionModel",
    }
    assert set(vision.__all__) == expected


def test___dir___returns_sorted_all():
    assert vision.__dir__() == sorted(vision.__all__)


@pytest.mark.parametrize(
    "name, module_path, class_name",
    [
        ("AzureOpenAIVisionModel", ".azure_openai_model", "AzureOpenAIVisionModel"),
        ("OpenAIVisionModel", ".openai_model", "OpenAIVisionModel"),
        ("HuggingFaceVisionModel", ".huggingface_model", "HuggingFaceVisionModel"),
        ("GrokVisionModel", ".grok_model", "GrokVisionModel"),
        ("GeminiVisionModel", ".gemini_model", "GeminiVisionModel"),
        ("AnthropicVisionModel", ".anthropic_model", "AnthropicVisionModel"),
    ],
)
def test_lazy_getattr_success(monkeypatch, name, module_path, class_name):
    captured = {}

    def fake_import(mod_path, package=None):
        captured["args"] = (mod_path, package)
        Dummy = type(class_name, (), {})
        return _make_module(**{class_name: Dummy})

    monkeypatch.setattr(vision.importlib, "import_module", fake_import)

    obj = getattr(vision, name)
    assert isinstance(obj, type)
    assert obj.__name__ == class_name
    assert captured["args"][0] == module_path
    assert captured["args"][1] == vision.__name__


@pytest.mark.parametrize(
    "name",
    [
        "AzureOpenAIVisionModel",
        "OpenAIVisionModel",
        "HuggingFaceVisionModel",
        "GrokVisionModel",
        "GeminiVisionModel",
        "AnthropicVisionModel",
    ],
)
def test_lazy_getattr_missing_optional_dep_raises_multimodal_hint(monkeypatch, name):
    def fake_import(*a, **kw):
        raise ModuleNotFoundError("simulated missing dependency")

    monkeypatch.setattr(vision.importlib, "import_module", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc:
        getattr(vision, name)

    msg = str(exc.value)
    assert "requires the 'multimodal' extra" in msg
    assert "pip install 'splitter-mr[multimodal]'" in msg


@pytest.mark.parametrize(
    "name, class_name",
    [
        ("AzureOpenAIVisionModel", "AzureOpenAIVisionModel"),
        ("OpenAIVisionModel", "OpenAIVisionModel"),
        ("HuggingFaceVisionModel", "HuggingFaceVisionModel"),
        ("GrokVisionModel", "GrokVisionModel"),
        ("GeminiVisionModel", "GeminiVisionModel"),
        ("AnthropicVisionModel", "AnthropicVisionModel"),
    ],
)
def test_lazy_getattr_import_succeeds_but_class_missing_raises_attributeerror(
    monkeypatch, name, class_name
):
    def fake_import(*a, **kw):
        return _make_module()  # no class inside

    monkeypatch.setattr(vision.importlib, "import_module", fake_import)

    with pytest.raises(AttributeError) as exc:
        getattr(vision, name)

    msg = str(exc.value)
    assert "has no attribute" in msg
    assert class_name in msg


def test_lazy_getattr_unknown_name_raises_attributeerror():
    with pytest.raises(AttributeError) as exc:
        vision.DoesNotExist
    msg = str(exc.value)
    assert f"module {vision.__name__!r} has no attribute 'DoesNotExist'" in msg


def test_repeated_access_triggers_import_each_time(monkeypatch):
    calls = {"n": 0}

    def fake_import(mod_path, package=None):
        calls["n"] += 1
        Dummy = type("OpenAIVisionModel", (), {"_marker": calls["n"]})
        return _make_module(OpenAIVisionModel=Dummy)

    monkeypatch.setattr(vision.importlib, "import_module", fake_import)

    c1 = vision.OpenAIVisionModel
    c2 = vision.OpenAIVisionModel
    assert calls["n"] == 2
    assert c1 is not c2
