import sys
import types

import pytest

import splitter_mr.model as model_pkg


def _make_dummy_models(**attrs):
    mod = types.SimpleNamespace(**attrs)
    mod.__name__ = "splitter_mr.model.models"
    return mod


def test_model___all___contains_expected_names():
    assert set(model_pkg.__all__) == {
        "BaseVisionModel",
        "AzureOpenAIVisionModel",
        "OpenAIVisionModel",
        "HuggingFaceVisionModel",
        "GrokVisionModel",
        "GeminiVisionModel",
        "AnthropicVisionModel",
    }


def test_model_base_class_is_exposed_without_lazy_import():
    # Ensure backing submodule is not present
    sys.modules.pop("splitter_mr.model.models", None)

    # Accessing BaseVisionModel should not require importing .models
    assert hasattr(model_pkg, "BaseVisionModel")
    _ = model_pkg.BaseVisionModel
    assert "splitter_mr.model.models" not in sys.modules


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
def test_model___getattr___delegates_to_models(monkeypatch, name):
    sentinel = object()
    dummy = _make_dummy_models(**{name: sentinel})
    monkeypatch.dict(sys.modules, {"splitter_mr.model.models": dummy}, clear=False)

    obj = getattr(model_pkg, name)
    assert obj is sentinel


def test_model___getattr___repeated_access_returns_same_object(monkeypatch):
    sentinel = object()
    dummy = _make_dummy_models(OpenAIVisionModel=sentinel)
    monkeypatch.dict(sys.modules, {"splitter_mr.model.models": dummy}, clear=False)

    a = model_pkg.OpenAIVisionModel
    b = model_pkg.OpenAIVisionModel
    assert a is b is sentinel


def test_model___getattr___unknown_name_raises_attributeerror():
    with pytest.raises(AttributeError) as exc:
        getattr(model_pkg, "DoesNotExist")
    assert f"module {model_pkg.__name__!r} has no attribute 'DoesNotExist'" in str(
        exc.value
    )


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
def test_model_import_succeeds_but_class_missing_raises_attributeerror(
    monkeypatch, name
):
    # Provide a models module that lacks the requested attribute
    dummy = _make_dummy_models()
    monkeypatch.dict(sys.modules, {"splitter_mr.model.models": dummy}, clear=False)

    with pytest.raises(AttributeError) as exc:
        getattr(model_pkg, name)

    msg = str(exc.value)
    # getattr(SimpleNamespace, name) raises AttributeError mentioning "has no attribute"
    assert "has no attribute" in msg
    assert name in msg
