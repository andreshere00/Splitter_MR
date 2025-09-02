import base64
import os
from unittest.mock import MagicMock

import pytest

from splitter_mr.model.models.anthropic_model import AnthropicVisionModel


@pytest.fixture
def fake_b64_png():
    # Minimal PNG header bytes, base64-encoded
    return base64.b64encode(b"\x89PNG\r\n\x1a\n....").decode("utf-8")


@pytest.fixture
def api_key():
    return "test-ANTHROPIC-KEY"


@pytest.fixture
def model_name():
    return "claude-3-opus-20240229"


def _patch_openai_client(monkeypatch):
    """
    Patch the OpenAI symbol used *inside the target module* so we don't
    instantiate the real client (and avoid base_url parsing issues).
    """
    mock_client = MagicMock()
    monkeypatch.setattr(
        "splitter_mr.model.models.anthropic_model.OpenAI",
        lambda *a, **k: mock_client,
        raising=True,
    )
    return mock_client


def _make_mock_response(text: str):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = text
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def test_init_with_api_key(monkeypatch, api_key, model_name):
    _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key, model_name=model_name)
    assert model.model_name == model_name
    assert hasattr(model, "client")


def test_init_env(monkeypatch, api_key):
    _patch_openai_client(monkeypatch)
    os.environ["ANTHROPIC_API_KEY"] = api_key
    try:
        model = AnthropicVisionModel(api_key=None)
        assert model.client is not None
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_init_no_key(monkeypatch):
    _patch_openai_client(monkeypatch)
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
    with pytest.raises(ValueError):
        AnthropicVisionModel(api_key=None)


def test_get_client(monkeypatch, api_key):
    mock_client = _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    assert model.get_client() is mock_client


def test_analyze_content_success(monkeypatch, fake_b64_png, api_key, model_name):
    mock_client = _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key, model_name=model_name)
    mock_response = _make_mock_response("Visible text from image")
    mock_client.chat.completions.create.return_value = mock_response

    result = model.analyze_content(
        prompt="Read all visible text.",
        file=fake_b64_png,
        file_ext="png",
    )
    assert result == "Visible text from image"

    # Validate payload structure
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == model_name
    assert "messages" in call_kwargs and isinstance(call_kwargs["messages"], list)
    user_msg = call_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    assert any(c["type"] == "text" for c in user_msg["content"])
    assert any(c["type"] == "image_url" for c in user_msg["content"])
    img_block = next(c for c in user_msg["content"] if c["type"] == "image_url")
    assert img_block["image_url"]["url"].startswith("data:image/")


def test_analyze_content_invalid_file(monkeypatch, api_key):
    _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    with pytest.raises(ValueError):
        model.analyze_content(prompt="Describe image", file=None)


def test_analyze_content_unsupported_mime(monkeypatch, fake_b64_png, api_key):
    _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    # Use a likely-unsupported extension
    with pytest.raises(ValueError):
        model.analyze_content(
            prompt="What do you see?", file=fake_b64_png, file_ext="tiff"
        )


def test_analyze_content_runtime_error(monkeypatch, fake_b64_png, api_key):
    mock_client = _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    # Simulate an unexpected/empty response shape
    bad_response = MagicMock()
    bad_response.choices = []
    mock_client.chat.completions.create.return_value = bad_response

    with pytest.raises(RuntimeError):
        model.analyze_content(prompt="Read text", file=fake_b64_png)


def test_analyze_content_api_exception(monkeypatch, fake_b64_png, api_key):
    mock_client = _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    # The class does not catch exceptions raised by `.create`; expect them to bubble.
    mock_client.chat.completions.create.side_effect = Exception("API Down")

    with pytest.raises(Exception):
        model.analyze_content(prompt="OCR", file=fake_b64_png)


def test_analyze_content_extra_parameters(monkeypatch, fake_b64_png, api_key):
    mock_client = _patch_openai_client(monkeypatch)
    model = AnthropicVisionModel(api_key=api_key)
    mock_response = _make_mock_response("Extra params handled")
    mock_client.chat.completions.create.return_value = mock_response

    result = model.analyze_content(
        prompt="Quick summary",
        file=fake_b64_png,
        file_ext="png",
        temperature=0.1,
        user="unittest",
    )
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["user"] == "unittest"
    assert result == "Extra params handled"
