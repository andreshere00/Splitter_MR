import base64
import os
from unittest.mock import MagicMock

import pytest

from splitter_mr.model.models.openrouter_model import OpenRouterVisionModel


@pytest.fixture
def fake_b64_png():
    return base64.b64encode(b"\x89PNG\r\n\x1a\n....").decode("utf-8")


@pytest.fixture
def api_key():
    return "test-OPENROUTER-KEY"


@pytest.fixture
def model_name():
    return "claude-haiku-4-5"


def _patch_openai_client(monkeypatch):
    mock_client = MagicMock()
    captured: dict = {}

    def factory(*args, **kwargs):
        captured["kwargs"] = kwargs
        return mock_client

    monkeypatch.setattr(
        "splitter_mr.model.models.openrouter_model.OpenAI",
        factory,
        raising=True,
    )
    return mock_client, captured


def _make_mock_response(text: str):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = text
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def test_init_with_api_key_sets_base_url(monkeypatch, api_key, model_name):
    _, captured = _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key, model_name=model_name)
    assert model.model_name == model_name
    assert captured["kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


def test_init_env(monkeypatch, api_key):
    _patch_openai_client(monkeypatch)
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        model = OpenRouterVisionModel(api_key=None)
        assert model.client is not None
    finally:
        del os.environ["OPENROUTER_API_KEY"]


def test_init_no_key_raises(monkeypatch):
    _patch_openai_client(monkeypatch)
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]
    with pytest.raises(ValueError):
        OpenRouterVisionModel(api_key=None)


def test_init_attribution_headers(monkeypatch, api_key):
    _, captured = _patch_openai_client(monkeypatch)
    OpenRouterVisionModel(
        api_key=api_key,
        site_url="https://example.com",
        app_name="Splitter MR",
    )
    headers = captured["kwargs"]["default_headers"]
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-OpenRouter-Title"] == "Splitter MR"


def test_get_client(monkeypatch, api_key):
    mock_client, _ = _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key)
    assert model.get_client() is mock_client


def test_analyze_content_success(monkeypatch, fake_b64_png, api_key, model_name):
    mock_client, _ = _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key, model_name=model_name)
    mock_response = _make_mock_response("Visible text from image")
    mock_client.chat.completions.create.return_value = mock_response

    result = model.analyze_content(
        prompt="Read all visible text.",
        file=fake_b64_png,
        file_ext="png",
    )
    assert result == "Visible text from image"

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == model_name
    user_msg = call_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    img_block = next(c for c in user_msg["content"] if c["type"] == "image_url")
    assert img_block["image_url"]["url"].startswith("data:image/")


def test_analyze_content_invalid_file(monkeypatch, api_key):
    _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key)
    with pytest.raises(ValueError):
        model.analyze_content(prompt="Describe image", file=None)


def test_analyze_content_unsupported_mime(monkeypatch, fake_b64_png, api_key):
    _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key)
    with pytest.raises(ValueError):
        model.analyze_content(
            prompt="What do you see?", file=fake_b64_png, file_ext="tiff"
        )


def test_analyze_content_runtime_error(monkeypatch, fake_b64_png, api_key):
    mock_client, _ = _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key)
    bad_response = MagicMock()
    bad_response.choices = []
    mock_client.chat.completions.create.return_value = bad_response

    with pytest.raises(RuntimeError):
        model.analyze_content(prompt="Read text", file=fake_b64_png)


def test_analyze_content_extra_parameters(monkeypatch, fake_b64_png, api_key):
    mock_client, _ = _patch_openai_client(monkeypatch)
    model = OpenRouterVisionModel(api_key=api_key)
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
