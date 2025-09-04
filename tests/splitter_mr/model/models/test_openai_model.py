from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model.models.openai_model import OpenAIVisionModel
from splitter_mr.schema import DEFAULT_IMAGE_CAPTION_PROMPT

# -------- Helpers & Fixtures -------- #


@pytest.fixture
def openai_vision_model():
    with patch("splitter_mr.model.models.openai_model.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        model = OpenAIVisionModel(api_key="sk-test", model_name="gpt-4o")
        return model


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# -------- Test cases --------- #


def _mock_create_returning(text="Extracted text!"):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=text))]
    return mock_response


def test_analyze_content_calls_api(openai_vision_model):
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = _mock_create_returning("Extracted text!")
        text = openai_vision_model.analyze_content("SOME_BASE64", prompt="What's here?")
        mock_create.assert_called_once()
        args = mock_create.call_args.kwargs
        assert args["model"] == "gpt-4o"
        assert args["messages"][0]["content"][0]["text"] == "What's here?"
        assert text == "Extracted text!"


def test_init_with_argument():
    with patch("splitter_mr.model.models.openai_model.OpenAI") as mock_openai:
        model = OpenAIVisionModel(api_key="my-secret", model_name="gpt-4o")
        mock_openai.assert_called_once_with(api_key="my-secret")
        assert model.model_name == "gpt-4o"


def test_init_with_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with patch("splitter_mr.model.models.openai_model.OpenAI") as mock_openai:
        _ = OpenAIVisionModel()
        mock_openai.assert_called_once_with(api_key="env-key")


def test_init_missing_key_raises():
    with pytest.raises(ValueError, match="API key.*not set"):
        OpenAIVisionModel()


def test_analyze_content_custom_params():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_create_returning("foo")
    with patch(
        "splitter_mr.model.models.openai_model.OpenAI", return_value=mock_client
    ):
        model = OpenAIVisionModel(api_key="x", model_name="vision")
        out = model.analyze_content("dGVzdA==", prompt="Extract!", temperature=0.2)
        called = mock_client.chat.completions.create.call_args.kwargs
        assert called["model"] == "vision"
        assert called["messages"][0]["content"][0]["text"] == "Extract!"
        assert called["temperature"] == 0.2
        assert out == "foo"


def test_analyze_content_uses_default_prompt_when_omitted(openai_vision_model):
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = _mock_create_returning("ok")
        _ = openai_vision_model.analyze_content("AAAA")
        called = mock_create.call_args.kwargs
        text_part = called["messages"][0]["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == DEFAULT_IMAGE_CAPTION_PROMPT


def test_analyze_content_includes_image_url_block_png(openai_vision_model):
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        mock_create.return_value = _mock_create_returning("ok")
        _ = openai_vision_model.analyze_content("Zm9vYmFy", prompt="go")
        called = mock_create.call_args.kwargs
        content = called["messages"][0]["content"]
        kinds = [c["type"] for c in content]
        assert "text" in kinds and "image_url" in kinds
        img = next(c for c in content if c["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("ext", ["tiff", "bmp", "svg", "heic"])
def test_analyze_content_raises_on_unsupported_mime(openai_vision_model, ext):
    # If the extension maps to an unsupported MIME type, we should error out
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        with pytest.raises(ValueError):
            openai_vision_model.analyze_content("BASE64DATA", file_ext=ext)

        # Make sure we never hit the network call
        mock_create.assert_not_called()


def test_analyze_content_accepts_jpg_and_normalizes_to_jpeg(openai_vision_model):
    # jpg should resolve to image/jpeg and proceed without error
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_create.return_value = mock_response

        _ = openai_vision_model.analyze_content("AAAA", file_ext="jpg")

        mock_create.assert_called_once()
        called = mock_create.call_args.kwargs
        content = called["messages"][0]["content"]
        img = next(part for part in content if part["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
