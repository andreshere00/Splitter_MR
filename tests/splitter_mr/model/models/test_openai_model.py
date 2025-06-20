from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model.models.openai_model import OpenAIVisionModel

# Helpers


@pytest.fixture
def openai_vision_model():
    with patch("splitter_mr.model.models.openai_model.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        model = OpenAIVisionModel(api_key="sk-test", model_name="gpt-4.1")
        return model


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# Test cases


def test_extract_text_calls_api(openai_vision_model):
    # Patch the correct method: chat.completions.create
    with patch.object(
        openai_vision_model.client.chat.completions, "create"
    ) as mock_create:
        # Setup mock return value
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Extracted text!"))
        ]
        mock_create.return_value = mock_response

        text = openai_vision_model.extract_text("SOME_BASE64", prompt="What's here?")
        mock_create.assert_called_once()
        args = mock_create.call_args[1]
        assert args["model"] == "gpt-4.1"
        assert args["messages"][0]["content"][0]["text"] == "What's here?"
        assert text == "Extracted text!"


def test_init_with_argument():
    with patch("splitter_mr.model.models.openai_model.OpenAI") as mock_openai:
        model = OpenAIVisionModel(api_key="my-secret", model_name="gpt-4.1")
        mock_openai.assert_called_once_with(api_key="my-secret")
        assert model.model_name == "gpt-4.1"


def test_init_with_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with patch("splitter_mr.model.models.openai_model.OpenAI") as mock_openai:
        model = OpenAIVisionModel()
        mock_openai.assert_called_once_with(api_key="env-key")


def test_init_missing_key_raises():
    with pytest.raises(ValueError, match="API key.*not set"):
        OpenAIVisionModel()


def test_extract_text_custom_params():
    # Setup mock client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="foo"))]
    mock_client.chat.completions.create.return_value = mock_response

    with patch(
        "splitter_mr.model.models.openai_model.OpenAI", return_value=mock_client
    ):
        model = OpenAIVisionModel(api_key="x", model_name="vision")
        out = model.extract_text("dGVzdA==", prompt="Extract!", temperature=0.2)
        mock_client.chat.completions.create.assert_called_once()
        args = mock_client.chat.completions.create.call_args[1]
        assert args["model"] == "vision"
        assert args["messages"][0]["content"][0]["text"] == "Extract!"
        assert args["temperature"] == 0.2
        assert out == "foo"
