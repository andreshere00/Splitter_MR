import os
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model import AzureOpenAIVisionModel


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear env vars before each test
    for var in [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_init_with_arguments():
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI"
    ) as mock_client:
        model = AzureOpenAIVisionModel(
            api_key="key",
            azure_endpoint="https://endpoint",
            azure_deployment="deployment",
            api_version="2025-04-14-preview",
        )
        mock_client.assert_called_once_with(
            api_key="key",
            azure_endpoint="https://endpoint",
            azure_deployment="deployment",
            api_version="2025-04-14-preview",
        )
        assert model.model_name == "deployment"


def test_init_with_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-endpoint")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI"
    ) as mock_client:
        model = AzureOpenAIVisionModel()
        mock_client.assert_called_once()
        assert model.model_name == "env-deployment"


@pytest.mark.parametrize(
    "missing_env,errmsg",
    [
        ("AZURE_OPENAI_API_KEY", "API key"),
        ("AZURE_OPENAI_ENDPOINT", "endpoint"),
        ("AZURE_OPENAI_DEPLOYMENT", "deployment name"),
    ],
)
def test_init_env_missing(monkeypatch, missing_env, errmsg):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "x")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "x")
    monkeypatch.delenv(missing_env, raising=False)
    with pytest.raises(ValueError) as exc:
        AzureOpenAIVisionModel()
    assert errmsg in str(exc.value)


def test_extract_text_makes_correct_call():
    mock_client = MagicMock()
    # The ._azure_deployment attribute should be present
    mock_client._azure_deployment = "deployment"
    # Mock client.chat.completions.create to return desired value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Extracted text"))]
    mock_client.chat.completions.create.return_value = mock_response

    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI",
        return_value=mock_client,
    ):
        model = AzureOpenAIVisionModel(
            api_key="key",
            azure_endpoint="endpoint",
            azure_deployment="deployment",
            api_version="ver",
        )
        # Provide dummy base64 bytes for 'file'
        text = model.extract_text("dGVzdF9pbWFnZQ==", prompt="Extract!")
        # Check if the correct payload was sent
        mock_client.chat.completions.create.assert_called_once()
        called_args = mock_client.chat.completions.create.call_args[1]
        assert called_args["model"] == "deployment"
        assert called_args["messages"][0]["content"][0]["text"] == "Extract!"
        assert text == "Extracted text"
