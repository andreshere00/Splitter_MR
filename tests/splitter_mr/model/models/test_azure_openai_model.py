import os
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model.models.azure_openai_model import AzureOpenAIVisionModel

SAMPLE_IMAGE_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xA"
    "AfEAACAQUBAQEAAAAAAAAAAAAAAQIDBAUGERIhIjFBUf/EABQBAQAAAAAAAAAAAAAAAAAAA"
    "AP/xAAZEQEAAwEBAAAAAAAAAAAAAAAAAQIRAAP/2gAMAwEAAhEDEQA/AJpiwH//Z"
)


# Helpers


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")


@pytest.fixture
def azure_vision_model(azure_env):
    return AzureOpenAIVisionModel()


def test_azure_init_env(azure_env):
    model = AzureOpenAIVisionModel()
    assert model.get_model_name() == "gpt-4.1"
    assert model.get_client()


# Test cases


def test_azure_extract_text_calls_api(azure_vision_model):
    with patch.object(
        azure_vision_model.client.responses, "create", autospec=True
    ) as mock_create:
        mock_create.return_value.output = [
            MagicMock(content=[MagicMock(text="Azure Extracted text!")])
        ]
        text = azure_vision_model.extract_text(SAMPLE_IMAGE_B64, prompt="What's here?")
        assert text == "Azure Extracted text!"
        mock_create.assert_called_once()


def test_azure_analyze_resource_calls_api(azure_vision_model):
    with patch.object(
        azure_vision_model.client.responses, "create", autospec=True
    ) as mock_create:
        mock_create.return_value.output = [
            MagicMock(content=[MagicMock(text="Azure Analysis result.")])
        ]
        result = azure_vision_model.analyze_resource(
            SAMPLE_IMAGE_B64, context="Report", prompt="Describe this image."
        )
        assert result == "Azure Analysis result."
        mock_create.assert_called_once()
