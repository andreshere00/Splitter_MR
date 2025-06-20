import os
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model.models.openai_model import OpenAIVisionModel

SAMPLE_IMAGE_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xA"
    "AfEAACAQUBAQEAAAAAAAAAAAAAAQIDBAUGERIhIjFBUf/EABQBAQAAAAAAAAAAAAAAAAAAA"
    "AP/xAAZEQEAAwEBAAAAAAAAAAAAAAAAAQIRAAP/2gAMAwEAAhEDEQA/AJpiwH//Z"
)

# Helpers


@pytest.fixture
def openai_vision_model():
    return OpenAIVisionModel(api_key="sk-test")


# Test cases


def test_init_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model = OpenAIVisionModel()
    assert model.get_client()


def test_extract_text_calls_api(openai_vision_model):
    with patch.object(
        openai_vision_model.client.responses, "create", autospec=True
    ) as mock_create:
        # Mock return value
        mock_create.return_value.output = [
            MagicMock(content=[MagicMock(text="Extracted text!")])
        ]
        text = openai_vision_model.extract_text(SAMPLE_IMAGE_B64, prompt="What's here?")
        assert text == "Extracted text!"
        mock_create.assert_called_once()
        payload = mock_create.call_args[1]["input"][0]
        assert payload["content"][0]["text"] == "What's here?"
