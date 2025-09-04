import base64
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model.models.gemini_model import GeminiVisionModel

# ----------------- Fixtures/Helpers ------------------


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_types():
    part = MagicMock()
    part.from_bytes = MagicMock(return_value="IMAGE_PART")
    types = MagicMock()
    types.Part = part
    return types


@pytest.fixture
def mock_sdk(monkeypatch, mock_client, mock_types):
    # Patch google.genai.Client and google.genai.types
    with patch(
        "splitter_mr.model.models.gemini_model.genai.Client", return_value=mock_client
    ):
        with patch("splitter_mr.model.models.gemini_model.types", mock_types):
            yield mock_client, mock_types


# ----------------- __init__ -------------------------


def test_init_with_api_key(mock_sdk):
    mock_client, _ = mock_sdk
    model = GeminiVisionModel(api_key="KEY123")
    assert model.api_key == "KEY123"
    assert model.client is mock_client
    assert model.model == mock_client.models


def test_init_with_env(monkeypatch, mock_sdk):
    mock_client, _ = mock_sdk
    monkeypatch.setenv("GEMINI_API_KEY", "ENVKEY")
    model = GeminiVisionModel()
    assert model.api_key == "ENVKEY"
    assert model.client is mock_client


def test_init_missing_key():
    with pytest.raises(ValueError):
        GeminiVisionModel(api_key=None)


# ----------------- get_client -----------------------


def test_get_client_returns_sdk_client(mock_sdk):
    mock_client, _ = mock_sdk
    model = GeminiVisionModel(api_key="KEY")
    assert model.get_client() is mock_client


# ----------------- analyze_content (happy path) -----


def test_analyze_content_success(mock_sdk):
    mock_client, mock_types = mock_sdk
    # Patch model.generate_content to return a mock response with text attr
    fake_response = MagicMock()
    fake_response.text = "Fake Result"
    mock_models = MagicMock()
    mock_models.generate_content.return_value = fake_response
    mock_client.models = mock_models

    # Compose test image bytes (simulate base64 encoding of image bytes)
    image_bytes = b"img_data"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    model = GeminiVisionModel(api_key="KEY")
    model.model = mock_models  # replace for .generate_content call

    result = model.analyze_content(
        prompt="Describe this image",
        file=image_b64.encode("utf-8"),
        file_ext="png",
        temperature=0.5,
    )

    # Ensure correct decode, Part usage, and SDK call
    mock_types.Part.from_bytes.assert_called_once_with(
        data=image_bytes, mime_type="image/png"
    )
    mock_models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents=["IMAGE_PART", "Describe this image"],
        temperature=0.5,
    )
    assert result == "Fake Result"


# ----------------- analyze_content: input errors ----


def test_analyze_content_no_file(mock_sdk):
    model = GeminiVisionModel(api_key="KEY")
    with pytest.raises(ValueError) as e:
        model.analyze_content(prompt="Test", file=None)
    assert "No image file provided" in str(e.value)


@pytest.mark.parametrize("bad_b64", ["badbase64", b"!!!notbase64!!!"])
def test_analyze_content_bad_base64(mock_sdk, bad_b64):
    model = GeminiVisionModel(api_key="KEY")
    with pytest.raises(ValueError) as e:
        model.analyze_content(prompt="x", file=bad_b64)
    assert "Failed to decode base64 image data" in str(e.value)


# ----------------- analyze_content: SDK/response errors ----


def test_analyze_content_sdk_failure(mock_sdk):
    mock_client, mock_types = mock_sdk
    mock_models = MagicMock()
    mock_models.generate_content.side_effect = RuntimeError("fail!")
    mock_client.models = mock_models
    image_b64 = base64.b64encode(b"img").decode("utf-8")
    model = GeminiVisionModel(api_key="KEY")
    model.model = mock_models

    with pytest.raises(RuntimeError) as e:
        model.analyze_content(prompt="what?", file=image_b64.encode("utf-8"))
    assert "Gemini model inference failed" in str(e.value)


# ----------------- analyze_content: mime_type fallback ----


def test_analyze_content_mime_type_fallback(mock_sdk):
    mock_client, mock_types = mock_sdk
    mock_models = MagicMock()
    mock_models.generate_content.return_value = MagicMock(text="ok")
    mock_client.models = mock_models

    image_bytes = b"x"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    model = GeminiVisionModel(api_key="KEY")
    model.model = mock_models

    # Use unknown extension to trigger default mime type
    result = model.analyze_content(
        prompt="x", file=image_b64.encode("utf-8"), file_ext="unknown"
    )
    mock_types.Part.from_bytes.assert_called_once_with(
        data=image_bytes, mime_type="image/jpeg"
    )
    assert result == "ok"


# ----------------- analyze_content: accepts string file ----


def test_analyze_content_accepts_str_file(mock_sdk):
    mock_client, mock_types = mock_sdk
    mock_models = MagicMock()
    mock_models.generate_content.return_value = MagicMock(text="works")
    mock_client.models = mock_models

    image_bytes = b"y"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    model = GeminiVisionModel(api_key="KEY")
    model.model = mock_models

    # Pass the base64 string, not bytes
    result = model.analyze_content(prompt="y", file=image_b64)
    mock_types.Part.from_bytes.assert_called_once_with(
        data=image_bytes, mime_type="image/jpeg"
    )
    assert result == "works"
