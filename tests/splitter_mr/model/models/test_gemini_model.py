import base64
import sys
import types as pytypes
from unittest import mock

import pytest

from splitter_mr.model.models.gemini_model import GeminiVisionModel

# ---- Helpers, fixtures and mocks ---- #


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def mock_gemini_sdk(monkeypatch):
    """Creates a fake google.genai module and types.Part."""
    fake_genai = pytypes.SimpleNamespace()
    fake_types = pytypes.SimpleNamespace()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content(self, model, contents, **params):
            class FakeResponse:
                text = "This is the analyzed image text"

            # Store inputs for later assertion
            self.called_with = (model, contents, params)
            return FakeResponse()

    class FakePart:
        @classmethod
        def from_bytes(cls, data, mime_type):
            return ("FakePart", data, mime_type)

    fake_genai.Client = FakeClient
    fake_genai.configure = lambda api_key: None
    fake_types.Part = FakePart

    monkeypatch.setitem(
        sys.modules, "google", pytypes.SimpleNamespace(generativeai=fake_genai)
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    return fake_genai, fake_types


# ---- Test cases ---- #


def test_init_with_api_key(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="sk-gemini-key", model_name="gemini-2.5-flash")
    assert m.api_key == "sk-gemini-key"
    assert m.model_name == "gemini-2.5-flash"
    assert hasattr(m.client, "models")


def test_init_reads_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env-key")
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(model_name="gemini-2.5-pro")
    assert m.api_key == "from-env-key"
    assert m.model_name == "gemini-2.5-pro"


def test_init_missing_key(monkeypatch):
    # No env, no param: should raise ValueError
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    with pytest.raises(ValueError, match="Google Gemini API key not provided"):
        GeminiVisionModel(model_name="gemini-2.5-pro")


def test_import_error(monkeypatch):
    # Simulate missing dependency
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google.genai.types", None)
    import splitter_mr.model.models.gemini_model as mod

    with mock.patch.object(mod, "_require_extra") as mreq:
        mreq.side_effect = ImportError("This feature requires the 'multimodal' extra")
        with pytest.raises(ImportError, match="multimodal"):
            mod.GeminiVisionModel(api_key="irrelevant")


def test_get_client(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key")
    client = m.get_client()
    assert client is m.client


def test_analyze_content_success(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key", model_name="gemini-vision")
    # Prepare a valid base64-encoded image string (just dummy data)
    img_bytes = b"\x89PNG...."
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    result = m.analyze_content("Describe this image", img_b64, file_ext="png")
    assert isinstance(result, str)
    assert result == "This is the analyzed image text"


def test_analyze_content_bytes(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key", model_name="gemini-vision")
    # Should accept both bytes and str for file
    img_bytes = base64.b64encode(b"imagecontent")
    result = m.analyze_content("Describe this image", img_bytes, file_ext="jpg")
    assert result == "This is the analyzed image text"


def test_analyze_content_invalid_base64(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key")
    with pytest.raises(ValueError, match="Failed to decode base64"):
        m.analyze_content("prompt", "not-base64-data", file_ext="png")


def test_analyze_content_file_none(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key")
    with pytest.raises(ValueError, match="No image file provided"):
        m.analyze_content("Some prompt", None)


def test_analyze_content_extra_parameters(monkeypatch):
    fake_genai, fake_types = mock_gemini_sdk(monkeypatch)
    m = GeminiVisionModel(api_key="key", model_name="gemini-vision")
    img_bytes = base64.b64encode(b"dummydata").decode("utf-8")
    # We want to see that extra parameters are passed through

    result = m.analyze_content(
        "Analyze", img_bytes, file_ext="png", max_tokens=99, temperature=0.5
    )
    # Access the called_with to verify args
    called_with = m.model.called_with
    assert called_with[0] == "gemini-vision"
    assert any("FakePart" in str(x) for x in called_with[1])
    assert called_with[2]["max_tokens"] == 99
    assert called_with[2]["temperature"] == 0.5
    assert result == "This is the analyzed image text"
