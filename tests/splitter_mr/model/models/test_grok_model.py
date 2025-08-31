# tests/test_grok_vision_model.py
import types
from importlib import import_module

import pytest

MODULE_UNDER_TEST = (
    "splitter_mr.model.models.grok_model"  # <-- change this to your module path
)


@pytest.fixture()
def mod(monkeypatch):
    """
    Import the module under test and return it, making sure it can be monkeypatched.
    """
    module = import_module(MODULE_UNDER_TEST)
    return module


class DummyCreateResponse:
    def __init__(self, content: str):
        self.choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content=content))
        ]


class DummyChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        # record call for assertions
        self.calls.append(kwargs)
        # default return
        return DummyCreateResponse("dummy-response")


class DummyClient:
    """
    Mimics openai.Client enough for these tests:
    - must expose `chat.completions.create(...)`
    """

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url
        self.chat = types.SimpleNamespace(completions=DummyChatCompletions())


# -----------------------
# Constructor and client
# -----------------------


def test_init_with_api_key_arg(monkeypatch, mod):
    # Patch the Client class in the module to our dummy
    monkeypatch.setattr(mod, "Client", DummyClient)

    model = mod.GrokVisionModel(api_key="XYZ", model_name="grok-4-mini")
    assert model.model_name == "grok-4-mini"
    assert isinstance(model.client, DummyClient)
    # Confirm the dummy got the right values
    assert model.client._api_key == "XYZ"
    assert model.client._base_url == "https://api.x.ai/v1"


def test_init_with_env_var(monkeypatch, mod):
    monkeypatch.setenv("XAI_API_KEY", "FROM_ENV")
    monkeypatch.setattr(mod, "Client", DummyClient)

    model = mod.GrokVisionModel()  # picks env var
    assert isinstance(model.client, DummyClient)
    assert model.client._api_key == "FROM_ENV"


def test_init_without_api_key_raises(monkeypatch, mod):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    # Do not patch Client; we want the constructor to fail before using it
    with pytest.raises(ValueError) as e:
        mod.GrokVisionModel(api_key=None)
    assert "Grok API key not provided" in str(e.value)


def test_get_client_returns_client(monkeypatch, mod):
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")
    assert model.get_client() is model.client


# -----------------------
# analyze_content: validation
# -----------------------


def test_analyze_content_raises_when_no_file(monkeypatch, mod):
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")
    with pytest.raises(ValueError) as e:
        model.analyze_content(file=None)
    assert "No file content provided" in str(e.value)


def test_analyze_content_unsupported_mime_raises(monkeypatch, mod):
    """
    Force mime resolution to 'image/webp' and mark it unsupported.
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    # Control mappings
    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {"webp": "image/webp"})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/png", "image/jpeg"})

    with pytest.raises(ValueError) as e:
        model.analyze_content(file="Zm9v", file_ext="webp")
    assert "Unsupported image MIME type: image/webp" in str(e.value)


# -----------------------
# analyze_content: mime paths
# -----------------------


def test_analyze_content_mime_from_custom_map(monkeypatch, mod):
    """
    GROK_MIME_BY_EXTENSION provides the mapping (preferred branch).
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {"heic": "image/heic"})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/heic"})

    # Also ensure DEFAULT_IMAGE_CAPTION_PROMPT exists
    monkeypatch.setattr(mod, "DEFAULT_IMAGE_CAPTION_PROMPT", "Describe image")

    resp = model.analyze_content(file="QUJD", file_ext="heic", prompt="What's in this?")
    assert resp == "dummy-response"

    # Assert the request composed the DataURI with the resolved MIME
    call = model.client.chat.completions.calls[-1]
    assert call["model"] == model.model_name
    assert "messages" in call
    msg = call["messages"][0]
    # Pydantic object — access attributes
    assert msg.role == "user"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "What's in this?"
    assert msg.content[1].type == "image_url"
    assert msg.content[1].image_url.url.startswith("data:image/heic;base64,")
    assert (
        msg.messages if False else True
    )  # no-op, just making sure nothing else breaks


def test_analyze_content_mime_from_mimetypes_map(monkeypatch, mod):
    """
    When GROK_MIME_BY_EXTENSION lacks the ext, fallback to mimetypes.types_map.
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/jpeg"})

    # monkeypatch the 'mimetypes.types_map' used by the module
    mod.mimetypes.types_map[".jpg"] = "image/jpeg"

    out = model.analyze_content(file="QUJD", file_ext="jpg")
    assert out == "dummy-response"
    call = model.client.chat.completions.calls[-1]
    msg = call["messages"][0]
    assert msg.content[1].image_url.url.startswith("data:image/jpeg;base64,")


def test_analyze_content_mime_default_png(monkeypatch, mod):
    """
    If neither custom map nor mimetypes knows the ext, defaults to image/png.
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/png"})

    # Ensure mimetypes has no mapping for '.xyz'
    mod.mimetypes.types_map.pop(".xyz", None)

    out = model.analyze_content(file="QUJD", file_ext="xyz")
    assert out == "dummy-response"
    call = model.client.chat.completions.calls[-1]
    msg = call["messages"][0]
    assert msg.content[1].image_url.url.startswith("data:image/png;base64,")


def test_analyze_content_file_ext_is_case_insensitive(monkeypatch, mod):
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/jpeg"})
    mod.mimetypes.types_map[".jpg"] = "image/jpeg"

    out = model.analyze_content(file="QUJD", file_ext="JPG")
    assert out == "dummy-response"
    call = model.client.chat.completions.calls[-1]
    msg = call["messages"][0]
    assert msg.content[1].image_url.url.startswith("data:image/jpeg;base64,")


# -----------------------
# Payload + parameters
# -----------------------


def test_analyze_content_builds_ordered_content_and_detail(monkeypatch, mod):
    """
    Ensures content ordering: text first, then image, and 'detail' is set.
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    # Allow jpeg
    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/jpeg"})
    mod.mimetypes.types_map[".jpg"] = "image/jpeg"

    out = model.analyze_content(
        file="QUJD",
        file_ext="jpg",
        prompt="Prompt A",
        detail="high",
        temperature=0.25,  # passthrough parameter
        max_tokens=128,
    )
    assert out == "dummy-response"

    call = model.client.chat.completions.calls[-1]
    assert call["temperature"] == 0.25
    assert call["max_tokens"] == 128

    msg = call["messages"][0]
    # 1) text first
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "Prompt A"
    # 2) image second + detail field present
    assert msg.content[1].type == "image_url"
    # The schema you showed has `detail` on image_url
    assert getattr(msg.content[1].image_url, "detail") == "high"


def test_analyze_content_uses_default_prompt_constant(monkeypatch, mod):
    """
    If prompt not provided, the function uses DEFAULT_IMAGE_CAPTION_PROMPT.
    """
    monkeypatch.setattr(mod, "Client", DummyClient)
    model = mod.GrokVisionModel(api_key="key")

    monkeypatch.setattr(mod, "GROK_MIME_BY_EXTENSION", {})
    monkeypatch.setattr(mod, "SUPPORTED_GROK_MIME_TYPES", {"image/png"})
    # define default prompt if missing
    monkeypatch.setattr(mod, "DEFAULT_IMAGE_CAPTION_PROMPT", "DEFAULT PROMPT")

    out = model.analyze_content(file="QUJD", file_ext="png")
    assert out == "dummy-response"

    call = model.client.chat.completions.calls[-1]
    msg = call["messages"][0]
    assert msg.content[0].text == "DEFAULT PROMPT"
