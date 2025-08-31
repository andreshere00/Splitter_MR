from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.schema import DEFAULT_IMAGE_CAPTION_PROMPT

# ------ Helpers and fixtures ------ #


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


_IMAGE_MIME_BY_EXT = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _mocked_client(return_text="OK"):
    client = MagicMock()
    client._azure_deployment = "deployment"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=return_text))]
    client.chat.completions.create.return_value = mock_response
    return client


# ------ Tests cases ------ #


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


def test_analyze_content_makes_correct_call():
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
        text = model.analyze_content("dGVzdF9pbWFnZQ==", prompt="Extract!")
        # Check if the correct payload was sent
        mock_client.chat.completions.create.assert_called_once()
        called_args = mock_client.chat.completions.create.call_args[1]
        assert called_args["model"] == "deployment"
        assert called_args["messages"][0]["content"][0]["text"] == "Extract!"
        assert text == "Extracted text"


def test_analyze_content_uses_jpeg_mime_for_jpg_ext():
    client = _mocked_client("jpeg!")
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("Zm9v", prompt="p", file_ext="jpg")

        # Grab the payload sent to the API
        called = client.chat.completions.create.call_args.kwargs
        content = called["messages"][0]["content"]
        image_part = next(x for x in content if x["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "ext,mime_prefix",
    [
        ("jpg", "data:image/jpeg;base64,"),
        ("jpeg", "data:image/jpeg;base64,"),
        ("png", "data:image/png;base64,"),
        ("gif", "data:image/gif;base64,"),
        ("webp", "data:image/webp;base64,"),
    ],
)
def test_analyze_content_sets_expected_mime_for_common_exts(ext, mime_prefix):
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("Zm9v", prompt="p", file_ext=ext)
        called = client.chat.completions.create.call_args.kwargs
        image_part = next(
            x for x in called["messages"][0]["content"] if x["type"] == "image_url"
        )
        assert image_part["image_url"]["url"].startswith(mime_prefix)


def test_analyze_content_forwards_extra_parameters():
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content(
            "Zm9v", prompt="hey", file_ext="png", temperature=0.2, presence_penalty=1
        )

        called = client.chat.completions.create.call_args.kwargs
        # params forwarded
        assert called["temperature"] == 0.2
        assert called["presence_penalty"] == 1
        # and still uses deployment as model
        assert called["model"] == "deployment"


def test_analyze_content_uses_default_prompt_when_omitted():
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("Zm9v")  # no prompt

        called = client.chat.completions.create.call_args.kwargs
        text_part = called["messages"][0]["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == DEFAULT_IMAGE_CAPTION_PROMPT


def test_init_respects_env_api_version(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env_key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-endpoint")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-depl")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2029-01-01-preview")

    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI"
    ) as mock_client:
        _ = AzureOpenAIVisionModel()
        mock_client.assert_called_once()
        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_version"] == "2029-01-01-preview"


def test_analyze_content_includes_image_url_block():
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("Zm9v", prompt="go", file_ext="png")

        called = client.chat.completions.create.call_args.kwargs
        content = called["messages"][0]["content"]
        # Should have both text and image blocks
        kinds = [c["type"] for c in content]
        assert "text" in kinds
        assert "image_url" in kinds


def test_analyze_content_unknown_ext_falls_back_to_png():
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("Zm9v", prompt="p", file_ext="totally-unknown-ext")
        called = client.chat.completions.create.call_args.kwargs
        image_part = next(
            x for x in called["messages"][0]["content"] if x["type"] == "image_url"
        )
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("ext", ["tiff", "bmp", "svg", "heic"])
def test_analyze_content_raises_on_unsupported_mime(ext):
    client = _mocked_client()
    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        with pytest.raises(ValueError, match="Unsupported image MIME type"):
            m.analyze_content("Zm9v", prompt="p", file_ext=ext)


@pytest.mark.parametrize(
    "ext,mime_prefix",
    [
        ("jpg", "data:image/jpeg;base64,"),
        ("jpeg", "data:image/jpeg;base64,"),
        ("png", "data:image/png;base64,"),
        ("gif", "data:image/gif;base64,"),
        ("webp", "data:image/webp;base64,"),
    ],
)
def test_analyze_content_accepts_common_mime_types(ext, mime_prefix):
    client = MagicMock()
    client._azure_deployment = "deployment"
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
    client.chat.completions.create.return_value = mock_resp

    with patch(
        "splitter_mr.model.models.azure_openai_model.AzureOpenAI", return_value=client
    ):
        m = AzureOpenAIVisionModel(
            api_key="k",
            azure_endpoint="e",
            azure_deployment="deployment",
            api_version="v",
        )
        _ = m.analyze_content("AAAA", prompt="p", file_ext=ext)

        called = client.chat.completions.create.call_args.kwargs
        content = called["messages"][0]["content"]
        image_part = next(x for x in content if x["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith(mime_prefix)
