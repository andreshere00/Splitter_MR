from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ApiVlmOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling.pipeline.vlm_pipeline import VlmPipeline

from splitter_mr.reader.utils.docling_utils import DoclingUtils

# ---------------------------------------------------------------------------
# Dummy client implementations
# ---------------------------------------------------------------------------


class DummyAzureClient:  # noqa: D101 (docstring not critical for dummy)
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        deployment: str | None = None,
        version: str | None = None,
        api_key: str = "key",
    ) -> None:
        self._azure_endpoint = endpoint
        self._azure_deployment = deployment
        self._api_version = version
        self.api_key = api_key


class DummyOpenAIClient:  # noqa: D101
    def __init__(self, *, api_key: str = "key") -> None:
        self.api_key = api_key


class UnknownClient(SimpleNamespace):  # noqa: D101
    """A client type that is *not* recognised by DoclingUtils."""

    api_key: str = "unknown-key"


# ---------------------------------------------------------------------------
# Global fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_openai_types(monkeypatch):
    """Patch the AzureOpenAI and OpenAI symbols *inside* docling_utils.

    That way `isinstance(client, AzureOpenAI)` passes for our dummy clients
    even though we do *not* have the real SDK in the test environment.
    """

    from splitter_mr.reader import utils as utils_pkg  # local import to grab module

    monkeypatch.setattr(
        utils_pkg.docling_utils, "AzureOpenAI", DummyAzureClient, raising=False
    )
    monkeypatch.setattr(
        utils_pkg.docling_utils, "OpenAI", DummyOpenAIClient, raising=False
    )
    yield  # allow the test to run


# ---------------------------------------------------------------------------
# Tests for get_vlm_url_and_headers
# ---------------------------------------------------------------------------


def test_get_vlm_url_and_headers_azure_success():
    """A fully‑specified Azure client should yield a well‑formed URL + header."""

    client = DummyAzureClient(
        endpoint="https://example.com/endpoint/",
        deployment="dep",
        version="v1",
        api_key="testkey",
    )
    url, headers = DoclingUtils.get_vlm_url_and_headers(client)

    parsed = urlparse(url)
    assert headers == {"Authorization": "Bearer testkey"}
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert parsed.path.endswith("/openai/deployments/dep/chat/completions")
    assert parse_qs(parsed.query)["api-version"] == ["v1"]


def test_get_vlm_url_and_headers_azure_missing_params():
    """Missing any Azure parameter should raise ValueError."""

    client = DummyAzureClient(endpoint=None, deployment="dep", version="v1")
    with pytest.raises(ValueError):
        DoclingUtils.get_vlm_url_and_headers(client)


def test_get_vlm_url_and_headers_openai():
    """An OpenAI client should map to the public chat completions endpoint."""

    client = DummyOpenAIClient(api_key="openai-key")
    url, headers = DoclingUtils.get_vlm_url_and_headers(client)

    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.openai.com"
    assert parsed.path == "/v1/chat/completions"
    assert headers == {"Authorization": "Bearer openai-key"}


def test_get_vlm_url_and_headers_invalid_client():
    """Any *other* client type should raise a ValueError (after header build)."""

    with pytest.raises(ValueError):
        DoclingUtils.get_vlm_url_and_headers(UnknownClient())


# ---------------------------------------------------------------------------
# Tests for get_pdf_pipeline
# ---------------------------------------------------------------------------


def test_get_pdf_pipeline_image_mode(monkeypatch):
    """Verify PdfPipelineOptions are plumbed through in image mode."""

    captured: dict[str, object] = {}

    def fake_converter_init(self, *, format_options=None, **_kwargs):  # noqa: D401,E501
        captured["format_options"] = format_options
        # DocumentConverter has no required side‑effects for these tests

    monkeypatch.setattr(
        DocumentConverter, "__init__", fake_converter_init, raising=True
    )

    util = DoclingUtils()
    converter = util.get_pdf_pipeline(
        mode="image",
        images_scale=3.0,
        generate_page_images=False,
        generate_picture_images=False,
    )

    # We still get *some* DocumentConverter instance back (constructor patched).
    assert isinstance(converter, DocumentConverter)

    fmt_opts = captured["format_options"]
    assert InputFormat.PDF in fmt_opts

    pdf_opt = fmt_opts[InputFormat.PDF]
    assert isinstance(pdf_opt.pipeline_options, PdfPipelineOptions)
    opts: PdfPipelineOptions = pdf_opt.pipeline_options
    assert opts.images_scale == 3.0
    assert opts.generate_page_images is False
    assert opts.generate_picture_images is False


def test_get_pdf_pipeline_vlm_mode_requires_client():
    """A client must be supplied for VLM mode."""

    util = DoclingUtils()
    with pytest.raises(ValueError):
        util.get_pdf_pipeline(mode="vlm", client=None)


def test_get_pdf_pipeline_vlm_mode(monkeypatch):
    """Ensure VlmPipeline and ApiVlmOptions are wired in correctly."""

    captured: dict[str, object] = {}

    def fake_converter_init(self, *, format_options=None, **_kwargs):
        captured["format_options"] = format_options

    monkeypatch.setattr(
        DocumentConverter, "__init__", fake_converter_init, raising=True
    )

    azure = DummyAzureClient(
        endpoint="https://ex.com/", deployment="d", version="v2", api_key="k"
    )
    util = DoclingUtils()
    converter = util.get_pdf_pipeline(
        mode="vlm",
        client=azure,
        model_name="mymodel",
        prompt="my prompt",
        timeout=42,
    )

    assert isinstance(converter, DocumentConverter)

    fmt_opts = captured["format_options"]
    assert InputFormat.PDF in fmt_opts
    pdf_opt = fmt_opts[InputFormat.PDF]

    # The PDF option should point to the VLM pipeline class
    assert pdf_opt.pipeline_cls is VlmPipeline

    vlm_opts: ApiVlmOptions = pdf_opt.pipeline_options.vlm_options
    assert vlm_opts.params["model"] == "mymodel"
    assert vlm_opts.prompt == "my prompt"
    assert vlm_opts.timeout == 42


def test_get_pdf_pipeline_invalid_mode():
    """Anything other than 'vlm' or 'image' should error."""

    util = DoclingUtils()
    with pytest.raises(ValueError):
        util.get_pdf_pipeline(mode="bogus")
