from types import SimpleNamespace

import pytest

from splitter_mr.reader.utils.docling_utils import (
    DoclingPipelineFactory,
    get_vlm_url_and_headers,
    markdown_pipeline,
    page_image_pipeline,
    vlm_pipeline,
)


class DummyAzureClient:
    def __init__(
        self, endpoint="https://ep", deployment="dep", version="v1", api_key="key"
    ):
        self._azure_endpoint = endpoint
        self._azure_deployment = deployment
        self._api_version = version
        self.api_key = api_key


class DummyOpenAIClient:
    def __init__(self, api_key="oaikey"):
        self.api_key = api_key


class DummyModel:
    def __init__(self, model_name="mymodel", text="caption", client=None):
        self.model_name = model_name
        self._text = text
        self._client = client or DummyOpenAIClient()

    def get_client(self):
        return self._client

    def extract_text(self, prompt, file):
        return f"{self._text}:prompt={prompt}"


class DummyPILImage:
    def save(self, buf, format):
        buf.write(b"\x89PNG\r\n\x1a\nfake")


@pytest.fixture(autouse=True)
def patch_docling(monkeypatch):
    # Patch DocumentConverter to fake the convert result for all pipelines
    class DummyPage:
        def __init__(self):
            self.image = SimpleNamespace(pil_image=DummyPILImage())

    class DummyDoc:
        def __init__(self, pages=None):
            self.pages = {1: DummyPage(), 2: DummyPage()} if pages is None else pages

        def export_to_markdown(
            self, image_mode=None, page_break_placeholder=None, image_placeholder=None
        ):
            return (
                f"md-export-{image_mode}-{page_break_placeholder}-{image_placeholder}"
            )

    class DummyConvertResult:
        def __init__(self, doc=None):
            self.document = doc or DummyDoc()

    monkeypatch.setattr(
        "splitter_mr.reader.utils.docling_utils.DocumentConverter",
        lambda *a, **kw: SimpleNamespace(convert=lambda path: DummyConvertResult()),
    )
    # Patch AzureOpenAI and OpenAI for isinstance checks
    import splitter_mr.reader.utils.docling_utils as docling_utils

    docling_utils.AzureOpenAI = DummyAzureClient
    docling_utils.OpenAI = DummyOpenAIClient
    yield


def test_get_vlm_url_and_headers_azure_success():
    client = DummyAzureClient(
        endpoint="https://myep", deployment="dep", version="2024-06-01", api_key="abc"
    )
    url, headers = get_vlm_url_and_headers(client)
    assert url.startswith("https://myep/openai/deployments/dep/chat/completions")
    assert "api-version=2024-06-01" in url
    assert headers == {"Authorization": "Bearer abc"}


def test_get_vlm_url_and_headers_azure_missing():
    client = DummyAzureClient(endpoint=None, deployment="dep", version="v1")
    with pytest.raises(ValueError):
        get_vlm_url_and_headers(client)


def test_get_vlm_url_and_headers_openai_success():
    client = DummyOpenAIClient(api_key="OPENAIKEY")
    url, headers = get_vlm_url_and_headers(client)
    assert url.startswith("https://api.openai.com/v1/chat/completions")
    assert headers == {"Authorization": "Bearer OPENAIKEY"}


def test_get_vlm_url_and_headers_invalid_client():
    class UnknownClient:
        api_key = "whatever"

    with pytest.raises(ValueError):
        get_vlm_url_and_headers(UnknownClient())


def test_page_image_pipeline_with_model(monkeypatch):
    model = DummyModel(text="OUT")
    out = page_image_pipeline(
        file_path="dummy.pdf",
        model=model,
        prompt="PROMPT",
        image_resolution=2.0,
        show_base64_images=False,
        page_placeholder="<!--PG-->",
    )
    assert "<!--PG-->" in out
    assert "OUT:prompt=PROMPT" in out


def test_page_image_pipeline_no_model_with_base64(monkeypatch):
    out = page_image_pipeline(
        file_path="dummy.pdf", model=None, show_base64_images=True
    )
    assert "data:image/png;base64," in out


def test_page_image_pipeline_value_error():
    with pytest.raises(ValueError):
        page_image_pipeline("f.pdf", model=None, show_base64_images=False)


def test_vlm_pipeline(monkeypatch):
    model = DummyModel(model_name="VLMMODEL", client=DummyAzureClient(api_key="k2"))
    out = vlm_pipeline(
        file_path="x.pdf",
        model=model,
        prompt="PR",
        page_placeholder="<!--PG-->",
        image_placeholder="<!--IMG-->",
    )
    assert out.startswith("md-export-")


def test_vlm_pipeline_openai_client(monkeypatch):
    model = DummyModel(model_name="XYZ", client=DummyOpenAIClient(api_key="opk"))
    out = vlm_pipeline(file_path="x.pdf", model=model, prompt="someprompt")
    assert out.startswith("md-export-")


def test_markdown_pipeline_pdf(monkeypatch):
    out = markdown_pipeline(
        file_path="doc.pdf",
        show_base64_images=True,
        page_placeholder="<!--PAGE-->",
        image_placeholder="<!--IMG-->",
        image_resolution=1.23,
        ext="pdf",
    )
    assert out.startswith("md-export-")


def test_markdown_pipeline_nonpdf(monkeypatch):
    out = markdown_pipeline(
        file_path="doc.docx",
        show_base64_images=False,
        page_placeholder="PG",
        image_placeholder="IMG",
        ext="docx",
    )
    assert out.startswith("md-export-")


def test_pipeline_factory_register_and_get():
    def dummy(file_path, **kw):
        return "ok"

    DoclingPipelineFactory.register("foo", dummy)
    got = DoclingPipelineFactory.get("foo")
    assert got is dummy


def test_pipeline_factory_run():
    def dummy(file_path, **kw):
        return f"ran:{file_path}"

    DoclingPipelineFactory.register("bar", dummy)
    out = DoclingPipelineFactory.run("bar", "file.txt", arg1="v")
    assert out == "ran:file.txt"


def test_pipeline_factory_get_unregistered():
    with pytest.raises(ValueError):
        DoclingPipelineFactory.get("doesnotexist")


def test_pipeline_factory_run_unregistered():
    with pytest.raises(ValueError):
        DoclingPipelineFactory.run("doesnotexist", "f.txt")
