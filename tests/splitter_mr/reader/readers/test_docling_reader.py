from types import SimpleNamespace

import pytest
from docling.document_converter import DocumentConverter

from splitter_mr.model import BaseModel
from splitter_mr.reader import DoclingReader, VanillaReader
from splitter_mr.reader.utils import DoclingUtils
from splitter_mr.schema import ReaderOutput


class DummyPipeline:
    def __init__(self, output):
        self.output = output

    def convert(self, path, **kwargs):
        return SimpleNamespace(
            document=SimpleNamespace(
                export_to_markdown=lambda image_mode=None: self.output
            )
        )


class DummyModel(BaseModel):
    def __init__(self, text):
        self.model_name = "dummy"
        self._client = SimpleNamespace(
            _azure_endpoint="https://endpoint",
            _azure_deployment="dep",
            _api_version="v1",
            api_key="key",
        )
        self._text = text

    def get_client(self):
        return self._client

    def extract_text(self, file, prompt):
        return self._text


@pytest.fixture(autouse=True)
def patch_docling_utils(monkeypatch):
    """Mock get_pdf_pipeline to return dummy pipelines based on mode"""

    def fake_get_pdf_pipeline(
        self, mode, client=None, model_name=None, prompt=None, **kwargs
    ):
        if mode == "vlm":
            return DummyPipeline(output="# VLM OUTPUT")
        elif mode == "image":
            return DummyPipeline(output="![alt](data:image/png;base64,AAA)")
        raise ValueError(mode)

    monkeypatch.setattr(DoclingUtils, "get_pdf_pipeline", fake_get_pdf_pipeline)
    yield


def fake_get_pdf_pipeline(
    self, mode, client=None, model_name=None, prompt=None, **kwargs
):
    if mode == "vlm":
        return DummyPipeline(output="# VLM OUTPUT")
    elif mode == "image":
        # Esta salida debe contener una imagen base64 para que _process_images funcione bien
        return DummyPipeline(output="![alt](data:image/png;base64,AAA)")
    raise ValueError(mode)


def test_unsupported_extension(monkeypatch, tmp_path):
    path = tmp_path / "file.xyz"
    path.write_text("dummy")
    # Mock VanillaReader.read
    called = {}

    def fake_vr_read(self, file_path, **kwargs):
        called["args"] = (file_path, kwargs)
        return ReaderOutput(
            text="vr",
            document_name="file.xyz",
            document_path=str(path),
            document_id="id",
            conversion_method="",
            reader_method="",
            ocr_method=None,
            metadata=None,
        )

    monkeypatch.setattr(VanillaReader, "read", fake_vr_read)

    reader = DoclingReader()
    out = reader.read(str(path))
    assert out.text == "vr"
    assert called["args"][0] == str(path)


def test_read_pdf_without_model(monkeypatch, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")

    reader = DoclingReader(model=None)
    # no model, so _read_pdf returns image pipeline output
    md = reader._read_pdf(str(path), prompt="p", scan_pdf_pages=False)
    assert md == "![alt](data:image/png;base64,AAA)"

    # process_images without model and hiding images
    processed = reader._process_images(md, prompt="p", show_base64_images=False)
    assert processed == "<!-- image -->"

    # keep images when show_base64_images True
    processed2 = reader._process_images(md, prompt="p", show_base64_images=True)
    # no model, skip caption, hide still applies? actually only caption for model
    assert "data:image/png" in processed2


def test_read_pdf_with_model_vlm(monkeypatch, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF")

    model = DummyModel(text="caption")
    reader = DoclingReader(model=model)
    out_text = reader._read_pdf(str(path), prompt="p", scan_pdf_pages=True)
    assert out_text == "# VLM OUTPUT"


def test_read_non_pdf_with_and_without_model(monkeypatch, tmp_path):
    # create dummy txt file
    path = tmp_path / "a.txt"
    path.write_text("text")
    # monkeypatch DocumentConverter
    monkeypatch.setattr(
        DocumentConverter,
        "convert",
        lambda self, fp: SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: "md")
        ),
    )

    # without model
    reader1 = DoclingReader()
    md1 = reader1._read_non_pdf(str(path), prompt="p")
    assert md1 == "md"

    # with model
    model = DummyModel(text="x")
    reader2 = DoclingReader(model=model)
    md2 = reader2._read_non_pdf(str(path), prompt="p")
    assert md2 == "# VLM OUTPUT"


def test_format_caption_keeps_and_captions(monkeypatch):
    model = DummyModel(text="hello")
    reader = DoclingReader(model=model)
    b64 = "AAA"
    uri = f"data:image/png;base64,{b64}"
    alt = "alt"
    md = f"![{alt}]({uri})"
    # caption only
    out1 = reader._process_images(md, prompt="p", show_base64_images=False)
    assert "hello" in out1
    assert "<!-- image -->" in out1
    # keep image
    out2 = reader._process_images(md, prompt="p", show_base64_images=True)
    assert "![alt](" in out2
    assert "hello" in out2
