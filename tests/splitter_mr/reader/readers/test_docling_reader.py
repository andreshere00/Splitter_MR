import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import AzureOpenAI, OpenAI

from splitter_mr.model.base_model import BaseModel
from splitter_mr.reader.readers.docling_reader import DoclingReader
from splitter_mr.schema import ReaderOutput

# Helpers


class ConcreteModel(BaseModel):
    def __init__(self, model_name="dummy-model"):
        self.model_name = model_name

    def get_client(self):
        return self

    def extract_text(self, *a, **kw):
        pass


@pytest.fixture
def patch_vlm(monkeypatch):
    monkeypatch.setattr(
        DoclingReader,
        "_get_vlm_url_and_headers",
        lambda self, client: ("https://dummy", {"Authorization": "Bearer dummy"}),
    )


class StubAzure(AzureOpenAI):
    def __init__(self):
        pass

    api_key = "stub"


class StubOpenAI(OpenAI):
    def __init__(self):
        pass

    api_key = "stub"


class DummyModel(BaseModel):
    def __init__(self, model_name="dummy-model"):
        self.api_key = "dummy-key"
        self.model_name = model_name
        self._azure_deployment = "dummy-deployment"
        self._azure_endpoint = "https://dummy-endpoint.com"
        self._api_version = "2025-01-01-preview"

    def get_client(self):
        return self

    def extract_text(self, *a, **kw):
        pass


@pytest.fixture
def dummy_pdf_path(tmp_path):
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4 test content")
    return str(p)


@pytest.fixture
def unsupported_file_path(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("Not supported extension.")
    return str(p)


@pytest.fixture
def dummy_model():
    return DummyModel()


# Test cases


def test_dummy_model_instantiable_and_methods_work():
    model = ConcreteModel()
    assert model.get_client() is model


def test_init_with_and_without_model(dummy_model):
    reader = DoclingReader(model=dummy_model)
    assert reader.model is dummy_model
    assert reader.api_key == "dummy-key"
    reader_no_model = DoclingReader()
    assert reader_no_model.model is None


def test_supported_extensions_check(dummy_model, dummy_pdf_path):
    reader = DoclingReader(model=dummy_model)
    assert Path(dummy_pdf_path).suffix.lstrip(".") in reader.SUPPORTED_EXTENSIONS


def test_unsupported_extensions_triggers_vanilla(monkeypatch, unsupported_file_path):
    reader = DoclingReader()
    called = {}

    def fake_vanilla_read(file_path, **kwargs):
        called["called"] = True
        return ReaderOutput(
            text="Fallback text",
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=str(uuid.uuid4()),
            conversion_method="vanilla",
            reader_method="vanilla",
            ocr_method=None,
            metadata=None,
        )

    monkeypatch.setattr(
        "splitter_mr.reader.readers.docling_reader.VanillaReader",
        lambda: MagicMock(read=fake_vanilla_read),
    )
    out = reader.read(unsupported_file_path)
    assert called["called"]
    assert out.conversion_method == "vanilla"


@patch("splitter_mr.reader.readers.docling_reader.DocumentConverter")
def test_read_success_with_mocked_docling(
    mock_docconv, dummy_model, dummy_pdf_path, patch_vlm
):
    reader = DoclingReader(model=dummy_model)
    fake_result = MagicMock()
    fake_result.document.export_to_markdown.return_value = "# Hello"
    mock_docconv.return_value.convert.return_value = fake_result
    out = reader.read(dummy_pdf_path)
    assert out.text == "# Hello"


@patch("splitter_mr.reader.readers.docling_reader.DocumentConverter")
def test_read_missing_file_raises(mock_docconv, dummy_model, patch_vlm):
    reader = DoclingReader(model=dummy_model)
    mock_docconv.return_value.convert.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        reader.read("missing.pdf")


def test_get_vlm_url_and_headers_for_azure(dummy_model):
    reader = DoclingReader(model=dummy_model)
    reader._azure_deployment = "dep"
    reader._azure_endpoint = "https://azure.endpoint"
    reader._api_version = "v2025"

    url, hdr = reader._get_vlm_url_and_headers(StubAzure())
    assert "azure.endpoint" in url
    assert hdr["Authorization"].startswith("Bearer")


def test_get_vlm_url_and_headers_for_openai(dummy_model, monkeypatch):
    reader = DoclingReader(model=dummy_model)
    monkeypatch.setattr(
        DoclingReader,
        "_get_vlm_url_and_headers",
        DoclingReader.__dict__["_get_vlm_url_and_headers"],
        raising=False,
    )
    url, hdr = reader._get_vlm_url_and_headers(StubOpenAI())
    assert "api.openai.com" in url and hdr["Authorization"].startswith("Bearer")
