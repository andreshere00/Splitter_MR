import json
import uuid

import pandas as pd
import pytest
import yaml

from splitter_mr.reader.readers.vanilla_reader import (
    SimpleHTMLTextExtractor,
    VanillaReader,
)
from splitter_mr.schema import ReaderOutput

# ---------- Helper Fixtures ----------


class DummyPDFPlumberReader:
    def __init__(self):
        self.last_kwargs = None  # store kwargs for assertions

    # element-wise extraction (legacy path)
    def read(self, *a, **kw):
        self.last_kwargs = kw
        return "ELEMENT_WISE_PDF_TEXT"

    # full-page vision pipeline
    def describe_pages(self, file_path, model, prompt, resolution=300, **kw):
        # record params so the test can inspect them
        self.last_kwargs = {
            "file_path": file_path,
            "model": model,
            "prompt": prompt,
            "resolution": resolution,
            **kw,
        }
        # pretend 2-page PDF
        return ["PAGE-1-MD", "PAGE-2-MD"]


class DummyVisionModel:
    model_name = "dummy-vlm"


@pytest.fixture(autouse=True)
def patch_pdf_reader(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFPlumberReader(),
    )
    yield


# ---------- Tests for SimpleHTMLTextExtractor ----------


def test_simple_html_text_extractor_basic():
    html = "<div>Hello <b>World</b> &amp; Friends!</div>"
    parser = SimpleHTMLTextExtractor()
    parser.feed(html)
    assert " ".join(parser.get_text().split()) == "Hello World & Friends!"


# ---------- VanillaReader: file_path handling ----------


def test_read_txt_file(tmp_path):
    path = tmp_path / "doc.txt"
    content = "hello world"
    path.write_text(content)

    reader = VanillaReader()
    out = reader.read(str(path))
    assert isinstance(out, ReaderOutput)
    assert out.text == content
    assert out.document_name == "doc.txt"
    assert out.document_path.endswith("doc.txt")
    assert out.conversion_method == "txt"
    assert out.reader_method == "vanilla"


def test_read_json_file(tmp_path):
    path = tmp_path / "data.json"
    data = {"foo": "bar"}
    path.write_text(json.dumps(data))

    reader = VanillaReader()
    out = reader.read(str(path))
    assert (
        json.loads(out.text)["foo"] == "bar" or out.text["foo"] == "bar"
    )  # Handles both str and dict


def test_read_yaml_file(tmp_path):
    path = tmp_path / "data.yaml"
    d = {"a": 123}
    path.write_text(yaml.safe_dump(d))

    reader = VanillaReader()
    out = reader.read(str(path))
    # YAML is loaded, so out.text should be dict
    assert isinstance(out.text, dict)
    assert out.text["a"] == 123
    assert out.conversion_method == "json"


def test_read_csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2")
    reader = VanillaReader()
    out = reader.read(str(path))
    assert "a,b" in out.text


def test_read_parquet_file(monkeypatch, tmp_path):
    # Patch pandas.read_parquet to return a DataFrame
    df = pd.DataFrame({"x": [1], "y": [2]})
    monkeypatch.setattr(pd, "read_parquet", lambda fp: df)
    path = tmp_path / "data.parquet"
    path.write_text("dummy")  # Content not used

    reader = VanillaReader()
    out = reader.read(str(path))
    assert "x,y" in out.text
    assert out.conversion_method == "csv"


def test_read_excel_file(monkeypatch, tmp_path):
    df = pd.DataFrame({"x": [1], "y": [2]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)
    path = tmp_path / "data.xlsx"
    path.write_text("dummy")  # Content not used
    reader = VanillaReader()
    out = reader.read(str(path))
    assert "x,y" in out.text
    assert out.conversion_method == "xlsx"


def test_read_pdf_file(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    out = reader.read(str(path))
    assert out.text == "ELEMENT_WISE_PDF_TEXT"
    assert out.conversion_method == "pdf"


def test_read_unsupported_extension(tmp_path):
    path = tmp_path / "file.xyz"
    path.write_text("dummy")
    reader = VanillaReader()
    with pytest.raises(ValueError):
        reader.read(str(path))


def test_read_txt_file_with_languages(tmp_path, monkeypatch):
    # Simulate file in LANGUAGES
    path = tmp_path / "doc.en"
    path.write_text("Hello")
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.LANGUAGES", ["en", "fr"]
    )
    reader = VanillaReader()
    out = reader.read(str(path))
    assert out.text == "Hello"


# ---------- file_path but actually URL ----------


def test_read_url(monkeypatch):
    url = "https://example.com/data.txt"
    content = "hello from url"

    class DummyResponse:
        def __init__(self, text, headers=None):
            self.text = text
            self.headers = headers or {"Content-Type": "text/plain"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"a": "b"}

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.requests.get",
        lambda u: DummyResponse(content),
    )
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: True)
    out = reader.read(url)
    assert out.text == content


def test_read_url_json(monkeypatch):
    url = "https://example.com/data.json"

    class DummyResponse:
        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"k": "v"}

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.requests.get",
        lambda u: DummyResponse(),
    )
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: True)
    out = reader.read(url)
    assert out.text == {"k": "v"}


# ---------- file_path but actually JSON string ----------


def test_read_file_path_as_json(monkeypatch):
    s = '{"foo": 3}'
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: False)
    out = reader.read(s)
    assert out.text["foo"] == 3


# ---------- file_path but actually YAML string ----------


def test_read_file_path_as_yaml(monkeypatch):
    s = "foo: bar"
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: False)
    out = reader.read(s)
    assert out.text["foo"] == "bar"


# ---------- explicit file_url ----------


def test_explicit_file_url(monkeypatch):
    url = "https://test.me/file.txt"
    content = "hi"

    class DummyResponse:
        headers = {"Content-Type": "text/plain"}

        def raise_for_status(self):
            pass

        text = content

        def json(self):
            return {"q": 7}

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.requests.get",
        lambda u: DummyResponse(),
    )
    reader = VanillaReader()
    out = reader.read(file_url=url)
    assert out.text == content
    assert out.document_name == "file.txt"
    assert out.document_path == url


def test_explicit_file_url_json(monkeypatch):
    url = "https://test.me/data.json"

    class DummyResponse:
        headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"x": "y"}

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.requests.get",
        lambda u: DummyResponse(),
    )
    reader = VanillaReader()
    out = reader.read(file_url=url)
    assert out.text == {"x": "y"}


# ---------- explicit json_document ----------


def test_explicit_json_document():
    reader = VanillaReader()
    out = reader.read(json_document='{"foo": "bar"}', document_name="abc")
    assert out.text["foo"] == "bar"
    assert out.document_name == "abc"
    assert out.conversion_method == "json"


def test_explicit_json_document_dict():
    reader = VanillaReader()
    out = reader.read(json_document={"x": 1})
    assert out.text["x"] == 1
    assert out.conversion_method == "json"


# ---------- explicit text_document ----------


def test_explicit_text_document_json():
    reader = VanillaReader()
    out = reader.read(text_document="[1,2,3]")
    assert out.text == [1, 2, 3]
    assert out.conversion_method == "json"


def test_explicit_text_document_yaml():
    reader = VanillaReader()
    out = reader.read(text_document="a: 1")
    assert out.text["a"] == 1
    assert out.conversion_method == "json"


def test_explicit_text_document_fallback():
    reader = VanillaReader()
    out = reader.read(text_document="plain text here")
    assert out.text == "plain text here"
    assert out.conversion_method == "txt"


# ---------- error branches ----------


def test_file_path_wrong_type():
    reader = VanillaReader()
    with pytest.raises(ValueError):
        reader.read(123)


def test_explicit_file_url_invalid():
    reader = VanillaReader()
    with pytest.raises((ValueError, TypeError)):
        reader.read(file_url=123)
    with pytest.raises(ValueError):
        reader.read(file_url="notaurl")


def test_unrecognized_source():
    reader = VanillaReader()
    with pytest.raises(ValueError):
        reader.read(foo="bar")


# ---------- metadata and ids ----------


def test_reader_metadata_and_ids(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("hello")
    reader = VanillaReader()
    doc_id = str(uuid.uuid4())
    out = reader.read(str(path), metadata={"source": "x"}, document_id=doc_id)
    assert out.metadata == {"source": "x"}
    assert out.document_id == doc_id


#  ---------- scan_pdf_pages functionalities ----------


def test_scan_pdf_pages_success(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    reader = VanillaReader(model=DummyVisionModel())  # default model OK
    out = reader.read(
        str(pdf_path),
        scan_pdf_pages=True,
        resolution=300,  # custom DPI should flow through
        vlm_parameters={"temperature": 0.0},
    )

    # → markdown with page headings
    assert "## Page 1" in out.text and "PAGE-1-MD" in out.text
    assert "## Page 2" in out.text and "PAGE-2-MD" in out.text

    # → metadata fields
    assert out.conversion_method == "png"
    assert out.ocr_method == "dummy-vlm"

    # → our DummyPDFPlumberReader captured the kwargs
    pdf_reader = reader.pdf_reader  # the instance inside VanillaReader
    recorded = pdf_reader.last_kwargs
    assert recorded["resolution"] == 300
    assert recorded["model"] is reader.model
    # default prompt should be passed untouched
    assert "Extract all the elements detected in the page" in recorded["prompt"]


def test_scan_pdf_pages_requires_model(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader(model=None)  # no default model

    with pytest.raises(ValueError):
        reader.read(str(pdf_path), scan_pdf_pages=True)


def test_pdf_with_model_no_scan(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    model = DummyVisionModel()
    reader = VanillaReader(model=None)  # no default; we pass per-call

    out = reader.read(str(pdf_path), model=model, scan_pdf_pages=False)

    assert out.text == "ELEMENT_WISE_PDF_TEXT"
    assert out.conversion_method == "pdf"
    assert out.ocr_method == model.model_name


def test_pdf_custom_placeholder(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    custom_placeholder = "<!-- custom-img -->"
    out = reader.read(str(pdf_path), placeholder=custom_placeholder)
    # Our DummyPDFPlumberReader records kwargs
    assert reader.pdf_reader.last_kwargs["placeholder"] == custom_placeholder


def test_pdf_custom_placeholder_with_model(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    model = DummyVisionModel()
    reader = VanillaReader(model=model)
    custom_placeholder = "<!-- myimg -->"
    out = reader.read(str(pdf_path), model=model, placeholder=custom_placeholder)
    assert reader.pdf_reader.last_kwargs["placeholder"] == custom_placeholder


def test_pdf_default_placeholder(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    out = reader.read(str(pdf_path))
    assert reader.pdf_reader.last_kwargs["placeholder"] == "<!-- image -->"
