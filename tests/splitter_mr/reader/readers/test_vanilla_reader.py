import json
import uuid

import pandas as pd
import pytest
import yaml

from splitter_mr.reader.readers.vanilla_reader import (
    SimpleHTMLTextExtractor,
    VanillaReader,
    _read_excel,
    _read_parquet,
    _read_text_file,
)
from splitter_mr.schema import DEFAULT_IMAGE_EXTRACTION_PROMPT, ReaderOutput

# ---------- Helper Fixtures ----------


class DummyVisionModel:
    model_name = "dummy-vlm"

    def get_client(self):
        return None

    def analyze_content(self, file, prompt=None, **kwargs):
        # Return a string that encodes what was sent, for assertion
        return f"EXTRACTED_TEXT:{file[:10]}:{prompt or 'NO_PROMPT'}"


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


@pytest.fixture
def dummy_docx_file(tmp_path):
    file = tmp_path / "test.docx"
    file.write_text("dummy docx")
    return file


@pytest.fixture
def dummy_pptx_file(tmp_path):
    file = tmp_path / "test.pptx"
    file.write_text("dummy pptx")
    return file


@pytest.fixture
def dummy_xlsx_file(tmp_path):
    file = tmp_path / "test.xlsx"
    # You don't need a real xlsx for the logic, just a path
    file.write_text("dummy xlsx")
    return file


@pytest.fixture(autouse=True)
def patch_pdf_reader(monkeypatch):
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFPlumberReader(),
    )
    yield


@pytest.fixture
def create_image(tmp_path):
    from PIL import Image

    def _make(ext):
        img_path = tmp_path / f"test_image.{ext}"
        # Just create a small RGB image for non-svg, SVG is handled separately
        if ext == "svg":
            img_path.write_text(
                '<svg height="100" width="100"><circle cx="50" cy="50" r="40" /></svg>'
            )
        else:
            img = Image.new("RGB", (10, 10), (255, 0, 0))
            img.save(img_path)
        return str(img_path)

    return _make


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
    val = json.loads(out.text)
    assert val["foo"] == "bar"


def test_read_yaml_file(tmp_path):
    path = tmp_path / "data.yaml"
    d = {"a": 123}
    path.write_text(yaml.safe_dump(d))

    reader = VanillaReader()
    out = reader.read(str(path))
    val = yaml.safe_load(out.text)
    assert isinstance(val, dict)
    assert val["a"] == 123
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
    path.write_text("dummy")

    reader = VanillaReader()
    out = reader.read(str(path))
    assert "x,y" in out.text
    assert out.conversion_method == "csv"


def test_excel_read_as_table(monkeypatch, dummy_xlsx_file):
    # Patch pd.read_excel to simulate table read
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)
    # Should not call _convert_office_to_pdf or pdf_reader
    reader = VanillaReader()
    out = reader.read(str(dummy_xlsx_file), as_table=True)
    assert "A,B" in out.text or "A,B\n" in out.text
    assert out.conversion_method == "xlsx"
    assert out.document_name == "test.xlsx"
    assert out.document_path.endswith("test.xlsx")


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
    # Simulate file in SUPPORTED_PROGRAMMING_LANGUAGES
    path = tmp_path / "doc.en"
    path.write_text("Hello")
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.SUPPORTED_PROGRAMMING_LANGUAGES",
        ["en", "fr"],
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
    val = json.loads(out.text)
    assert val["k"] == "v"


# ---------- file_path but actually JSON string ----------


def test_read_file_path_as_json(monkeypatch):
    s = '{"foo": 3}'
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: False)
    out = reader.read(s)
    val = json.loads(out.text)
    assert val["foo"] == 3


# ---------- file_path but actually YAML string ----------


def test_read_file_path_as_yaml(monkeypatch):
    s = "foo: bar"
    reader = VanillaReader()
    monkeypatch.setattr(reader, "is_valid_file_path", lambda p: False)
    monkeypatch.setattr(reader, "is_url", lambda p: False)
    out = reader.read(s)
    val = yaml.safe_load(out.text)
    assert val["foo"] == "bar"


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
    val = json.loads(out.text)
    assert val["x"] == "y"


# ---------- explicit json_document ----------


def test_explicit_json_document():
    reader = VanillaReader()
    out = reader.read(json_document='{"foo": "bar"}', document_name="abc")
    val = json.loads(out.text)
    assert val["foo"] == "bar"
    assert out.document_name == "abc"
    assert out.conversion_method == "json"


def test_explicit_json_document_dict():
    reader = VanillaReader()
    out = reader.read(json_document={"x": 1})
    val = json.loads(out.text)
    assert val["x"] == 1
    assert out.conversion_method == "json"


# ---------- explicit text_document ----------


def test_explicit_text_document_json():
    reader = VanillaReader()
    out = reader.read(text_document="[1,2,3]")
    val = json.loads(out.text)
    assert val == [1, 2, 3]
    assert out.conversion_method == "json"


def test_explicit_text_document_yaml():
    reader = VanillaReader()
    out = reader.read(text_document="a: 1")
    val = yaml.safe_load(out.text)
    assert val["a"] == 1
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

    reader = VanillaReader(model=DummyVisionModel())
    out = reader.read(
        str(pdf_path),
        scan_pdf_pages=True,
        resolution=300,
        vlm_parameters={"temperature": 0.0},
    )

    # → markdown with correct page separator and contents
    assert (
        "<!-- page -->" in out.text
        and "PAGE-1-MD" in out.text
        and "PAGE-2-MD" in out.text
    )
    assert out.text.count("<!-- page -->") == 2

    # → metadata fields
    assert out.conversion_method == "png"
    assert out.ocr_method == "dummy-vlm"

    # → our DummyPDFPlumberReader captured the kwargs
    pdf_reader = reader.pdf_reader  # the instance inside VanillaReader
    recorded = pdf_reader.last_kwargs
    assert recorded["resolution"] == 300
    assert recorded["model"] is reader.model
    assert DEFAULT_IMAGE_EXTRACTION_PROMPT in recorded["prompt"]


def test_pdf_custom_placeholder(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    custom_placeholder = "<!-- custom-img -->"
    reader.read(str(pdf_path), image_placeholder=custom_placeholder)
    assert reader.pdf_reader.last_kwargs["image_placeholder"] == custom_placeholder


def test_pdf_custom_placeholder_with_model(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    model = DummyVisionModel()
    reader = VanillaReader(model=model)
    custom_placeholder = "<!-- myimg -->"
    reader.read(str(pdf_path), model=model, image_placeholder=custom_placeholder)
    assert reader.pdf_reader.last_kwargs["image_placeholder"] == custom_placeholder


def test_pdf_default_placeholder(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    reader.read(str(pdf_path))
    assert reader.pdf_reader.last_kwargs["image_placeholder"] == "<!-- image -->"


@pytest.mark.parametrize(
    "pdf_text, page_placeholder, expected",
    [
        ("page 1 <!-- page --> page 2", "<!-- page -->", "<!-- page -->"),
        ("single page, no marker", "<!-- page -->", None),
        ("abc |##| xyz", "|##|", "|##|"),
        ("abc, not here", "|##|", None),
        ("", "<!-- page -->", None),
    ],
)
def test_page_placeholder_field_for_pdf(
    monkeypatch, tmp_path, pdf_text, page_placeholder, expected
):
    # Patch DummyPDFPlumberReader.read to return pdf_text
    class DummyPDF:
        def __init__(self):
            self.last_kwargs = {}

        def read(self, *a, **kw):
            self.last_kwargs = kw
            return pdf_text

        def describe_pages(self, *a, **kw):
            # Only needed for scan_pdf_pages=True, not in this test
            return []

    # Patch VanillaReader.pdf_reader to our dummy
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader", lambda: DummyPDF()
    )
    path = tmp_path / "test.pdf"
    path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader()
    out = reader.read(str(path), page_placeholder=page_placeholder)
    assert out.page_placeholder == expected


def test_page_placeholder_field_scan_pdf_pages(monkeypatch, tmp_path):
    # For scan_pdf_pages=True, we want to join with custom marker
    class DummyPDF:
        def __init__(self):
            self.last_kwargs = {}

        def read(self, *a, **kw):
            return "no split"

        def describe_pages(self, *a, **kw):
            # Simulate two page markdowns
            return ["page1", "page2"]

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader", lambda: DummyPDF()
    )
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader(model=DummyVisionModel())
    out = reader.read(str(pdf_path), scan_pdf_pages=True, page_placeholder="###PAGE###")
    assert "###PAGE###" in out.text
    assert out.page_placeholder == "###PAGE###"


def test_page_placeholder_field_scan_pdf_pages_none(monkeypatch, tmp_path):
    # If the placeholder is not in the joined text, page_placeholder should be None
    class DummyPDF:
        def __init__(self):
            self.last_kwargs = {}

        def read(self, *a, **kw):
            return "irrelevant"

        def describe_pages(self, *a, **kw):
            return ["no_marker1", "no_marker2"]

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader", lambda: DummyPDF()
    )
    pdf_path = tmp_path / "no_marker.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    reader = VanillaReader(model=DummyVisionModel())

    out = reader.read(str(pdf_path), scan_pdf_pages=True, page_placeholder="%%PAGE%%")
    assert out.page_placeholder is None


# ---------- Office processing tests ----------


def test_office_docx_to_pdf(monkeypatch, dummy_docx_file, tmp_path):
    # Patch _convert_office_to_pdf and pdf_reader.read
    def fake_convert(file_path):
        fake_pdf = tmp_path / "converted.pdf"
        fake_pdf.write_text("FAKE_PDF")
        return str(fake_pdf)

    class DummyPDFReader:
        def read(self, pdf_path, **kw):
            assert pdf_path.endswith("converted.pdf")
            return "FAKE_PDF_TEXT"

    monkeypatch.setattr(
        VanillaReader, "_convert_office_to_pdf", staticmethod(fake_convert)
    )
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFReader(),
    )

    reader = VanillaReader()
    out = reader.read(str(dummy_docx_file))
    assert isinstance(out, ReaderOutput)
    assert out.text == "FAKE_PDF_TEXT"
    assert out.document_name == "converted.pdf"
    assert out.document_path.endswith("converted.pdf")
    assert out.conversion_method == "pdf"


def test_office_pptx_to_pdf(monkeypatch, dummy_pptx_file, tmp_path):
    def fake_convert(file_path):
        fake_pdf = tmp_path / "slide.pdf"
        fake_pdf.write_text("PPT2PDF")
        return str(fake_pdf)

    class DummyPDFReader:
        def read(self, pdf_path, **kw):
            assert pdf_path.endswith("slide.pdf")
            return "SLIDES_AS_PDF"

    monkeypatch.setattr(
        VanillaReader, "_convert_office_to_pdf", staticmethod(fake_convert)
    )
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFReader(),
    )
    reader = VanillaReader()
    out = reader.read(str(dummy_pptx_file))
    assert out.text == "SLIDES_AS_PDF"
    assert out.document_name == "slide.pdf"
    assert out.document_path.endswith("slide.pdf")
    assert out.conversion_method == "pdf"


def test_excel_to_pdf_default(monkeypatch, dummy_xlsx_file, tmp_path):
    # as_table=False, so convert to PDF and run through pdf_reader.read
    def fake_convert(file_path):
        fake_pdf = tmp_path / "excel_as_pdf.pdf"
        fake_pdf.write_text("EXCELPDF")
        return str(fake_pdf)

    class DummyPDFReader:
        def read(self, pdf_path, **kw):
            assert pdf_path.endswith("excel_as_pdf.pdf")
            return "EXCEL_AS_PDF_TEXT"

    monkeypatch.setattr(
        VanillaReader, "_convert_office_to_pdf", staticmethod(fake_convert)
    )
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFReader(),
    )
    reader = VanillaReader()
    out = reader.read(str(dummy_xlsx_file))
    assert out.text == "EXCEL_AS_PDF_TEXT"
    assert out.document_name == "excel_as_pdf.pdf"
    assert out.document_path.endswith("excel_as_pdf.pdf")
    assert out.conversion_method == "pdf"


def test_excel_as_table_and_pdf(monkeypatch, dummy_xlsx_file, tmp_path):
    # Sanity: as_table=False triggers PDF, as_table=True triggers pandas/CSV
    def fake_convert(file_path):
        fake_pdf = tmp_path / "xls2pdf.pdf"
        fake_pdf.write_text("FOO")
        return str(fake_pdf)

    class DummyPDFReader:
        def read(self, pdf_path, **kw):
            assert pdf_path.endswith("xls2pdf.pdf")
            return "EXCEL2PDF"

    # Case: as_table=False
    monkeypatch.setattr(
        VanillaReader, "_convert_office_to_pdf", staticmethod(fake_convert)
    )
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader",
        lambda: DummyPDFReader(),
    )
    reader = VanillaReader()
    out = reader.read(str(dummy_xlsx_file), as_table=False)
    assert out.text == "EXCEL2PDF"
    assert out.document_name == "xls2pdf.pdf"
    assert out.conversion_method == "pdf"
    # Case: as_table=True
    df = pd.DataFrame({"C": [5]})
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: df)
    out2 = reader.read(str(dummy_xlsx_file), as_table=True)
    assert "C" in out2.text
    assert out2.conversion_method == "xlsx"


# ---------- Image handling tests ----------


@pytest.mark.parametrize("ext", ["png", "jpg", "jpeg"])
def test_image_file_handling(create_image, ext):
    img_path = create_image(ext)
    reader = VanillaReader(model=DummyVisionModel())
    out = reader.read(file_path=img_path)
    # The output text should be from DummyVisionModel, starting with EXTRACTED_TEXT:
    assert out.text.startswith("EXTRACTED_TEXT:")
    assert out.document_name.endswith(f".{ext}")
    assert out.document_path.endswith(f".{ext}")
    assert out.conversion_method == "image"
    assert out.ocr_method == "dummy-vlm"


def test_image_file_with_custom_prompt(create_image):
    img_path = create_image("png")
    prompt = "Describe this image"
    reader = VanillaReader(model=DummyVisionModel())
    out = reader.read(file_path=img_path, prompt=prompt)
    assert prompt in out.text  # Dummy returns prompt in result


def test_image_file_no_model(create_image):
    img_path = create_image("jpg")
    reader = VanillaReader(model=None)
    with pytest.raises(ValueError):
        reader.read(file_path=img_path)


def test_image_file_base64_encoding(create_image):
    img_path = create_image("jpeg")

    class CheckBase64Model(DummyVisionModel):
        def analyze_content(self, file, prompt=None, **kwargs):
            import base64

            # Should be a base64 string
            try:
                base64.b64decode(file)
                return "BASE64_OK"
            except Exception:
                return "BASE64_FAIL"

    reader = VanillaReader(model=CheckBase64Model())
    out = reader.read(file_path=img_path)
    assert out.text == "BASE64_OK"


def test_image_file_unsupported_ext(create_image):
    img_path = create_image("svg")  # Only PNG/JPG/JPEG supported
    reader = VanillaReader(model=DummyVisionModel())
    with pytest.raises(ValueError):
        reader.read(file_path=img_path)


def test_image_ext_custom_supported(monkeypatch, create_image):
    img_path = create_image("bmp")
    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.SUPPORTED_VANILLA_IMAGE_EXTENSIONS",
        {"bmp"},
    )
    reader = VanillaReader(model=DummyVisionModel())
    out = reader.read(file_path=img_path)
    assert out.conversion_method == "image"


def test_init_assigns_model_and_pdf_reader(monkeypatch):
    class DummyPDF:
        pass

    monkeypatch.setattr(
        "splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader", lambda: DummyPDF()
    )
    m = object()
    reader = VanillaReader(model=m)
    assert reader.model is m
    assert isinstance(reader.pdf_reader, DummyPDF)


def test_dispatch_source_invalid_key():
    reader = VanillaReader()
    with pytest.raises(ValueError, match="Unrecognized document source"):
        reader._dispatch_source("not_a_real_source", "x", {})


def test_handle_local_path_rejects_non_string():
    reader = VanillaReader()
    with pytest.raises(ValueError, match="file_path must be a string or Path object"):
        reader._handle_local_path(123, {})


def test_handle_fallback_raw_fallback(monkeypatch):
    reader = VanillaReader()
    # Patch JSON/text to always fail, so fallback hits last branch
    monkeypatch.setattr(
        reader,
        "_handle_explicit_json",
        lambda r, k: (_ for _ in ()).throw(Exception("fail")),
    )
    monkeypatch.setattr(
        reader,
        "_handle_explicit_text",
        lambda r, k: (_ for _ in ()).throw(Exception("fail")),
    )
    result = reader._handle_fallback("raw-stuff", {"document_name": "docname"})
    assert result == ("docname", None, "raw-stuff", "txt", None)


def test_handle_image_to_llm_raises_if_no_model(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"foo")
    reader = VanillaReader(model=None)
    with pytest.raises(ValueError, match="No vision model provided"):
        reader._handle_image_to_llm(None, str(img))


def test_scan_pdf_pages_returns_joined_text(tmp_path):
    class DummyModel:
        model_name = "x"

    class DummyPDFPlumber:
        def describe_pages(self, file_path, model, prompt, resolution=300, **kw):
            return ["page-a", "page-b"]

    reader = VanillaReader(model=DummyModel())
    reader.pdf_reader = DummyPDFPlumber()
    out = reader._scan_pdf_pages(
        str(tmp_path / "file.pdf"), model=DummyModel(), page_placeholder="ZPAGEZ"
    )
    assert out.count("ZPAGEZ") == 2
    assert "page-a" in out and "page-b" in out


def test_surface_page_placeholder_excludes_percent():
    out = VanillaReader._surface_page_placeholder(
        scan=True, placeholder="%foo%", text="hello"
    )
    assert out is None
    out2 = VanillaReader._surface_page_placeholder(
        scan=False, placeholder="PAGE", text="PAGE in text"
    )
    assert out2 == "PAGE"
    out3 = VanillaReader._surface_page_placeholder(
        scan=False, placeholder="NOTIN", text="absent"
    )
    assert out3 is None


def test_convert_office_to_pdf_missing_soffice(monkeypatch, tmp_path):
    # Simulate soffice missing
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    reader = VanillaReader()
    with pytest.raises(RuntimeError, match="LibreOffice/soffice is required"):
        reader._convert_office_to_pdf(str(tmp_path / "file.docx"))


def test_convert_office_to_pdf_subprocess_fail(monkeypatch, tmp_path):
    # Simulate soffice present but fails
    monkeypatch.setattr("shutil.which", lambda cmd: True)

    class DummyProc:
        returncode = 1
        stderr = b"failed!"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: DummyProc())
    reader = VanillaReader()
    with pytest.raises(RuntimeError, match="LibreOffice failed converting"):
        reader._convert_office_to_pdf(str(tmp_path / "file.docx"))


def test_convert_office_to_pdf_pdf_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda cmd: True)

    class DummyProc:
        returncode = 0
        stderr = b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: DummyProc())
    reader = VanillaReader()
    # output PDF does not exist
    with pytest.raises(RuntimeError, match="Expected PDF not found"):
        reader._convert_office_to_pdf(str(tmp_path / "notthere.docx"))


def test_read_text_file_yaml(tmp_path):
    f = tmp_path / "file.yaml"
    f.write_text("k: v")
    out = _read_text_file(str(f), "yaml")
    assert "k: v" in out


def test_read_text_file_plain(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("foobar")
    out = _read_text_file(str(f), "txt")
    assert out == "foobar"


def test_read_parquet_importerror(monkeypatch, tmp_path):
    # Simulate ImportError in pandas
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no pyarrow")),
    )
    f = tmp_path / "f.parquet"
    f.write_text("x")
    with pytest.raises(ImportError):
        _read_parquet(str(f))


def test_read_parquet_valueerror(monkeypatch, tmp_path):
    # Simulate ValueError in pandas
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad file")),
    )
    f = tmp_path / "g.parquet"
    f.write_text("y")
    with pytest.raises(ValueError):
        _read_parquet(str(f))


def test_read_excel_importerror(monkeypatch, tmp_path):
    # Simulate ImportError in pandas
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no openpyxl")),
    )
    f = tmp_path / "f.xlsx"
    f.write_text("z")
    with pytest.raises(ImportError):
        _read_excel(str(f))


def test_read_excel_valueerror(monkeypatch, tmp_path):
    # Simulate ValueError in pandas
    monkeypatch.setattr(
        pd, "read_excel", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad excel"))
    )
    f = tmp_path / "f.xlsx"
    f.write_text("x")
    with pytest.raises(ValueError):
        _read_excel(str(f))
