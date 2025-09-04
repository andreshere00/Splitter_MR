import io
import os
from unittest.mock import ANY, MagicMock, patch

import pytest

from splitter_mr.model.base_model import BaseVisionModel
from splitter_mr.reader.readers.markitdown_reader import MarkItDownReader

# Helpers


@pytest.fixture
def mock_split_pdfs(tmp_path):
    """Fixture to patch _split_pdf_to_temp_pdfs and create dummy temp PDF files."""

    def _make(n_pages=3):
        temp_files = []
        for i in range(n_pages):
            temp_file = tmp_path / f"temp_page_{i}.pdf"
            temp_file.write_text(f"Fake content page {i}")
            temp_files.append(str(temp_file))
        return temp_files

    return _make


class FakeOpenAI:
    pass  # You only need to pass isinstance(client, OpenAI) in your code


class FakeVisionModel(BaseVisionModel):
    def __init__(self, model_name="gpt-4o-vision"):
        self.model_name = model_name

    def get_client(self):
        return FakeOpenAI()  # Returns an "OpenAI" client!

    def analyze_content(self, prompt, file, file_ext, **kwargs):
        return "dummy"


def patch_vision_models():
    """
    Returns (patch_BaseVisionModel, DummyVisionModel).
    """

    base = "splitter_mr.reader.readers.markitdown_reader"
    return (
        patch(f"{base}.BaseVisionModel", FakeVisionModel),
        FakeVisionModel,
    )


# Test cases


def patch_pdf_pages(pages=1):
    pixmap = MagicMock()
    pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\nfakepng"
    page = MagicMock()
    page.get_pixmap.return_value = pixmap
    pdf_doc = MagicMock()
    pdf_doc.__len__.return_value = pages
    pdf_doc.load_page.return_value = page
    return patch(
        "splitter_mr.reader.readers.markitdown_reader.fitz.open", return_value=pdf_doc
    )


def test_markitdown_reader_reads_and_converts(tmp_path):
    test_file = tmp_path / "foo.pdf"
    test_file.write_text("fake pdf content")
    with patch(
        "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
    ) as MockMarkItDown:
        mock_md = MockMarkItDown.return_value
        mock_md.convert.return_value = MagicMock(
            text_content="# Converted Markdown!\nSome text."
        )
        reader = MarkItDownReader()
        result = reader.read(
            str(test_file), document_id="doc-1", metadata={"source": "unit test"}
        )
        mock_md.convert.assert_called_once_with(str(test_file), llm_prompt=ANY)
        assert result.text == "# Converted Markdown!\nSome text."
        assert result.document_name == "foo.pdf"
        assert result.document_path == str(test_file)
        assert result.document_id == "doc-1"
        assert result.conversion_method == "markdown"
        assert result.metadata == {"source": "unit test"}
        assert result.reader_method == "markitdown"


def test_markitdown_reader_defaults(tmp_path):
    test_file = tmp_path / "bar.docx"
    test_file.write_text("dummy docx")
    with patch(
        "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
    ) as MockMarkItDown:
        mock_md = MockMarkItDown.return_value
        mock_md.convert.return_value = MagicMock(text_content="## Dummy MD")
        reader = MarkItDownReader()
        result = reader.read(str(test_file))
        assert result.document_name == "bar.docx"
        assert result.conversion_method == "markdown"
        assert result.ocr_method is None
        assert hasattr(result, "document_id")
        assert hasattr(result, "metadata")


@patch("splitter_mr.reader.readers.markitdown_reader.OpenAI", FakeOpenAI)
def test_scan_pdf_pages_calls_convert_per_page(tmp_path):
    pdf = tmp_path / "multi.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=3),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
    ):
        reader = MarkItDownReader(model=DummyVisionModel())
        MockMID.return_value.convert.return_value = MagicMock(text_content="## page-md")
        result = reader.read(str(pdf), scan_pdf_pages=True)
        assert MockMID.return_value.convert.call_count == 3
        assert "<!-- page -->" in result.text
        assert result.conversion_method == "markdown"
        for call in MockMID.return_value.convert.call_args_list:
            assert "llm_prompt" in call.kwargs


@patch("splitter_mr.reader.readers.markitdown_reader.OpenAI", FakeOpenAI)
def test_scan_pdf_pages_uses_custom_prompt(tmp_path):
    pdf = tmp_path / "single.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=1),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
    ):
        reader = MarkItDownReader(model=DummyVisionModel())
        MockMID.return_value.convert.return_value = MagicMock(text_content="foo")
        custom_prompt = "My **special** OCR prompt"
        reader.read(str(pdf), scan_pdf_pages=True, prompt=custom_prompt)
        _, kwargs = MockMID.return_value.convert.call_args
        assert kwargs["llm_prompt"] == custom_prompt


@patch("splitter_mr.reader.readers.markitdown_reader.OpenAI", FakeOpenAI)
def test_scan_pdf_pages_splits_each_page(tmp_path):
    """Test PDF is split and scanned page by page with VisionModel."""
    pdf = tmp_path / "multi.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=3),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
    ):
        reader = MarkItDownReader(model=DummyVisionModel())
        # Simulate each page conversion returning "PAGE-MD"
        MockMID.return_value.convert.side_effect = [
            MagicMock(text_content="PAGE-MD"),
            MagicMock(text_content="PAGE-MD"),
            MagicMock(text_content="PAGE-MD"),
        ]
        result = reader.read(str(pdf), scan_pdf_pages=True)
        # Should call convert 3 times (one for each page)
        assert MockMID.return_value.convert.call_count == 3
        # Output contains all pages and the correct headings
        assert "<!-- page -->" in result.text
        assert "PAGE-MD" in result.text
        # Metadata should reflect scan mode
        assert result.conversion_method == "markdown"
        assert result.ocr_method == "gpt-4o-vision"


@patch("splitter_mr.reader.readers.markitdown_reader.OpenAI", FakeOpenAI)
def test_scan_pdf_pages_custom_prompt(tmp_path):
    """Test that a custom prompt is passed for page scanning."""
    pdf = tmp_path / "onepage.pdf"
    pdf.write_text("pdf")
    patch_oa, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=1),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
    ):
        MockMID.return_value.convert.return_value = MagicMock(text_content="CUSTOM")
        reader = MarkItDownReader(model=DummyVisionModel())
        custom_prompt = "Describe this page in detail."
        reader.read(str(pdf), scan_pdf_pages=True, prompt=custom_prompt)
        # Should pass prompt to convert
        args, kwargs = MockMID.return_value.convert.call_args
        assert kwargs["llm_prompt"] == custom_prompt


@pytest.mark.parametrize(
    "md_text, page_placeholder, expected",
    [
        ("text <!-- page --> more", "<!-- page -->", "<!-- page -->"),
        # etc...
    ],
)
def test_page_placeholder_field(
    monkeypatch, tmp_path, md_text, page_placeholder, expected
):
    class DummyVisionModel:
        model_name = "gpt-4o-vision"

        def get_client(self):
            return None

    file_path = tmp_path / "doc.pdf"
    file_path.write_text("fake pdf")

    monkeypatch.setattr(
        MarkItDownReader,
        "_pdf_pages_to_markdown",
        lambda self, file_path, md, prompt, page_placeholder: md_text,
    )
    monkeypatch.setattr(
        MarkItDownReader, "_get_markitdown", lambda self: (None, "gpt-4o-vision")
    )

    reader = MarkItDownReader(model=DummyVisionModel())
    out = reader.read(
        str(file_path), page_placeholder=page_placeholder, scan_pdf_pages=True
    )
    assert out.page_placeholder == expected


def test_page_placeholder_field_no_scan(monkeypatch, tmp_path):
    file_path = tmp_path / "plain.txt"
    file_path.write_text("irrelevant")

    # Patch MarkItDown.convert to control output when scan_pdf_pages is False (default)
    class DummyMD:
        def convert(self, file_path, llm_prompt=None):
            class Result:
                text_content = "foo <!-- page --> bar"

            return Result()

    monkeypatch.setattr(
        "splitter_mr.reader.readers.markitdown_reader.MarkItDown",
        lambda *a, **kw: DummyMD(),
    )
    reader = MarkItDownReader()
    out = reader.read(str(file_path))
    # The placeholder appears, so should be picked up
    assert out.page_placeholder == "<!-- page -->"


def test_page_placeholder_absent_no_scan(monkeypatch, tmp_path):
    file_path = tmp_path / "plain.txt"
    file_path.write_text("irrelevant")

    # Patch MarkItDown.convert to output something without placeholder
    class DummyMD:
        def convert(self, file_path, llm_prompt=None):
            class Result:
                text_content = "something else"

            return Result()

    monkeypatch.setattr(
        "splitter_mr.reader.readers.markitdown_reader.MarkItDown",
        lambda *a, **kw: DummyMD(),
    )
    reader = MarkItDownReader()
    out = reader.read(str(file_path))
    assert out.page_placeholder is None


def test_split_by_pages_no_model_single_page(tmp_path, mock_split_pdfs):
    pdf = tmp_path / "single.pdf"
    pdf.write_text("pdf one page")
    temp_files = mock_split_pdfs(1)
    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove") as mock_remove,
    ):
        MockMarkItDown.return_value.convert.return_value = MagicMock(
            text_content="Just one page"
        )
        reader = MarkItDownReader()
        result = reader.read(
            str(pdf), split_by_pages=True, page_placeholder="<<{page}>>"
        )
        assert MockMarkItDown.return_value.convert.call_count == 1
        assert "<<1>>" in result.text
        assert "Just one page" in result.text
        mock_remove.assert_called_once_with(temp_files[0])


def test_split_by_pages_placeholder_detected(tmp_path, mock_split_pdfs):
    pdf = tmp_path / "ph.pdf"
    pdf.write_text("pdf content")
    temp_files = mock_split_pdfs(1)
    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove"),
    ):
        MockMarkItDown.return_value.convert.return_value = MagicMock(
            text_content="AA <!--pagebreak--> BB"
        )
        reader = MarkItDownReader()
        result = reader.read(
            str(pdf), split_by_pages=True, page_placeholder="<!--pagebreak-->"
        )
        assert result.page_placeholder == "<!--pagebreak-->"
        assert "AA <!--pagebreak--> BB" in result.text


def test_split_by_pages_no_model_multiple_pages(tmp_path, mock_split_pdfs):
    file = tmp_path / "docx_file.docx"
    file.write_text("fake docx content")
    temp_files = mock_split_pdfs(3)
    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._convert_to_pdf",
            return_value="dummy.pdf",
        ) as mock_convert,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove") as mock_remove,
    ):
        mock_md = MockMarkItDown.return_value
        mock_md.convert.side_effect = [
            MagicMock(text_content="Page ONE"),
            MagicMock(text_content="Page TWO"),
            MagicMock(text_content="Page THREE"),
        ]
        reader = MarkItDownReader()
        result = reader.read(
            str(file), split_by_pages=True, page_placeholder="<p {page}>"
        )
        assert mock_convert.called
        assert mock_md.convert.call_count == 3
        for f in temp_files:
            mock_md.convert.assert_any_call(f, llm_prompt=ANY)
            mock_remove.assert_any_call(f)
        assert "<p 1>" in result.text
        assert "<p 2>" in result.text
        assert "<p 3>" in result.text
        assert all(
            page in result.text for page in ["Page ONE", "Page TWO", "Page THREE"]
        )
        assert result.ocr_method is None
        assert (
            result.conversion_method == "markdown"
        )  # could be "pdf" if you wish to reflect the conversion


def test_split_by_pages_ignored_for_txt(tmp_path):
    txt_file = tmp_path / "x.txt"
    txt_file.write_text("fake text")
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_text("fake pdf")
    temp_files = [str(dummy_pdf)]

    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._convert_to_pdf",
            return_value=str(dummy_pdf),
        ) as mock_convert,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove"),
    ):
        MockMarkItDown.return_value.convert.return_value = MagicMock(
            text_content="reg text"
        )
        reader = MarkItDownReader()
        result = reader.read(str(txt_file), split_by_pages=True)

        # _convert_to_pdf should **not** be invoked for .txt
        mock_convert.assert_not_called()

        # convert() should receive the dummy PDF path returned by _split_pdf_to_temp_pdfs
        MockMarkItDown.return_value.convert.assert_called_once_with(
            str(dummy_pdf), llm_prompt=ANY
        )

        assert result.text.startswith("<!-- page -->")
        assert result.text.endswith("reg text")


@pytest.mark.parametrize("ext", ["docx", "pptx", "xlsx"])
def test_split_by_pages_convertible_extensions(tmp_path, ext):
    doc_file = tmp_path / f"doc_file.{ext}"
    doc_file.write_text("fake content")
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_text("fake pdf")
    temp_files = [str(dummy_pdf)]

    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._convert_to_pdf",
            return_value=str(dummy_pdf),
        ) as mock_convert,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove"),
    ):
        MockMarkItDown.return_value.convert.return_value = MagicMock(
            text_content="converted page"
        )
        reader = MarkItDownReader()
        result = reader.read(str(doc_file), split_by_pages=True)

        mock_convert.assert_called_once_with(str(doc_file))

        MockMarkItDown.return_value.convert.assert_called_once_with(
            str(dummy_pdf), llm_prompt=ANY
        )
        assert "<!-- page -->" in result.text
        assert "converted page" in result.text


def test_split_by_pages_placeholder_not_detected(tmp_path, mock_split_pdfs):
    file = tmp_path / "ph2.pptx"
    file.write_text("content")
    temp_files = mock_split_pdfs(1)

    with (
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
        ) as MockMarkItDown,
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._convert_to_pdf",
            return_value="dummy.pdf",
        ),
        patch(
            "splitter_mr.reader.readers.markitdown_reader.MarkItDownReader._split_pdf_to_temp_pdfs",
            return_value=temp_files,
        ),
        patch("os.remove"),
    ):
        MockMarkItDown.return_value.convert.return_value = MagicMock(
            text_content="AA BB CC"
        )
        reader = MarkItDownReader()
        result = reader.read(
            str(file), split_by_pages=True, page_placeholder="<!--pagebreak-->"
        )

        assert result.page_placeholder == "<!--pagebreak-->"


def test_convert_to_pdf_raises_if_soffice_missing(tmp_path, monkeypatch):
    test_file = tmp_path / "dummy.docx"
    test_file.write_text("fake")

    # Patch shutil.which to simulate "soffice" missing
    monkeypatch.setattr("shutil.which", lambda _: None)
    reader = MarkItDownReader()
    with pytest.raises(RuntimeError) as exc:
        reader._convert_to_pdf(str(test_file))
    assert "LibreOffice" in str(exc.value)


def test_convert_to_pdf_raises_on_subprocess_error(tmp_path, monkeypatch):
    test_file = tmp_path / "fail.docx"
    test_file.write_text("fake")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/soffice")
    monkeypatch.setattr("tempfile.mkdtemp", lambda: str(tmp_path))

    # Patch subprocess.run to return failure
    class DummyResult:
        returncode = 1
        stderr = b"conversion failed"

    monkeypatch.setattr("subprocess.run", lambda *a, **k: DummyResult())
    reader = MarkItDownReader()
    with pytest.raises(RuntimeError) as exc:
        reader._convert_to_pdf(str(test_file))
    assert "Failed to convert" in str(exc.value)


def test_convert_to_pdf_raises_when_pdf_not_created(tmp_path, monkeypatch):
    test_file = tmp_path / "fail.docx"
    test_file.write_text("fake")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/soffice")
    monkeypatch.setattr("tempfile.mkdtemp", lambda: str(tmp_path))

    class DummyResult:
        returncode = 0
        stderr = b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: DummyResult())
    reader = MarkItDownReader()
    with pytest.raises(RuntimeError) as exc:
        reader._convert_to_pdf(str(test_file))
    assert "PDF was not created" in str(exc.value)


def test_get_markitdown_raises_on_wrong_model():
    class InvalidVisionModel(BaseVisionModel):
        def __init__(self, model_name="bad"):
            self.model_name = model_name

        def get_client(self):
            return object()  # <-- Not an OpenAI instance!

        def analyze_content(self, prompt, file, file_ext, **kwargs):
            return "dummy"

    reader = MarkItDownReader(model=InvalidVisionModel())
    with pytest.raises(ValueError) as exc:
        reader._get_markitdown()
    assert "Incompatible client" in str(exc.value)


def test_split_pdf_to_temp_pdfs(tmp_path):
    import pypdf

    # Create a real 2-page PDF
    pdf_path = tmp_path / "test.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    reader = MarkItDownReader()
    files = reader._split_pdf_to_temp_pdfs(str(pdf_path))
    assert len(files) == 2
    for f in files:
        assert f.endswith(".pdf")
        assert os.path.exists(f)
        os.remove(f)


def test_read_sets_conversion_method_json(tmp_path):
    file = tmp_path / "foo.json"
    file.write_text("{}")
    with patch(
        "splitter_mr.reader.readers.markitdown_reader.MarkItDown"
    ) as MockMarkItDown:
        mock_md = MockMarkItDown.return_value
        mock_md.convert.return_value = MagicMock(text_content='{"text":"hi"}')
        reader = MarkItDownReader()
        result = reader.read(str(file))
        assert result.conversion_method == "json"
        assert result.text == '{"text":"hi"}'


def test_pdf_pages_to_streams(tmp_path, monkeypatch):
    pdf_path = tmp_path / "img.pdf"
    # Simulate fitz.open and pixmap
    dummy_doc = MagicMock()
    dummy_doc.__len__.return_value = 2
    dummy_page = MagicMock()
    dummy_pixmap = MagicMock()
    dummy_pixmap.tobytes.return_value = b"FAKEPNG"
    dummy_page.get_pixmap.return_value = dummy_pixmap
    dummy_doc.load_page.return_value = dummy_page
    monkeypatch.setattr("fitz.open", lambda _: dummy_doc)
    reader = MarkItDownReader()
    streams = reader._pdf_pages_to_streams(str(pdf_path))
    assert len(streams) == 2
    for i, stream in enumerate(streams, 1):
        assert isinstance(stream, io.BytesIO)
        assert stream.name == f"page_{i}.png"
