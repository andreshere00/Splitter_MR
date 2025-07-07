from unittest.mock import ANY, MagicMock, patch

import pytest

from splitter_mr.reader import MarkItDownReader

# Helpers


def patch_vision_models():
    """
    Returns (patch_OpenAIVisionModel, patch_AzureOpenAIVisionModel, DummyVisionModel).
    """

    class DummyVisionModel:
        def __init__(self):
            self.model_name = "gpt-4o-vision"

        def get_client(self):
            return MagicMock()

    base = "splitter_mr.reader.readers.markitdown_reader"
    return (
        patch(f"{base}.OpenAIVisionModel", DummyVisionModel),
        patch(f"{base}.AzureOpenAIVisionModel", DummyVisionModel),
        DummyVisionModel,
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


def test_scan_pdf_pages_calls_convert_per_page(tmp_path):
    pdf = tmp_path / "multi.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, patch_az, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=3),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
        patch_az,
    ):
        reader = MarkItDownReader(model=DummyVisionModel())
        MockMID.return_value.convert.return_value = MagicMock(text_content="## page-md")
        result = reader.read(str(pdf), scan_pdf_pages=True)
        assert MockMID.return_value.convert.call_count == 3
        assert "<!-- page -->" in result.text
        assert result.conversion_method == "markdown"
        for call in MockMID.return_value.convert.call_args_list:
            assert "llm_prompt" in call.kwargs


def test_scan_pdf_pages_uses_custom_prompt(tmp_path):
    pdf = tmp_path / "single.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, patch_az, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=1),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
        patch_az,
    ):
        reader = MarkItDownReader(model=DummyVisionModel())
        MockMID.return_value.convert.return_value = MagicMock(text_content="foo")
        custom_prompt = "My **special** OCR prompt"
        reader.read(str(pdf), scan_pdf_pages=True, prompt=custom_prompt)
        _, kwargs = MockMID.return_value.convert.call_args
        assert kwargs["llm_prompt"] == custom_prompt


def test_scan_pdf_pages_splits_each_page(tmp_path):
    """Test PDF is split and scanned page by page with VisionModel."""
    pdf = tmp_path / "multi.pdf"
    pdf.write_text("dummy pdf")
    patch_oa, patch_az, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=3),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
        patch_az,
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


def test_scan_pdf_pages_custom_prompt(tmp_path):
    """Test that a custom prompt is passed for page scanning."""
    pdf = tmp_path / "onepage.pdf"
    pdf.write_text("pdf")
    patch_oa, patch_az, DummyVisionModel = patch_vision_models()
    with (
        patch_pdf_pages(pages=1),
        patch("splitter_mr.reader.readers.markitdown_reader.MarkItDown") as MockMID,
        patch_oa,
        patch_az,
    ):
        MockMID.return_value.convert.return_value = MagicMock(text_content="CUSTOM")
        reader = MarkItDownReader(model=DummyVisionModel())
        custom_prompt = "Describe this page in detail."
        reader.read(str(pdf), scan_pdf_pages=True, prompt=custom_prompt)
        # Should pass prompt to convert
        args, kwargs = MockMID.return_value.convert.call_args
        assert kwargs["llm_prompt"] == custom_prompt
