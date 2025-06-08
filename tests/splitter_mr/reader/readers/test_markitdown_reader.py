from unittest.mock import MagicMock, patch

from splitter_mr.reader import MarkItDownReader

# Test cases


def test_markitdown_reader_reads_and_converts(tmp_path):
    # Create a dummy file to simulate input
    test_file = tmp_path / "foo.pdf"
    test_file.write_text("fake pdf content")

    # Patch MarkItDown and its convert() method
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

        # Check MarkItDown is called with correct file
        mock_md.convert.assert_called_once_with(str(test_file))

        # Validate returned object fields
        assert result.text == "# Converted Markdown!\nSome text."
        assert result.document_name == "foo.pdf"
        assert result.document_path == str(test_file)
        assert result.document_id == "doc-1"
        assert result.conversion_method == "markdown"
        assert result.metadata == {"source": "unit test"}
        assert result.reader_method == "markitdown"


def test_markitdown_reader_defaults(tmp_path):
    # Check that missing optional kwargs work as expected
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
