from unittest.mock import MagicMock, patch

from splitter_mr.reader import DoclingReader

# Test cases


def test_docling_reader_reads_and_converts(tmp_path):
    # Simulate a supported file type (e.g., PDF)
    file = tmp_path / "foo.pdf"
    file.write_text("some pdf-like content")

    with patch(
        "splitter_mr.reader.readers.docling_reader.DocumentConverter"
    ) as MockConverter:
        mock_converter = MockConverter.return_value
        # Fake docling result structure: .convert().document.export_to_markdown()
        mock_document = MagicMock()
        mock_document.export_to_markdown.return_value = "# Heading\nSome text"
        mock_converter.convert.return_value.document = mock_document

        reader = DoclingReader()
        result = reader.read(str(file), document_id="doc-42", metadata={"src": "test"})

        # Check correct call chain
        mock_converter.convert.assert_called_once_with(str(file))
        mock_document.export_to_markdown.assert_called_once()

        # Validate returned dict
        assert result.text == "# Heading\nSome text"
        assert result.document_name == "foo.pdf"
        assert result.document_path == str(file)
        assert result.document_id == "doc-42"
        assert result.conversion_method == "markdown"
        assert result.metadata == {"src": "test"}


def test_docling_reader_txt_to_md(tmp_path):
    # Simulate reading a .txt file (unsupported by DoclingReader)
    txt_file = tmp_path / "foo.txt"
    txt_file.write_text("plain text content")

    with patch(
        "splitter_mr.reader.readers.docling_reader.VanillaReader"
    ) as MockVanillaReader:
        mock_vanilla = MockVanillaReader.return_value
        # Simulate the ReaderOutput object returned
        mock_reader_output = MagicMock(
            document_name="foo.txt", text="plain text content", document_id="id-txt"
        )
        mock_vanilla.read.return_value = mock_reader_output

        reader = DoclingReader()
        result = reader.read(str(txt_file), document_id="id-txt")

        # Should use VanillaReader for .txt
        mock_vanilla.read.assert_called_once_with(
            file_path=str(txt_file), document_id="id-txt"
        )
        assert result.document_name == "foo.txt"
        assert result.text == "plain text content"
        assert result.document_id == "id-txt"


def test_docling_reader_defaults(tmp_path):
    # Should work even if kwargs are not provided
    file = tmp_path / "bar.docx"
    file.write_text("dummy docx content")

    with patch(
        "splitter_mr.reader.readers.docling_reader.DocumentConverter"
    ) as MockConverter:
        mock_converter = MockConverter.return_value
        mock_document = MagicMock()
        mock_document.export_to_markdown.return_value = "# DocX"
        mock_converter.convert.return_value.document = mock_document

        reader = DoclingReader()
        result = reader.read(str(file))
        assert result.document_name == "bar.docx"
        assert result.conversion_method == "markdown"
        assert hasattr(result, "document_id")
        assert hasattr(result, "metadata")
