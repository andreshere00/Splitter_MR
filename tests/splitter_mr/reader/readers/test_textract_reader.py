from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("botocore")
pytest.importorskip("PIL")

from botocore.exceptions import ClientError, NoCredentialsError

from splitter_mr.reader.readers.textract_reader import TextractReader
from splitter_mr.schema import (
    DEFAULT_PAGE_PLACEHOLDER,
    TEXTRACT_OCR_METHOD,
    TEXTRACT_SYNC_MAX_BYTES,
    ReaderConfigException,
    ReaderOutput,
    TextractReaderException,
)

# ---- Mocks, fixtures & helpers ---- #


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.detect_document_text.return_value = {
        "Blocks": [
            {
                "Id": "page-1",
                "BlockType": "PAGE",
                "Relationships": [{"Type": "CHILD", "Ids": ["line-1"]}],
            },
            {"Id": "line-1", "BlockType": "LINE", "Text": "Hello world"},
        ]
    }
    return client


@pytest.fixture
def reader(mock_client: MagicMock) -> TextractReader:
    return TextractReader(client=mock_client)


def _textract_blocks(*lines: str) -> dict:
    line_blocks = [
        {"Id": f"line-{idx}", "BlockType": "LINE", "Text": line}
        for idx, line in enumerate(lines, start=1)
    ]
    return {
        "Blocks": [
            {
                "Id": "page-1",
                "BlockType": "PAGE",
                "Relationships": [
                    {"Type": "CHILD", "Ids": [block["Id"] for block in line_blocks]}
                ],
            },
            *line_blocks,
        ]
    }


# ---- Happy path ---- #


def test_read_txt_delegates_to_vanilla_and_sets_textract_identity(
    reader: TextractReader, tmp_path
):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("plain text", encoding="utf-8")

    with patch(
        "splitter_mr.reader.readers.textract_reader.VanillaReader.read",
        return_value=ReaderOutput(
            text="plain text",
            document_name="notes.txt",
            document_path=str(txt_file),
            conversion_method="txt",
            reader_method="vanilla",
        ),
    ) as mock_read:
        output = reader.read(str(txt_file), metadata={"source": "unit"})

    mock_read.assert_called_once()
    assert output.text == "plain text"
    assert output.reader_method == "textract"
    assert output.ocr_method is None
    assert output.conversion_method == "txt"
    assert output.metadata == {"source": "unit"}


def test_read_pdf_rasterizes_pages_and_calls_textract_per_page(
    reader: TextractReader, mock_client: MagicMock, tmp_path
):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    with patch.object(
        TextractReader,
        "_pdf_to_png_pages",
        return_value=[b"page-one", b"page-two"],
    ) as mock_raster:
        mock_client.detect_document_text.side_effect = [
            _textract_blocks("Page one"),
            _textract_blocks("Page two"),
        ]
        output = reader.read(str(pdf_file))

    mock_raster.assert_called_once()
    assert mock_client.detect_document_text.call_count == 2
    assert "Page one" in output.text
    assert "Page two" in output.text
    assert output.reader_method == "textract"
    assert output.ocr_method == TEXTRACT_OCR_METHOD
    assert output.conversion_method == "png"
    assert output.page_placeholder == DEFAULT_PAGE_PLACEHOLDER


def test_read_office_converts_to_pdf_before_ocr(
    reader: TextractReader, mock_client: MagicMock, tmp_path
):
    docx_file = tmp_path / "report.docx"
    docx_file.write_text("fake docx", encoding="utf-8")

    with (
        patch("shutil.which", return_value="/usr/bin/soffice"),
        patch.object(
            TextractReader,
            "_convert_office_to_pdf",
            return_value=str(tmp_path / "report.pdf"),
        ) as mock_convert,
        patch.object(
            TextractReader, "_pdf_to_png_pages", return_value=[b"png-page"]
        ) as mock_raster,
    ):
        output = reader.read(str(docx_file))

    mock_convert.assert_called_once()
    mock_raster.assert_called_once()
    mock_client.detect_document_text.assert_called_once()
    assert output.document_name == "report.docx"
    assert output.reader_method == "textract"


def test_read_image_normalizes_to_png_before_ocr(
    reader: TextractReader, mock_client: MagicMock, tmp_path
):
    image_file = tmp_path / "scan.jpg"
    image_file.write_bytes(b"fake-jpeg")

    with patch.object(
        TextractReader, "_image_to_png_pages", return_value=[b"png-page"]
    ) as mock_normalize:
        output = reader.read(str(image_file))

    mock_normalize.assert_called_once()
    mock_client.detect_document_text.assert_called_once()
    assert output.conversion_method == "png"


def test_blocks_to_text_uses_page_child_lines_only():
    blocks = _textract_blocks("Alpha", "Beta")["Blocks"]
    blocks.append({"Id": "word-1", "BlockType": "WORD", "Text": "Alpha"})
    text = TextractReader._blocks_to_text(blocks)
    assert text == "Alpha\nBeta"


# ---- Error paths ---- #


def test_read_missing_file_raises_config_error(reader: TextractReader):
    with pytest.raises(ReaderConfigException, match="File not found"):
        reader.read("/tmp/does-not-exist.pdf")


def test_read_unsupported_extension_raises_config_error(
    reader: TextractReader, tmp_path
):
    bad_file = tmp_path / "data.parquet"
    bad_file.write_text("x", encoding="utf-8")

    with pytest.raises(ReaderConfigException, match="Unsupported file extension"):
        reader.read(str(bad_file))


def test_read_office_missing_libreoffice_raises_textract_error(
    reader: TextractReader, tmp_path
):
    docx_file = tmp_path / "report.docx"
    docx_file.write_text("fake", encoding="utf-8")

    with patch("shutil.which", return_value=None):
        with pytest.raises(TextractReaderException, match="LibreOffice/soffice"):
            reader.read(str(docx_file))


def test_detect_text_missing_credentials_raises_config_error(
    reader: TextractReader,
):
    reader._client.detect_document_text.side_effect = NoCredentialsError()
    with pytest.raises(ReaderConfigException, match="AWS credentials were not found"):
        reader._detect_text(b"png")


def test_detect_text_client_error_is_wrapped(reader: TextractReader):
    reader._client.detect_document_text.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "DetectDocumentText",
    )
    with pytest.raises(TextractReaderException, match="detect_document_text failed"):
        reader._detect_text(b"png")


def test_extract_text_from_pages_oversized_page_raises(
    reader: TextractReader,
):
    oversized = b"x" * (TEXTRACT_SYNC_MAX_BYTES + 1)
    with pytest.raises(TextractReaderException, match="synchronous size limit"):
        reader._extract_text_from_pages([oversized], DEFAULT_PAGE_PLACEHOLDER)


# ---- Edge cases ---- #


def test_read_custom_page_placeholder_is_applied(
    reader: TextractReader, mock_client: MagicMock, tmp_path
):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    with patch.object(TextractReader, "_pdf_to_png_pages", return_value=[b"png"]):
        output = reader.read(str(pdf_file), page_placeholder="<<{page}>>")

    assert "<<1>>" in output.text
    assert output.page_placeholder == "<<{page}>>"


def test_blocks_to_text_falls_back_to_line_blocks_without_pages():
    blocks = [{"Id": "line-1", "BlockType": "LINE", "Text": "Fallback line"}]
    assert TextractReader._blocks_to_text(blocks) == "Fallback line"


def test_read_empty_pdf_pages_raises_textract_error(reader: TextractReader, tmp_path):
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    with patch.object(TextractReader, "_pdf_to_png_pages", return_value=[]):
        with pytest.raises(
            TextractReaderException, match="No pages available for Textract processing"
        ):
            reader.read(str(pdf_file))
