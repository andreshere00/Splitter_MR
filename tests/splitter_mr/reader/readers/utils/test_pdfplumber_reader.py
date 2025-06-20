from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.reader.readers.utils.pdfplumber_reader import PDFPlumberReader

# Helpers


class DummyModel:
    def extract_text(self, file, prompt=None):
        return "Dummy caption"


@pytest.fixture
def mock_word_lines():
    return [
        {"text": "Hello", "top": 10.0, "bottom": 18.0, "x0": 50},
        {"text": "world", "top": 10.1, "bottom": 18.1, "x0": 80},
        {"text": "This", "top": 35.0, "bottom": 45.0, "x0": 50},
        {"text": "is", "top": 35.2, "bottom": 45.2, "x0": 80},
    ]


@pytest.fixture
def fake_table():
    return [["Header1", "Header2"], ["Row1Cell1", "Row1Cell2"], ["Row2Cell1", ""]]


@pytest.fixture
def fake_image_dict():
    return [{"x0": 100, "top": 200, "x1": 300, "bottom": 400}]


# Test cases


def test_group_by_lines(mock_word_lines):
    reader = PDFPlumberReader()
    result = reader.group_by_lines(mock_word_lines)
    assert len(result) == 2
    assert result[0]["content"] == "Hello world"
    assert result[1]["content"] == "This is"


def test_is_real_table():
    reader = PDFPlumberReader()
    assert reader.is_real_table([["a", "b"], ["c", "d"]])
    assert not reader.is_real_table([["a"], ["b"], ["c"]])  # mostly single col
    assert not reader.is_real_table([[]])  # blank row
    assert not reader.is_real_table([])  # empty


def test_table_to_markdown(fake_table):
    reader = PDFPlumberReader()
    md = reader.table_to_markdown(fake_table)
    assert "| Header1 | Header2 |" in md
    assert "| Row1Cell1 | Row1Cell2 |" in md


@patch("pdfplumber.open")
def test_read_extracts_text(
    mock_pdfplumber_open, mock_word_lines, fake_table, fake_image_dict
):
    # Set up fake pdfplumber
    fake_page = MagicMock()
    fake_page.extract_words.return_value = mock_word_lines
    fake_page.find_tables.return_value = []
    fake_page.images = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf", show_images=False)
    assert "Hello world" in md
    assert "This is" in md


@patch("pdfplumber.open")
def test_read_with_table(mock_pdfplumber_open, mock_word_lines, fake_table):
    fake_table_obj = MagicMock()
    fake_table_obj.bbox = (10, 20, 30, 40)
    fake_table_obj.extract.return_value = fake_table

    fake_page = MagicMock()
    fake_page.extract_words.return_value = mock_word_lines
    fake_page.find_tables.return_value = [fake_table_obj]
    fake_page.images = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf")
    assert "| Header1 | Header2 |" in md


@patch("pdfplumber.open")
def test_read_with_images_and_annotations(mock_pdfplumber_open):
    # Fake image object
    fake_image = {"x0": 10, "top": 20, "x1": 30, "bottom": 40}
    # Patch the image extraction chain
    fake_img_obj = MagicMock()
    fake_img_obj.to_image.return_value = MagicMock(
        save=lambda buf, format: buf.write(b"fakeimg")
    )
    fake_page = MagicMock()
    fake_page.extract_words.return_value = []
    fake_page.find_tables.return_value = []
    fake_page.images = [fake_image]
    fake_page.within_bbox.return_value = fake_img_obj

    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    # Without annotation
    md = reader.read("fakefile.pdf", show_images=True)
    assert "data:image/png;base64" in md

    # With annotation
    md = reader.read("fakefile.pdf", show_images=False, model=DummyModel())
    assert "Dummy caption" in md


@patch("pdfplumber.open")
def test_blocks_to_markdown_omitted_image_indicator(mock_pdfplumber_open):
    # Test for image omitted placeholder
    fake_image = {"x0": 10, "top": 20, "x1": 30, "bottom": 40}
    fake_img_obj = MagicMock()
    fake_img_obj.to_image.return_value = MagicMock(
        save=lambda buf, format: buf.write(b"fakeimg")
    )
    fake_page = MagicMock()
    fake_page.extract_words.return_value = []
    fake_page.find_tables.return_value = []
    fake_page.images = [fake_image]
    fake_page.within_bbox.return_value = fake_img_obj
    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    reader = PDFPlumberReader()
    md = reader.read("fakefile.pdf", show_images=False)
    assert "Image" in md or "image" in md


def test_blocks_to_markdown_table_and_text():
    reader = PDFPlumberReader()
    blocks = [
        {"type": "text", "top": 10, "bottom": 20, "content": "Text 1", "page": 1},
        {
            "type": "table",
            "top": 30,
            "bottom": 40,
            "content": [["a", "b"], ["c", "d"]],
            "page": 1,
        },
        {"type": "text", "top": 50, "bottom": 60, "content": "Text 2", "page": 1},
    ]
    md = reader.blocks_to_markdown(blocks, show_images=True)
    assert "Text 1" in md
    assert "| a | b |" in md
    assert "Text 2" in md


def test_blocks_to_markdown_handles_blank():
    reader = PDFPlumberReader()
    blocks = []
    md = reader.blocks_to_markdown(blocks)
    assert isinstance(md, str)
