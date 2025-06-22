import os
from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.reader import VanillaReader

# Helpers


@pytest.fixture
def reader():
    return VanillaReader()


# Test cases


def test_read_txt(tmp_path, reader):
    f = tmp_path / "foo.txt"
    f.write_text("hello world\nnew line")
    result = reader.read(file_path=str(f))
    assert result.text == "hello world\nnew line"
    assert result.document_name == "foo.txt"
    assert result.document_path == os.path.relpath(str(f))
    assert result.conversion_method == "txt"


def test_read_txt_as_positional(tmp_path, reader):
    f = tmp_path / "bar.txt"
    f.write_text("positional arg")
    result = reader.read(str(f))
    assert result.text == "positional arg"
    assert result.document_name == "bar.txt"
    assert result.document_path == os.path.relpath(str(f))


def test_read_html(tmp_path, reader):
    f = tmp_path / "foo.html"
    f.write_text("<html><body>hi</body></html>")
    result = reader.read(file_path=str(f))
    assert "hi" in result.text


def test_read_json(tmp_path, reader):
    f = tmp_path / "foo.json"
    f.write_text('{"a": 1, "b": 2}')
    result = reader.read(file_path=str(f))
    assert isinstance(result.text, str)
    assert '"a": 1' in result.text
    assert result.document_name == "foo.json"


def test_read_json_kwarg_dict(reader):
    data = {"hello": "world"}
    result = reader.read(json_document=data)
    assert result.text["hello"] == "world"
    assert result.conversion_method == "json"


def test_read_json_kwarg_str(reader):
    result = reader.read(json_document='{"z": 42}')
    assert result.text["z"] == 42
    assert result.conversion_method == "json"


def test_read_csv(tmp_path, reader):
    f = tmp_path / "foo.csv"
    content = "x,y\n1,2\n3,4"
    f.write_text(content)
    result = reader.read(file_path=str(f))
    assert "x,y" in result.text


def test_read_yaml(tmp_path, reader):
    f = tmp_path / "foo.yaml"
    content = "a: 1\nb: 2"
    f.write_text(content)
    result = reader.read(file_path=str(f))
    assert isinstance(result.text, dict)
    assert result.text["a"] == 1
    assert result.conversion_method == "json"


def test_read_yml(tmp_path, reader):
    f = tmp_path / "foo.yml"
    content = "hello: world"
    f.write_text(content)
    result = reader.read(file_path=str(f))
    assert isinstance(result.text, dict)
    assert result.text["hello"] == "world"


def test_read_parquet(tmp_path, reader):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    f = tmp_path / "foo.parquet"
    df.to_parquet(f)
    result = reader.read(file_path=str(f))
    assert "a,b" in result.text
    assert "1,3" in result.text or "2,4" in result.text
    assert result.conversion_method == "csv"


def test_metadata_and_doc_id(tmp_path, reader):
    f = tmp_path / "foo.txt"
    f.write_text("meta test")
    result = reader.read(file_path=str(f), document_id="id123", metadata={"x": 1})
    assert result.document_id == "id123"
    assert result.metadata == {"x": 1}


def test_unsupported_extension(tmp_path, reader):
    f = tmp_path / "foo.unsupported"
    f.write_text("should fail")
    with pytest.raises(
        ValueError,
        match="Unsupported file extension: unsupported. Use another Reader component.",
    ):
        reader.read(file_path=str(f))


def test_text_document_kwarg(reader):
    result = reader.read(text_document="plain text here")
    assert result.text == "plain text here"
    assert result.conversion_method == "txt"


def test_file_path_auto_json(reader):
    # file_path is actually JSON text, not a real file
    json_str = '{"auto": "json"}'
    result = reader.read(file_path=json_str)
    assert result.text["auto"] == "json"
    assert result.conversion_method == "json"


def test_file_path_auto_yaml(reader):
    # file_path is actually YAML text, not a real file
    yaml_str = "foo: bar"
    result = reader.read(file_path=yaml_str)
    assert result.text["foo"] == "bar"
    assert result.conversion_method == "json"


def test_file_path_auto_url(reader, requests_mock):
    url = "http://testdomain.com/some.txt"
    requests_mock.get(url, text="file from url!")
    result = reader.read(file_path=url)
    assert result.text == "file from url!"
    assert result.document_path == url
    assert result.conversion_method == "txt"


def test_file_url_kwarg(reader, requests_mock):
    url = "http://somesite.com/data.txt"
    requests_mock.get(url, text="hello from url")
    result = reader.read(file_url=url)
    assert result.text == "hello from url"
    assert result.document_name == "data.txt"
    assert result.document_path == url


def test_json_document_invalid(reader):
    with pytest.raises(TypeError):
        reader.read(json_document=["not", "a", "dict", "or", "json"])


def test_text_document_json_fallback(reader):
    result = reader.read(text_document='{"x":123}')
    assert result.text["x"] == 123
    assert result.conversion_method == "json"


def test_text_document_yaml_fallback(reader):
    result = reader.read(text_document="a: 42")
    assert result.text["a"] == 42
    assert result.conversion_method == "json"


# ------------------- PDF / Images ---------------------


@patch("splitter_mr.reader.readers.utils.pdfplumber_reader.pdfplumber.open")
@patch("splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader")
def test_read_pdf_no_images(mock_pdfplumber, mock_pdfplumber_open, tmp_path, reader):
    f = tmp_path / "foo.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content")
    fake_md = "---\n## Page 1\n---\nThis is text from PDF"
    instance = mock_pdfplumber.return_value
    instance.read.return_value = fake_md
    mock_pdfplumber_open.return_value.__enter__.return_value = MagicMock()
    result = reader.read(file_path=str(f))
    assert result.text == fake_md
    instance.read.assert_called_once_with(str(f), show_images=False)


@patch("splitter_mr.reader.readers.utils.pdfplumber_reader.pdfplumber.open")
@patch("splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader")
def test_read_pdf_with_images_flag(
    mock_pdfplumber, mock_pdfplumber_open, tmp_path, reader
):
    f = tmp_path / "foo.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content")
    fake_md = "---\n## Page 1\n---\n![Image page 1](...)"
    instance = mock_pdfplumber.return_value
    instance.read.return_value = fake_md
    mock_pdfplumber_open.return_value.__enter__.return_value = MagicMock()
    result = reader.read(file_path=str(f), show_images=True)
    assert "Image page" in result.text
    instance.read.assert_called_once_with(str(f), show_images=True)


@patch("splitter_mr.reader.readers.utils.pdfplumber_reader.pdfplumber.open")
@patch("splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader")
def test_read_pdf_with_model_and_prompt(
    mock_pdfplumber, mock_pdfplumber_open, tmp_path, reader
):
    f = tmp_path / "foo.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content")
    fake_md = "---\n## Page 1\n---\nDummy image caption"
    instance = mock_pdfplumber.return_value
    instance.read.return_value = fake_md
    mock_pdfplumber_open.return_value.__enter__.return_value = MagicMock()

    class DummyModel:
        model_name = "dummy"

    model = DummyModel()
    result = reader.read(
        file_path=str(f), model=model, prompt="caption!", show_images=False
    )
    assert "Dummy image caption" in result.text
    instance.read.assert_called_once_with(
        str(f), model=model, prompt="caption!", show_images=False
    )


@patch("splitter_mr.reader.readers.utils.pdfplumber_reader.pdfplumber.open")
@patch("splitter_mr.reader.readers.vanilla_reader.PDFPlumberReader")
def test_read_pdf_omitted_image_indicator(
    mock_pdfplumber, mock_pdfplumber_open, tmp_path, reader
):
    f = tmp_path / "foo.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content")
    fake_md = "---\n## Page 1\n---\n![Image]()"
    instance = mock_pdfplumber.return_value
    instance.read.return_value = fake_md
    mock_pdfplumber_open.return_value.__enter__.return_value = MagicMock()
    result = reader.read(file_path=str(f), show_images=False)
    assert "Image" in result.text or "image" in result.text
    instance.read.assert_called_once_with(str(f), show_images=False)
