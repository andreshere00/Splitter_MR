import os

import pytest

from splitter_mr.reader import VanillaReader


@pytest.fixture
def reader():
    return VanillaReader()


def test_read_txt(tmp_path, reader):
    f = tmp_path / "foo.txt"
    f.write_text("hello world\nnew line")
    result = reader.read(str(f))
    assert result["text"] == "hello world\nnew line"
    assert result["document_name"] == "foo.txt"
    assert result["document_path"] == os.path.abspath(str(f))
    assert result["conversion_method"] is None


def test_read_html(tmp_path, reader):
    f = tmp_path / "foo.html"
    f.write_text("<html><body>hi</body></html>")
    result = reader.read(str(f))
    assert "<body>hi</body>" in result["text"]


def test_read_json(tmp_path, reader):
    f = tmp_path / "foo.json"
    f.write_text('{"a": 1, "b": 2}')
    result = reader.read(str(f))
    assert '"a": 1' in result["text"]
    assert result["document_name"] == "foo.json"


def test_read_csv(tmp_path, reader):
    f = tmp_path / "foo.csv"
    content = "x,y\n1,2\n3,4"
    f.write_text(content)
    result = reader.read(str(f))
    assert "x,y" in result["text"]


def test_read_yaml(tmp_path, reader):
    import yaml

    f = tmp_path / "foo.yaml"
    content = "a: 1\nb: 2"
    f.write_text(content)
    result = reader.read(str(f))
    assert isinstance(result["text"], dict)
    assert result["text"]["a"] == 1
    assert result["conversion_method"] == "json"


def test_read_yml(tmp_path, reader):
    import yaml

    f = tmp_path / "foo.yml"
    content = "hello: world"
    f.write_text(content)
    result = reader.read(str(f))
    assert isinstance(result["text"], dict)
    assert result["text"]["hello"] == "world"


def test_read_parquet(tmp_path, reader):
    # Only runs if pyarrow or fastparquet is installed
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    f = tmp_path / "foo.parquet"
    df.to_parquet(f)
    result = reader.read(str(f))
    assert "a,b" in result["text"]
    assert "1,3" in result["text"] or "2,4" in result["text"]
    assert result["conversion_method"] == "pandas"


def test_metadata_and_doc_id(tmp_path, reader):
    f = tmp_path / "foo.txt"
    f.write_text("meta test")
    result = reader.read(str(f), document_id="id123", metadata={"x": 1})
    assert result["document_id"] == "id123"
    assert result["metadata"] == {"x": 1}


def test_unsupported_extension(tmp_path, reader):
    f = tmp_path / "foo.unsupported"
    f.write_text("should fail")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        reader.read(str(f))
