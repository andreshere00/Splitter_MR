import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import CharacterSplitter


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text="abcdefghijklmnopqrstuvwxyz",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="txt",
        ocr_method=None,
    )


def test_basic_split(reader_output):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    result = splitter.split(reader_output)
    print(result)
    assert result.chunks == ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
    assert result.split_method == "character_splitter"
    assert result.split_params["chunk_size"] == 5
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=2)
    result = splitter.split(reader_output)
    # Expected start positions: 0, 3, 6, 9, ...
    assert result.chunks[0] == "abcde"
    assert result.chunks[1] == "defgh"
    assert result.chunks[2] == "ghijk"
    assert result.chunks[3] == "jklmn"
    # Overlapping by 2 each time
    # The last chunk should start at position 24
    assert result.chunks[-1] == "yz"


def test_split_with_overlap_float(reader_output):
    splitter = CharacterSplitter(chunk_size=10, chunk_overlap=0.3)
    result = splitter.split(reader_output)
    # overlap = int(10 * 0.3) == 3
    # chunks: 0:10, 7:17, 14:24, 21:31
    assert result.chunks[0] == "abcdefghij"
    assert result.chunks[1] == "hijklmnopq"
    assert result.chunks[2] == "opqrstuvwx"
    assert (
        result.chunks[3] == "vwx yz"
        or result.chunks[3] == "y"
        or result.chunks[3].endswith("z")
    )  # last chunk


def test_chunk_overlap_equals_chunk_size_raises(reader_output):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=5)
    with pytest.raises(ValueError):
        splitter.split(reader_output)


def test_output_contains_metadata(reader_output):
    splitter = CharacterSplitter(chunk_size=10, chunk_overlap=0)
    result = splitter.split(reader_output)
    # Check required fields as attributes
    for field in [
        "chunks",
        "chunk_id",
        "document_name",
        "document_path",
        "document_id",
        "conversion_method",
        "ocr_method",
        "split_method",
        "split_params",
        "metadata",
    ]:
        assert hasattr(result, field)


def test_empty_text():
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    reader_output = ReaderOutput(text="")
    result = splitter.split(reader_output)
    assert result.chunks == []
