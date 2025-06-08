import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import WordSplitter


@pytest.fixture
def reader_output():
    return ReaderOutput(
        text="The quick brown fox jumps over the lazy dog and runs away",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
        metadata={},
    )


def test_basic_split(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert hasattr(result, "chunks")
    assert result.chunks == [
        "The quick brown fox",
        "jumps over the lazy",
        "dog and runs away",
    ]
    assert result.split_method == "word_splitter"
    assert result.split_params["chunk_size"] == 4
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=2)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "The quick brown fox"
    assert result.chunks[1] == "brown fox jumps over"
    assert result.chunks[2] == "jumps over the lazy"
    assert result.chunks[3] == "the lazy dog and"
    assert result.chunks[4] == "dog and runs away"


def test_split_with_overlap_float(reader_output):
    splitter = WordSplitter(chunk_size=6, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "The quick brown fox jumps over"
    assert result.chunks[1] == "fox jumps over the lazy dog"
    assert result.chunks[2] == "the lazy dog and runs away"


def test_chunk_overlap_equals_chunk_size_raises(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=4)
    with pytest.raises(ValueError):
        splitter.split(reader_output)


def test_output_contains_metadata(reader_output):
    splitter = WordSplitter(chunk_size=4, chunk_overlap=0)
    result = splitter.split(reader_output)
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
    splitter = WordSplitter(chunk_size=5, chunk_overlap=0)
    reader_output = ReaderOutput(text="")
    result = splitter.split(reader_output)
    assert result.chunks == []
