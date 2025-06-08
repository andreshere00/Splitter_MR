import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import ParagraphSplitter

# Helpers


@pytest.fixture
def reader_output():
    # 5 paragraphs, mixed line breaks
    return ReaderOutput(
        text=(
            "Para1 first sentence. Para1 second sentence.\n"
            "Para2 here.\n"
            "Para3 is this line.\n"
            "Para4 once more.\n"
            "Para5 and last."
        ),
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
    )


# Tests cases


def test_basic_split(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert hasattr(result, "chunks")
    assert (
        result.chunks[0] == "Para1 first sentence. Para1 second sentence.\nPara2 here."
    )
    assert result.chunks[1] == "Para3 is this line.\nPara4 once more."
    assert result.chunks[2] == "Para5 and last."
    assert result.split_method == "paragraph_splitter"
    assert result.split_params["chunk_size"] == 2
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=3)
    result = splitter.split(reader_output)
    # Each chunk after the first should start with the last 3 words of the previous chunk
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    first_words = first_chunk.split()[-3:]
    assert " ".join(first_words) in second_chunk


def test_split_with_overlap_float(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    if len(result.chunks) > 1:
        prev_words = result.chunks[0].split()
        overlap = set(prev_words) & set(result.chunks[1].split())
        assert len(overlap) >= 1


def test_custom_linebreak():
    text = "P1||P2||P3"
    reader_output = ReaderOutput(text=text, document_path="/tmp/sample.txt")
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0, line_break="||")
    result = splitter.split(reader_output)
    assert result.chunks[0] == "P1||P2"
    assert result.chunks[1] == "P3"


def test_output_contains_metadata(reader_output):
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
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
    splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=0)
    reader_output = ReaderOutput(text="")
    result = splitter.split(reader_output)
    assert result.chunks == []
