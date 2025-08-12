import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import SentenceSplitter


# Helpers
@pytest.fixture
def reader_output():
    return ReaderOutput(
        text=(
            "Hello world! How are you? I am fine. "
            "Testing sentence splitting. "
            "Short. End! And another?"
        ),
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="text",
        ocr_method=None,
        metadata={},
    )


def test_basic_split(reader_output):
    splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
    result = splitter.split(reader_output)
    assert result.chunks[0] == "Hello world! How are you? I am fine."
    assert result.chunks[1] == "Testing sentence splitting. Short. End!"
    assert result.chunks[2] == "And another?"
    assert result.split_method == "sentence_splitter"
    assert result.split_params["chunk_size"] == 3
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=2)
    result = splitter.split(reader_output)
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    first_words = first_chunk.split()[-2:]
    assert " ".join(first_words) in second_chunk


def test_split_with_overlap_float(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    if len(result.chunks) > 1:
        prev_words = result.chunks[0].split()
        overlap = set(prev_words) & set(result.chunks[1].split())
        assert len(overlap) >= 1


def test_separator_variants():
    text = "A|B|C|D"
    ro = ReaderOutput(text=text, document_path="/tmp/sample.txt")
    # IMPORTANT: pass a literal separator as a LIST (or escape the regex as r"\|")
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0, separators=["|"])
    result = splitter.split(ro)
    assert result.chunks[0] == "A| B|"
    assert result.chunks[1] == "C| D"


def test_output_contains_metadata(reader_output):
    splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
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
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0)
    ro = ReaderOutput(text="")
    # New behavior: return a single empty chunk instead of raising
    out = splitter.split(ro)
    assert out.chunks == [""]
