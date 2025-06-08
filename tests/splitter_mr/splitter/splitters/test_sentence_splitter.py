import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import SentenceSplitter

# Helpers


@pytest.fixture
def reader_output():
    # 7 sentences, various endings, extra whitespace to test stripping.
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


# Test cases


def test_basic_split(reader_output):
    splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
    result = splitter.split(reader_output)
    # Should split into 3-sentence chunks
    # Sentences: ["Hello world!", "How are you?", "I am fine.", "Testing sentence splitting.", "Short.", "End!", "And another?"]
    assert result.chunks[0] == "Hello world! How are you? I am fine."
    assert result.chunks[1] == "Testing sentence splitting. Short. End!"
    assert result.chunks[2] == "And another?"
    assert result.split_method == "sentence_splitter"
    assert result.split_params["chunk_size"] == 3
    assert result.split_params["chunk_overlap"] == 0


def test_split_with_overlap_int(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=2)
    result = splitter.split(reader_output)
    # Each chunk after the first should start with the last 2 words of the previous chunk
    # Let's check the overlap for the second chunk:
    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]
    first_words = first_chunk.split()[-2:]
    assert " ".join(first_words) in second_chunk


def test_split_with_overlap_float(reader_output):
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0.5)
    result = splitter.split(reader_output)
    # overlap = int(max_words_in_sent * 0.5)
    # Let's check the overlap logic for the second chunk:
    if len(result.chunks) > 1:
        prev_words = result.chunks[0].split()
        # Since float overlap, overlap might be small; just check repeated words in sequence
        overlap = set(prev_words) & set(result.chunks[1].split())
        assert len(overlap) >= 1


def test_separator_variants():
    text = "A|B|C|D"
    reader_output = ReaderOutput(text=text, document_path="/tmp/sample.txt")
    splitter = SentenceSplitter(chunk_size=2, chunk_overlap=0, separators="|")
    result = splitter.split(reader_output)
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
    reader_output = ReaderOutput(text="")
    result = splitter.split(reader_output)
    assert result.chunks == []
