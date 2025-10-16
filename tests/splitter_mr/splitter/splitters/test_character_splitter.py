import warnings

import pytest

from splitter_mr.schema.exceptions import InvalidChunkException, SplitterOutputException
from splitter_mr.schema.models import ReaderOutput
from splitter_mr.schema.warnings import SplitterInputWarning
from splitter_mr.splitter import CharacterSplitter

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def reader_output_alpha():
    return ReaderOutput(
        text="abcdefghijklmnopqrstuvwxyz",
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        document_id="123",
        conversion_method="txt",
        ocr_method=None,
    )


# -----------------------
# Happy-path behavior
# -----------------------


def test_basic_split_no_overlap(reader_output_alpha):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    result = splitter.split(reader_output_alpha)
    assert result.chunks == ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
    assert result.split_method == "character_splitter"
    assert result.split_params == {"chunk_size": 5, "chunk_overlap": 0}


def test_overlap_int(reader_output_alpha):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=2)
    result = splitter.split(reader_output_alpha)
    # start positions: 0, 3, 6, 9, ...
    assert result.chunks[:4] == ["abcde", "defgh", "ghijk", "jklmn"]
    assert result.chunks[-1] == "yz"  # last starts at 24


def test_overlap_float(reader_output_alpha):
    splitter = CharacterSplitter(chunk_size=10, chunk_overlap=0.3)  # overlap=3, step=7
    result = splitter.split(reader_output_alpha)
    assert result.chunks[0] == "abcdefghij"  # 0:10
    assert result.chunks[1] == "hijklmnopq"  # 7:17
    assert result.chunks[2] == "opqrstuvwx"  # 14:24
    assert result.chunks[3] == "vwxyz"  # 21: -> end


def test_tight_overlap_step_one(reader_output_alpha):
    # overlap = chunk_size - 1 => step becomes 1 (max overlap without being equal)
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=4)
    result = splitter.split(reader_output_alpha)
    # First few chunks slide by 1 char
    assert result.chunks[0] == "abcde"
    assert result.chunks[1] == "bcdef"
    assert result.chunks[2] == "cdefg"
    assert result.chunks[-1].endswith("z")


def test_chunk_size_one(reader_output_alpha):
    splitter = CharacterSplitter(chunk_size=1, chunk_overlap=0)
    result = splitter.split(reader_output_alpha)
    # every char becomes a chunk
    assert result.chunks[:5] == list("abcde")
    assert result.chunks[-1] == "z"
    # metadata presence
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


# -----------------------
# Warnings
# -----------------------


def test_empty_text_warns_and_returns_single_empty_chunk():
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    ro = ReaderOutput(text="")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = splitter.split(ro)
        assert result.chunks == [""]
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)


def test_whitespace_only_text_warns_and_chunks_spaces():
    splitter = CharacterSplitter(chunk_size=3, chunk_overlap=0)
    text = "     "  # 5 spaces
    ro = ReaderOutput(text=text)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = splitter.split(ro)
        # Warning is emitted since text.strip() == ""
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)
        # But chunks are made from the raw text (not coerced to empty)
        assert result.chunks == ["   ", "  "]


def test_json_declared_invalid_emits_warning():
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    ro = ReaderOutput(text="{not json", conversion_method="json")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = splitter.split(ro)
        assert len(result.chunks) > 0
        assert any(isinstance(rec.message, SplitterInputWarning) for rec in w)


def test_json_declared_valid_no_warning():
    splitter = CharacterSplitter(chunk_size=8, chunk_overlap=0)
    ro = ReaderOutput(text='{"a": 1, "b": [2,3]}', conversion_method="json")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = splitter.split(ro)
        # must not warn
        assert not any(isinstance(rec.message, SplitterInputWarning) for rec in w)
        assert result.chunks[0].startswith("{")
        assert result.split_params["chunk_size"] == 8


# -----------------------
# Parameter validation (ValueError from __init__)
# -----------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},  # too small
        {"chunk_size": -2},  # negative
        {"chunk_size": 10.5},  # non-int
        {"chunk_size": 5, "chunk_overlap": -1},  # int negative
        {"chunk_size": 5, "chunk_overlap": 5},  # int == chunk_size
        {"chunk_size": 5, "chunk_overlap": 7},  # int > chunk_size
        {"chunk_size": 10, "chunk_overlap": -0.1},  # float < 0
        {"chunk_size": 10, "chunk_overlap": 1.0},  # float >= 1.0
        {"chunk_size": 10, "chunk_overlap": "1"},  # wrong type
    ],
)
def test_invalid_constructor_params_raise_valueerror(kwargs):
    with pytest.raises(ValueError):
        CharacterSplitter(**kwargs)


def test_float_overlap_edge_values_ok():
    # exactly 0.0 is valid
    CharacterSplitter(chunk_size=10, chunk_overlap=0.0)
    # just below 1.0 is valid -> floor in _coerce_overlap
    CharacterSplitter(chunk_size=10, chunk_overlap=0.999999)


# -----------------------
# Exceptions raised during split()
# -----------------------


def test_invalid_chunk_build_wrapped_as_invalid_chunk_exception(reader_output_alpha):
    # Trigger an exception during slicing inside the chunk loop to ensure it’s wrapped.
    class BoomText(str):
        def __getitem__(self, key):
            raise Exception("boom")

    ro = reader_output_alpha
    ro.text = BoomText(reader_output_alpha.text)

    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)
    with pytest.raises(InvalidChunkException):
        splitter.split(ro)


def test_splitter_output_exception_wrapped_when_output_validation_fails(
    monkeypatch, reader_output_alpha
):
    splitter = CharacterSplitter(chunk_size=5, chunk_overlap=0)

    # Cause mismatch between number of chunks and chunk_id list
    def bad_generate_chunk_ids(n):
        return ["only-one-id"]

    monkeypatch.setattr(splitter, "_generate_chunk_ids", bad_generate_chunk_ids)

    with pytest.raises(SplitterOutputException):
        splitter.split(reader_output_alpha)


# -----------------------
# Additional edges
# -----------------------


def test_long_text_with_step_one_progress(reader_output_alpha):
    # chunk_size=10, overlap=9 -> step=1 ensures forward progress in a tight loop
    splitter = CharacterSplitter(chunk_size=10, chunk_overlap=9)
    result = splitter.split(reader_output_alpha)
    # Sliding window of width 10
    assert result.chunks[0] == "abcdefghij"
    assert result.chunks[1] == "bcdefghijk"
    assert result.chunks[2] == "cdefghijkl"
    assert result.chunks[-1].endswith("z")
