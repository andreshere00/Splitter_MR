import re
from typing import List

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter.splitters.keyword_splitter import KeywordSplitter

# ---- Helpers, mocks and fixtures ---- #


@pytest.fixture()
def sample_text() -> str:
    return (
        "Intro text.\n"
        "TODO: refactor parser.\n"
        "Body continues here.\n"
        "NOTE: verify edge cases.\n"
        "Ending.\n"
    )


@pytest.fixture()
def reader_output(sample_text: str) -> ReaderOutput:
    """ReaderOutput with deterministic document metadata for tests."""
    return ReaderOutput(
        text=sample_text,
        document_name="sample.txt",
        document_path="/tmp/sample.txt",
        conversion_method="txt",
        reader_method="vanilla",
        ocr_method=None,
        page_placeholder=None,
        metadata={"source": "unit-test"},
    )


TODO_PATTERN: str = r"^TODO:"
NOTE_PATTERN: str = r"^NOTE:"
TODO_FULL: str = "TODO:"
NOTE_FULL: str = "NOTE:"


def _make_splitter(
    include: str = "before",
    chunk_size: int = 1000,
    flags: int = re.MULTILINE,
) -> KeywordSplitter:
    patterns = {
        "todo": TODO_PATTERN,
        "note": NOTE_PATTERN,
    }
    return KeywordSplitter(
        patterns=patterns,
        flags=flags,
        include_delimiters=include,
        chunk_size=chunk_size,
    )


# ---- Test cases ---- #


def test_split_before_attaches_match_to_preceding_chunk_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="before")
    out = sp.split(reader_output)
    assert any(chunk.strip().endswith(TODO_FULL) for chunk in out.chunks)
    assert any(chunk.strip().endswith(NOTE_FULL) for chunk in out.chunks)
    assert out.split_method == "keyword"


def test_split_after_attaches_match_to_following_chunk_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="after")
    out = sp.split(reader_output)
    assert any(chunk.strip().startswith(TODO_FULL) for chunk in out.chunks)
    assert any(chunk.strip().startswith(NOTE_FULL) for chunk in out.chunks)


def test_split_both_duplicates_delimiter_on_both_sides_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="both")
    out = sp.split(reader_output)
    assert any(chunk.strip().endswith(TODO_FULL) for chunk in out.chunks)
    assert any(chunk.strip().startswith(TODO_FULL) for chunk in out.chunks)


def test_split_none_excludes_delimiters_from_chunks_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="none")
    out = sp.split(reader_output)
    joined = "\n".join(out.chunks)
    assert TODO_FULL not in joined and NOTE_FULL not in joined


def test_split_populates_metadata_counts_and_spans_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter()
    out = sp.split(reader_output)
    meta = out.metadata.get("keyword_matches", {})
    counts = meta.get("counts", {})
    spans: List = meta.get("spans", [])
    assert counts.get("todo", 0) == 1
    assert counts.get("note", 0) == 1
    assert all(isinstance(t, tuple) and len(t) == 2 for t in spans)


def test_split_generates_chunk_ids_matching_number_of_chunks_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter()
    out = sp.split(reader_output)
    assert len(out.chunk_id) == len(out.chunks)


def test_split_carries_reader_metadata_forward_expected(reader_output: ReaderOutput):
    sp = _make_splitter()
    out = sp.split(reader_output)
    assert out.document_name == reader_output.document_name
    assert out.document_path == reader_output.document_path
    assert out.document_id == reader_output.document_id
    assert out.metadata.get("source") == "unit-test"


def test_split_respects_chunk_size_soft_wrap_expected():
    text = "TODO: " + ("word " * 100)  # long body
    ro = ReaderOutput(
        text=text, document_name="long.txt", document_path="/tmp/long.txt"
    )
    sp = _make_splitter(include="before", chunk_size=50)
    out = sp.split(ro)
    assert all(len(c) <= 50 for c in out.chunks)
    assert len(out.chunks) > 1


def test_split_hard_splits_oversized_token_expected():
    big_token = "A" * 120
    text = f"Intro {big_token} Outro"
    ro = ReaderOutput(text=text, document_name="big.txt", document_path="/tmp/big.txt")
    sp = KeywordSplitter(patterns=[r"XYZ"], include_delimiters="none", chunk_size=50)
    out = sp.split(ro)
    assert any(len(c) == 50 for c in out.chunks)


def test_split_works_with_list_patterns_and_implicit_names_expected(
    reader_output: ReaderOutput,
):
    sp = KeywordSplitter(
        patterns=[TODO_PATTERN, NOTE_PATTERN],
        include_delimiters="before",
        flags=re.MULTILINE,
    )
    out = sp.split(reader_output)
    counts = out.metadata.get("keyword_matches", {}).get("counts", {})
    assert sum(counts.values()) == 2


def test_split_respects_regex_flags_ignorecase_expected():
    ro = ReaderOutput(
        text="todo: item\nnote: item\n",
        document_name="c.txt",
        document_path="/tmp/c.txt",
    )
    sp = KeywordSplitter(
        patterns={"todo": TODO_PATTERN, "note": NOTE_PATTERN},
        flags=re.IGNORECASE | re.MULTILINE,
        include_delimiters="before",
    )
    out = sp.split(ro)
    counts = out.metadata.get("keyword_matches", {}).get("counts", {})
    assert counts.get("todo", 0) == 1
    assert counts.get("note", 0) == 1


def test_split_no_matches_returns_single_chunk_expected():
    ro = ReaderOutput(
        text="No keywords present.",
        document_name="plain.txt",
        document_path="/tmp/plain.txt",
    )
    sp = KeywordSplitter(patterns=[r"XYZ"], include_delimiters="none")
    out = sp.split(ro)
    assert len(out.chunks) >= 1
    assert "No keywords present" in out.chunks[0]


def test_split_trims_empty_chunks_and_never_returns_empty_list_expected():
    ro = ReaderOutput(
        text="", document_name="empty.txt", document_path="/tmp/empty.txt"
    )
    sp = _make_splitter()
    out = sp.split(ro)
    assert len(out.chunks) == 1
    assert out.chunks[0] == ""


def test_split_after_mode_preserves_order_with_stitching_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="after")
    out = sp.split(reader_output)
    idx_todo = next(
        i for i, c in enumerate(out.chunks) if c.strip().startswith(TODO_FULL)
    )
    idx_note = next(
        i for i, c in enumerate(out.chunks) if c.strip().startswith(NOTE_FULL)
    )
    assert idx_todo < idx_note


def test_split_metadata_pattern_names_exposed_in_params_expected(
    reader_output: ReaderOutput,
):
    sp = _make_splitter(include="before")
    out = sp.split(reader_output)
    assert "pattern_names" in out.split_params


def test_split_both_mode_increases_chunk_count_expected(reader_output: ReaderOutput):
    sp_before = _make_splitter(include="before")
    sp_both = _make_splitter(include="both")
    out_before = sp_before.split(reader_output)
    out_both = sp_both.split(reader_output)
    assert len(out_both.chunks) >= len(out_before.chunks)
