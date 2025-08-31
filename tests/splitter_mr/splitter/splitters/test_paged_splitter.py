import pytest

from splitter_mr.schema.models import ReaderOutput, SplitterOutput
from splitter_mr.splitter.splitters.paged_splitter import PagedSplitter

# ---- Fixtures, helpers and mocks ---- #


def make_reader_output(
    text,
    page_placeholder="<!-- page -->",
    document_name="doc.md",
    document_path="tmp/doc.md",
    document_id=None,
    conversion_method=None,
    reader_method=None,
    ocr_method=None,
):
    return ReaderOutput(
        text=text,
        document_name=document_name,
        document_path=document_path,
        document_id=document_id,
        conversion_method=conversion_method,
        reader_method=reader_method,
        ocr_method=ocr_method,
        page_placeholder=page_placeholder,
    )


# ---- Tests cases ---- #


def test_init_valid():
    s = PagedSplitter(chunk_size=2, chunk_overlap=3)
    assert s.chunk_size == 2
    assert s.chunk_overlap == 3


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_init_chunk_size_invalid(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be"):
        PagedSplitter(chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, -100])
def test_init_chunk_overlap_invalid(chunk_overlap):
    with pytest.raises(ValueError, match="chunk_overlap must be"):
        PagedSplitter(chunk_overlap=chunk_overlap)


def test_split_basic_pages():
    text = "<!-- page --> First <!-- page --> Second <!-- page --> Third"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    # Three pages, each a chunk
    assert out.chunks == ["First", "Second", "Third"]
    assert out.split_method == "paged_splitter"
    assert out.split_params["chunk_size"] == 1


def test_split_multiple_pages_per_chunk():
    text = "<!-- page --> 1 <!-- page --> 2 <!-- page --> 3 <!-- page --> 4"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=2)
    out = splitter.split(ro)
    # Should group: [1 + 2], [3 + 4]
    assert out.chunks == ["1\n2", "3\n4"]


def test_split_with_overlap():
    text = "<!-- page --> abcde <!-- page --> fghij <!-- page --> klmno"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1, chunk_overlap=2)
    out = splitter.split(ro)
    # overlap 2 chars from previous chunk
    # 1st chunk: "abcde"
    # 2nd chunk: last 2 chars of "abcde" + "fghij" = "de" + "fghij" = "defghij"
    # 3rd chunk: last 2 chars of "defghij" + "klmno" = "ij" + "klmno" = "ijklmno"
    assert out.chunks == ["abcde", "defghij", "ijklmno"]


def test_split_removes_empty_pages():
    text = "<!-- page --> foo <!-- page -->   <!-- page --> bar <!-- page -->"
    ro = make_reader_output(text)
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    # empty pages between markers should be ignored
    assert out.chunks == ["foo", "bar"]


def test_split_handles_leading_trailing_whitespace():
    text = "  <!-- page -->   page1   <!-- page -->   page2   "
    ro = make_reader_output(text)
    splitter = PagedSplitter()
    out = splitter.split(ro)
    # whitespace should be stripped from chunks
    assert out.chunks == ["page1", "page2"]


def test_split_with_missing_placeholder_raises():
    text = "Just a plain text with no pages"
    ro = make_reader_output(text, page_placeholder="")
    splitter = PagedSplitter()
    with pytest.raises(ValueError, match="does not contain page placeholders"):
        splitter.split(ro)


def test_split_returns_correct_metadata():
    text = "<!-- page --> X <!-- page --> Y"
    ro = make_reader_output(text, document_name="abc", document_path="zzz")
    splitter = PagedSplitter(chunk_size=1)
    out = splitter.split(ro)
    assert out.document_name == "abc"
    assert out.document_path == "zzz"
    assert out.split_method == "paged_splitter"
    # check ids (chunk_id is generated, so just check length matches)
    assert len(out.chunks) == len(out.chunk_id)


def test_split_output_is_splitteroutput():
    text = "<!-- page --> hi"
    ro = make_reader_output(text)
    splitter = PagedSplitter()
    out = splitter.split(ro)
    assert isinstance(out, SplitterOutput)
