import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import HTMLTagSplitter


@pytest.fixture
def basic_html():
    return (
        "<html><body>"
        "<div><p>First para</p></div>"
        "<div><p>Second para</p></div>"
        "<div><p>Third para</p></div>"
        "</body></html>"
    )


@pytest.fixture
def reader_output(basic_html):
    return ReaderOutput(
        text=basic_html,
        document_name="sample.html",
        document_path="/tmp/sample.html",
        document_id="123",
        conversion_method="html",
        ocr_method=None,
    )


def test_split_with_explicit_tag(reader_output):
    splitter = HTMLTagSplitter(chunk_size=1000, tag="div")
    result = splitter.split(reader_output)
    assert hasattr(result, "chunks")
    # All three <div> fit in one chunk under chunk_size
    assert len(result.chunks) == 1
    # The single chunk contains all three divs
    assert result.chunks[0].count("<div>") == 3
    assert result.chunks[0].count("<p>") == 3
    assert result.split_params["tag"] == "div"
    assert result.split_method == "html_tag_splitter"


def test_split_with_auto_tag(reader_output):
    splitter = HTMLTagSplitter(chunk_size=1000)  # No tag specified
    result = splitter.split(reader_output)
    # All three <div> fit in one chunk under chunk_size
    assert len(result.chunks) == 1
    assert result.chunks[0].count("<div>") == 3
    assert result.split_params["tag"] == "div"


def test_split_with_shallowest_tag():
    html = (
        "<html><body>"
        "<section><div>Div1</div></section>"
        "<section><div>Div2</div></section>"
        "</body></html>"
    )
    reader_output = ReaderOutput(text=html)
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # <section> is shallower and as frequent as <div>
    assert all("<section>" in chunk for chunk in result.chunks)
    assert result.split_params["tag"] == "section"


def test_split_without_body_tag():
    html = "<div><span>No body tag here</span></div>"
    reader_output = ReaderOutput(text=html)
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # Fallback to <div>
    assert len(result.chunks) == 1
    assert "<div>" in result.chunks[0]
    assert result.split_params["tag"] == "div"


def test_split_with_no_repeated_tags():
    html = "<html><body><header>Header</header><footer>Footer</footer></body></html>"
    reader_output = ReaderOutput(text=html)
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # Only <header> and <footer> exist, so pick the first (header)
    assert len(result.chunks) == 1
    assert "<header>" in result.chunks[0] or "<footer>" in result.chunks[0]
    assert result.split_params["tag"] in ("header", "footer")


def test_output_contains_metadata(reader_output):
    splitter = HTMLTagSplitter(chunk_size=1000, tag="div")
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


def test_empty_html():
    splitter = HTMLTagSplitter(chunk_size=1000, tag="div")
    reader_output = ReaderOutput(text="")
    result = splitter.split(reader_output)
    assert result.chunks == []
