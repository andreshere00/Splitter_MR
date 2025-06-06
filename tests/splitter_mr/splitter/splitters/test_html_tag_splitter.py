import pytest

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
    return {
        "text": basic_html,
        "document_name": "sample.html",
        "document_path": "/tmp/sample.html",
        "document_id": "123",
        "conversion_method": "html",
        "ocr_method": None,
    }


def test_split_with_explicit_tag(reader_output):
    splitter = HTMLTagSplitter(chunk_size=1000, tag="div")
    result = splitter.split(reader_output)
    assert "chunks" in result
    # There are three divs, so three chunks
    assert len(result["chunks"]) == 3
    # Each chunk should contain exactly one <div>
    for chunk in result["chunks"]:
        assert chunk.count("<div>") == 1
        assert chunk.count("<p>") == 1
    assert result["split_params"]["tag"] == "div"
    assert result["split_method"] == "html_tag_splitter"


def test_split_with_auto_tag(reader_output):
    splitter = HTMLTagSplitter(chunk_size=1000)  # No tag specified
    result = splitter.split(reader_output)
    # Should auto-detect <div> as most frequent/shallowest
    assert len(result["chunks"]) == 3
    for chunk in result["chunks"]:
        assert "<div>" in chunk
    assert result["split_params"]["tag"] == "div"


def test_split_with_shallowest_tag():
    html = (
        "<html><body>"
        "<section><div>Div1</div></section>"
        "<section><div>Div2</div></section>"
        "</body></html>"
    )
    reader_output = {"text": html}
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # <section> is shallower and as frequent as <div>
    assert all("<section>" in chunk for chunk in result["chunks"])
    assert result["split_params"]["tag"] == "section"


def test_split_without_body_tag():
    html = "<div><span>No body tag here</span></div>"
    reader_output = {"text": html}
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # Fallback to <div>
    assert len(result["chunks"]) == 1
    assert "<div>" in result["chunks"][0]
    assert result["split_params"]["tag"] == "div"


def test_split_with_no_repeated_tags():
    html = "<html><body><header>Header</header><footer>Footer</footer></body></html>"
    reader_output = {"text": html}
    splitter = HTMLTagSplitter(chunk_size=1000)
    result = splitter.split(reader_output)
    # Only <header> and <footer> exist, so pick the first (header)
    assert len(result["chunks"]) == 1
    assert "<header>" in result["chunks"][0] or "<footer>" in result["chunks"][0]
    assert result["split_params"]["tag"] in ("header", "footer")


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
        assert field in result


def test_empty_html():
    splitter = HTMLTagSplitter(chunk_size=1000, tag="div")
    reader_output = {"text": ""}
    result = splitter.split(reader_output)
    assert result["chunks"] == []
