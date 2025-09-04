from types import SimpleNamespace

from splitter_mr.splitter.splitters.html_tag_splitter import HTMLTagSplitter

# ---- Mocks, fixtures and helpers ---- #


def make_reader_output(html: str) -> SimpleNamespace:
    """Create a minimal ReaderOutput-like object for tests."""
    return SimpleNamespace(
        text=html,
        document_name="doc.html",
        document_path="/tmp/doc.html",
        document_id="doc-1",
        conversion_method="markdown",
        reader_method="docling",
        ocr_method=None,
    )


# ---- Test cases ---- #


def test_init_sets_tag_and_chunk_size():
    s = HTMLTagSplitter(chunk_size=123, tag="div", to_markdown=False)
    assert s.chunk_size == 123
    assert s.tag == "div"


def test_split_with_explicit_tag_simple_divs():
    html = "<html><body><div>A</div><div>B</div></body></html>"
    ro = make_reader_output(html)

    # Batching enabled (default), large chunk_size -> both divs in a single chunk
    splitter = HTMLTagSplitter(chunk_size=10000, tag="div", to_markdown=False)
    out = splitter.split(ro)
    assert out.split_method == "html_tag_splitter"
    assert out.split_params["tag"] == "div"
    assert out.split_params["chunk_size"] == 10000
    assert out.split_params["batch"] is True
    assert out.split_params["to_markdown"] is False
    assert len(out.chunks) == 1
    assert ">A<" in out.chunks[0]
    assert ">B<" in out.chunks[0]
    assert len(out.chunk_id) == 1

    # No batching: one chunk per element (ignore chunk_size)
    splitter = HTMLTagSplitter(chunk_size=1, tag="div", batch=False, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["batch"] is False
    assert len(out.chunks) == 2
    assert any(">A<" in c for c in out.chunks)
    assert any(">B<" in c for c in out.chunks)


def test_auto_tag_picks_most_frequent_then_shallowest():
    html = """
    <html><body>
      <section><p>One</p></section>
      <section><p>Two</p></section>
    </body></html>
    """
    ro = make_reader_output(html)

    # With batching (default), both sections in one chunk
    splitter = HTMLTagSplitter(chunk_size=10000, tag=None, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["tag"] == "section"
    assert out.split_params["batch"] is True
    assert len(out.chunks) == 1
    assert ">One<" in out.chunks[0]
    assert ">Two<" in out.chunks[0]

    # Without batching, one per section
    splitter = HTMLTagSplitter(chunk_size=1, tag=None, batch=False, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["batch"] is False
    assert len(out.chunks) == 2
    assert any(">One<" in c for c in out.chunks)
    assert any(">Two<" in c for c in out.chunks)


def test_auto_tag_no_body_fallback_to_div_produces_no_chunks():
    html = "<html><head></head></html>"
    ro = make_reader_output(html)
    splitter = HTMLTagSplitter(chunk_size=10000, tag=None, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["tag"] == "div"
    assert out.chunks == [""]


def test_auto_tag_body_without_children_fallback_to_div():
    html = "<html><body><!-- empty --></body></html>"
    ro = make_reader_output(html)
    splitter = HTMLTagSplitter(chunk_size=10000, tag=None, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["tag"] == "div"
    assert out.chunks == [""]


def test_auto_tag_fallback_to_first_tag_when_no_repeats():
    html = "<html><body><article>Only</article></body></html>"
    ro = make_reader_output(html)
    splitter = HTMLTagSplitter(chunk_size=10000, tag=None, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["tag"] == "article"
    assert len(out.chunks) == 1
    assert "Only" in out.chunks[0]


def test_table_with_thead_tr_tag_includes_header_once():
    html = """
    <html><body>
      <table id="t1">
        <thead><tr><th>H1</th><th>H2</th></tr></thead>
        <tbody>
          <tr><td>A1</td><td>A2</td></tr>
          <tr><td>B1</td><td>B2</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)

    # With batching (default): rows fit into chunk_size → one chunk
    splitter = HTMLTagSplitter(chunk_size=10000, tag="tr", to_markdown=False)
    out = splitter.split(ro)
    # The splitter escalates 'tr' to 'table' for proper table handling

    print(out)
    assert out.split_params["tag"] == "table"
    assert len(out.chunks) == 1
    assert "<tr><th>H1</th><th>H2</th></tr>" in out.chunks[0]
    assert "<tr><td>A1</td><td>A2</td></tr>" in out.chunks[0]
    assert "<tr><td>B1</td><td>B2</td></tr>" in out.chunks[0]

    # Without batching: per current behavior, one chunk per TABLE (full table), not per row
    splitter = HTMLTagSplitter(chunk_size=1, tag="tr", batch=False, to_markdown=False)
    out = splitter.split(ro)
    print(out)
    assert out.split_params["tag"] == "tr"
    assert len(out.chunks) == 2
    assert "<tr><th>H1</th><th>H2</th></tr>" in out.chunks[0]
    assert "<tr><td>A1</td><td>A2</td></tr>" in out.chunks[0]
    assert "<tr><td>B1</td><td>B2</td></tr>" in out.chunks[1]


def test_element_is_table_header_path_no_duplication():
    html = """
    <html><body>
      <table>
        <thead><tr><th>A</th></tr></thead>
        <tbody><tr><td>1</td></tr></tbody>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)
    # Default batching, big chunk_size -> single chunk with one <thead>
    splitter = HTMLTagSplitter(chunk_size=10000, tag="table", to_markdown=False)
    out = splitter.split(ro)
    assert len(out.chunks) == 1
    # We do not duplicate the header: only the original thead exists
    assert out.chunks[0].count("<thead") == 1


def test_chunking_splits_when_length_exceeds_chunk_size():
    # Build many small divs so that the serialized chunk exceeds a tiny chunk_size
    items = "".join(f"<div>Item {i:02d}</div>" for i in range(12))
    html = f"<html><body>{items}</body></html>"
    ro = make_reader_output(html)

    # Batching with small threshold should produce multiple chunks
    splitter = HTMLTagSplitter(chunk_size=160, tag="div", to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["batch"] is True
    assert len(out.chunks) > 1
    # Ensure order preserved: first chunk has early items; last chunk has later items
    assert "Item 00" in out.chunks[0]
    assert "Item 11" in out.chunks[-1]

    # No batching: one per div (ignores chunk_size)
    splitter = HTMLTagSplitter(chunk_size=1, tag="div", batch=False, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["batch"] is False
    assert len(out.chunks) == 12
    assert "Item 00" in out.chunks[0]
    assert "Item 11" in out.chunks[-1]


def test_no_matching_elements_produces_empty_chunks_and_ids():
    html = "<html><body><span>No divs here</span></body></html>"
    ro = make_reader_output(html)
    splitter = HTMLTagSplitter(tag="div", chunk_size=10000, to_markdown=False)
    out = splitter.split(ro)
    assert out.chunks == [""]
    assert out.split_params["tag"] == "div"


def test_all_in_one_when_chunk_size_is_one_and_batch_true_non_table():
    html = "<html><body><div>A</div><div>B</div><div>C</div></body></html>"
    ro = make_reader_output(html)

    # batch=True and chunk_size=1 -> all elements grouped in ONE chunk
    splitter = HTMLTagSplitter(chunk_size=1, tag="div", batch=True, to_markdown=False)
    out = splitter.split(ro)
    assert out.split_params["batch"] is True
    assert out.split_params["chunk_size"] == 1
    assert len(out.chunks) == 1
    assert all(x in out.chunks[0] for x in [">A<", ">B<", ">C<"])


def test_table_batching_small_chunk_size_creates_multiple_chunks_and_copies_header():
    html = """
    <html><body>
      <table id="t1">
        <thead><tr><th>H1</th></tr></thead>
        <tbody>
          <tr><td>A</td></tr>
          <tr><td>B</td></tr>
          <tr><td>C</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)

    # Force multiple chunks by making chunk_size tiny
    splitter = HTMLTagSplitter(chunk_size=120, tag="tr", batch=True, to_markdown=False)
    out = splitter.split(ro)

    assert out.split_params["tag"] == "table"  # escalated
    assert len(out.chunks) >= 2  # multiple chunks
    # Header must be present in every chunk
    for chunk in out.chunks:
        assert "<thead><tr><th>H1</th></tr></thead>" in chunk
        assert "<tbody>" in chunk and "</tbody>" in chunk
    # Together, all rows should be covered across chunks
    joined = "".join(out.chunks)
    for cell in ["<td>A</td>", "<td>B</td>", "<td>C</td>"]:
        assert cell in joined


def test_to_markdown_true_non_table_converts_html_to_md():
    html = "<html><body><div>Hi <strong>there</strong>!</div></body></html>"
    ro = make_reader_output(html)

    splitter = HTMLTagSplitter(
        chunk_size=10000, tag="div", batch=True, to_markdown=True
    )
    out = splitter.split(ro)

    assert out.split_params["to_markdown"] is True
    # Should not look like raw HTML; should contain markdown formatting
    assert "**there**" in out.chunks[0]
    assert "<html>" not in out.chunks[0]


def test_to_markdown_true_table_converts_table_to_md():
    html = """
    <html><body>
      <table>
        <tr><th>X</th><th>Y</th></tr>
        <tr><td>1</td><td>2</td></tr>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)

    # With batch=False and tag='table', we get 1 chunk per table; then MD conversion kicks in
    splitter = HTMLTagSplitter(chunk_size=1, tag="table", batch=False, to_markdown=True)
    out = splitter.split(ro)
    assert out.split_params["tag"] == "table"
    md = out.chunks[0]
    # Markdown table header and separator
    assert "| X | Y |" in md
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md


def test_non_batch_header_only_tags_are_skipped_for_row_emission():
    html = """
    <html><body>
      <table>
        <thead><tr><th>H</th></tr></thead>
        <tbody><tr><td>R1</td></tr><tr><td>R2</td></tr></tbody>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)

    # tag='thead' with batch=False should not emit a chunk for the header-only piece
    splitter = HTMLTagSplitter(
        chunk_size=1, tag="thead", batch=False, to_markdown=False
    )
    out = splitter.split(ro)
    # No chunks because we skip header-only elements entirely under this path
    assert (
        out.chunks == [""]
        or len(out.chunks) == 0
        or all("<td>" not in c for c in out.chunks)
    )

    # tag='th' with batch=False also skipped
    splitter = HTMLTagSplitter(chunk_size=1, tag="th", batch=False, to_markdown=False)
    out = splitter.split(ro)
    assert out.chunks == [""] or len(out.chunks) == 0


def test_table_without_rows_emits_table_even_when_batching_rows():
    html = """
    <html><body>
      <table id="empty">
        <thead><tr><th>H</th></tr></thead>
        <tbody></tbody>
      </table>
    </body></html>
    """
    ro = make_reader_output(html)
    splitter = HTMLTagSplitter(
        chunk_size=50, tag="table", batch=True, to_markdown=False
    )
    out = splitter.split(ro)
    # No <tr> rows to batch -> still outputs the (wrapped) table
    assert len(out.chunks) == 1
    assert '<table id="empty">' in out.chunks[0]


def test_find_all_exception_yields_empty_chunk(monkeypatch):
    # Make BeautifulSoup.find_all raise to exercise the except branch
    html = "<html><body><div>A</div></body></html>"
    ro = make_reader_output(html)

    # Monkeypatch the method on a Soup instance created inside split; we patch bs4.BeautifulSoup.find_all
    class Boom(Exception):
        pass

    orig_find_all = None

    import bs4

    orig_find_all = bs4.BeautifulSoup.find_all

    def raiser(self, *args, **kwargs):
        raise Boom("boom")

    try:
        bs4.BeautifulSoup.find_all = raiser
        splitter = HTMLTagSplitter(chunk_size=10000, tag="div", to_markdown=False)
        out = splitter.split(ro)
        assert out.chunks == [""]
    finally:
        # restore
        bs4.BeautifulSoup.find_all = orig_find_all


def test_auto_tag_same_count_picks_shallowest():
    html = """
    <html><body>
      <section><p>A</p></section>
      <div><p>B</p></div>
      <!-- Make counts equal for 'p' and another tag, but ensure 'section' and 'div' tie → choose shallowest -->
      <wrapper>
        <section><p>C</p></section>
        <div><p>D</p></div>
      </wrapper>
    </body></html>
    """
    ro = make_reader_output(html)

    # We don't specify tag; auto_tag should choose the most frequent, then shallowest among ties.
    splitter = HTMLTagSplitter(chunk_size=1, tag=None, batch=False, to_markdown=False)
    out = splitter.split(ro)
    print(out)
    assert out.split_params["tag"] == "p"
    # With batch=False, one chunk per chosen element
    assert len(out.chunks) == 4
