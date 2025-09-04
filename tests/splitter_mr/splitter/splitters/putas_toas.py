from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter import HeaderSplitter

SAMPLE_MD = """# Sample Markdown

This is some basic, sample markdown.

## Second Heading

- Unordered lists, and:
  1. One
  1. Two
  1. Three
- More
> Blockquote

And **bold**, *italics*, and even *italics and later **bold***. Even ~~strikethrough~~. [A link](https://markdowntohtml.com) to somewhere.

And code highlighting:

```js
var foo = 'bar';

function baz(s) {
   return foo + ':' + s;
}
````

Or inline code like `var foo = 'bar';`.

Or an image of bears

![bears](http://placebear.com/200/200)

The end ...
"""

SAMPLE_HTML = """

<h1 id="sample-markdown">Sample Markdown</h1>
<p>This is some basic, sample markdown.</p>
<h2 id="second-heading">Second Heading</h2>
<ul>
<li>Unordered lists, and:<ol>
<li>One</li>
<li>Two</li>
<li>Three</li>
</ol>
</li>
<li>More</li>
</ul>
<blockquote>
<p>Blockquote</p>
</blockquote>
<p>And <strong>bold</strong>, <em>italics</em>, and even <em>italics and later <strong>bold</strong></em>. Even <del>strikethrough</del>. <a href="https://markdowntohtml.com">A link</a> to somewhere.</p>
<p>And code highlighting:</p>
<pre><code class="lang-js">var foo = 'bar';

function baz(s) {
return foo + ':' + s;
} </code></pre>

<p>Or inline code like <code>var foo = &#39;bar&#39;;</code>.</p>
<p>Or an image of bears</p>
<p><img src="http://placebear.com/200/200" alt="bears"></p>
<p>The end ...</p>
"""


@pytest.fixture
def markdown_reader_output():
    return ReaderOutput(
        text=SAMPLE_MD,
        document_name="doc.md",
        document_path="/tmp/doc.md",
        document_id="42",
        conversion_method=None,
        ocr_method=None,
    )


@pytest.fixture
def html_reader_output():
    return ReaderOutput(
        text=SAMPLE_HTML,
        document_name="doc.html",
        document_path="/tmp/doc.html",
        document_id="99",
        conversion_method="html",
        ocr_method=None,
    )


def test_markdown_splitter_on_md_content(markdown_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [
            MagicMock(page_content="Chunk 1"),
            MagicMock(page_content="Chunk 2"),
        ]
        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
        result = splitter.split(markdown_reader_output)
        MockMD.assert_called_once_with(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
            return_each_line=False,
            strip_headers=False,
        )
        mock_md.split_text.assert_called_once_with(SAMPLE_MD)
        assert result.chunks == ["Chunk 1", "Chunk 2"]
        assert result.split_method == "header_splitter"


def test_html_conversion_and_split(monkeypatch, html_reader_output):
    # Patch MarkdownHeaderTextSplitter
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [
            MagicMock(page_content="Converted chunk 1"),
            MagicMock(page_content="Converted chunk 2"),
        ]
    # Patch HtmlToMarkdown to check it is used
    with patch("splitter_mr.splitter.splitters.utils.HtmlToMarkdown") as MockHtmlToMd:
        mock_converter = MockHtmlToMd.return_value
        # Return known markdown on conversion
        mock_converter.convert.return_value = SAMPLE_MD
        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
        result = splitter.split(html_reader_output)
        # Should have converted HTML to markdown before splitting
        mock_converter.convert.assert_called_once_with(SAMPLE_HTML)
        # Should have split the produced markdown
        mock_md.split_text.assert_called_once_with(SAMPLE_MD)
        assert result.chunks == ["Converted chunk 1", "Converted chunk 2"]
        assert result.split_method == "header_splitter"


def test_value_error_on_bad_semantic_header(markdown_reader_output):
    splitter = HeaderSplitter(headers_to_split_on=["NOPE", "Header 2"])
    with pytest.raises(ValueError, match="Invalid header: NOPE"):
        splitter.split(markdown_reader_output)


def test_output_metadata_fields(markdown_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [MagicMock(page_content="chunk")]
        splitter = HeaderSplitter()
        result = splitter.split(markdown_reader_output)

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


def test_empty_text_raises_value_error():
    ro = ReaderOutput(
        text="",
        document_name="empty.md",
        document_path="/tmp/empty.md",
        document_id="1",
    )
    splitter = HeaderSplitter()
    with pytest.raises(ValueError, match="empty or None"):
        splitter.split(ro)


def test_guess_filetype_logic():
    # .md extension
    ro = ReaderOutput(
        text="# test", document_name="foo.md", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "md"
    # .html extension
    ro = ReaderOutput(
        text="<h1>x</h1>", document_name="foo.html", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "html"
    # HTML content, but not extension
    ro = ReaderOutput(
        text="<h1>x</h1>", document_name="foo.txt", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "html"
    # Markdown by default
    ro = ReaderOutput(
        text="# X", document_name="foo.txt", document_path="", document_id="1"
    )
    assert HeaderSplitter._guess_filetype(ro) == "md"
