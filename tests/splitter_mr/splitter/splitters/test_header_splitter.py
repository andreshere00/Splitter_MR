from unittest.mock import MagicMock, patch

import pytest

from splitter_mr.splitter.splitters.header_splitter import HeaderSplitter


@pytest.fixture
def markdown_reader_output():
    return {
        "text": "# Title\n\n## Subtitle\nText for subtitle.",
        "document_name": "doc.md",
        "document_path": "/tmp/doc.md",
        "document_id": "42",
        "conversion_method": None,
        "ocr_method": None,
    }


@pytest.fixture
def html_reader_output():
    return {
        "text": "<h1>Title</h1><h2>Sub</h2><p>Body</p>",
        "document_name": "doc.html",
        "document_path": "/tmp/doc.html",
        "document_id": "99",
        "conversion_method": "html",
        "ocr_method": None,
    }


def test_uses_html_splitter_on_html_content(html_reader_output):
    # Should use HTMLHeaderTextSplitter when any tag is found
    with (
        patch(
            "splitter_mr.splitter.splitters.header_splitter.HTMLHeaderTextSplitter"
        ) as MockHTML,
        patch(
            "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
        ) as MockMD,
    ):
        mock_html = MockHTML.return_value
        mock_html.split_text.return_value = [MagicMock(page_content="HTML Chunk")]
        splitter = HeaderSplitter(
            headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
        )
        result = splitter.split(html_reader_output)
        MockHTML.assert_called_once_with(
            headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
        )
        mock_html.split_text.assert_called_once_with(html_reader_output["text"])
        MockMD.assert_not_called()
        assert result["chunks"] == ["HTML Chunk"]
        assert result["split_method"] == "header_splitter"


def test_uses_markdown_splitter_on_md_content(markdown_reader_output):
    # Should use MarkdownHeaderTextSplitter when no HTML tags found
    with (
        patch(
            "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
        ) as MockMD,
        patch(
            "splitter_mr.splitter.splitters.header_splitter.HTMLHeaderTextSplitter"
        ) as MockHTML,
    ):
        mock_md = MockMD.return_value
        mock_md.split_text.return_value = [MagicMock(page_content="MD Chunk")]
        splitter = HeaderSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )
        result = splitter.split(markdown_reader_output)
        MockMD.assert_called_once_with(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )
        mock_md.split_text.assert_called_once_with(markdown_reader_output["text"])
        MockHTML.assert_not_called()
        assert result["chunks"] == ["MD Chunk"]
        assert result["split_method"] == "header_splitter"


def test_value_error_on_bad_markdown_headers(markdown_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.MarkdownHeaderTextSplitter"
    ) as MockMD:
        MockMD.side_effect = ValueError("bad headers")
        splitter = HeaderSplitter(headers_to_split_on=[("NOPE", "bad")])
        with pytest.raises(
            ValueError, match="Incorrect Header selection for Markdown Splitter"
        ):
            splitter.split(markdown_reader_output)


def test_type_error_on_bad_html_headers(html_reader_output):
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.HTMLHeaderTextSplitter"
    ) as MockHTML:
        MockHTML.side_effect = ValueError("bad headers")
        splitter = HeaderSplitter(headers_to_split_on=[("NOPE", "bad")])
        with pytest.raises(
            TypeError, match="Incorrect Header selection for HTML splitter"
        ):
            splitter.split(html_reader_output)


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
            assert field in result


def test_html_dispatch_on_html_fragment():
    # Should dispatch to HTML splitter even for small fragments
    html_fragment = "<div>Only one tag</div>"
    reader_output = {
        "text": html_fragment,
        "document_name": "foo.txt",
        "document_path": "/tmp/foo.txt",
        "document_id": "123",
    }
    with patch(
        "splitter_mr.splitter.splitters.header_splitter.HTMLHeaderTextSplitter"
    ) as MockHTML:
        mock_html = MockHTML.return_value
        mock_html.split_text.return_value = [MagicMock(page_content="HTML Chunk")]
        splitter = HeaderSplitter(headers_to_split_on=[("div", "Div")])
        result = splitter.split(reader_output)
        MockHTML.assert_called_once()
        assert result["chunks"] == ["HTML Chunk"]
