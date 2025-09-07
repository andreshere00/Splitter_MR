import copy
from typing import List, Optional

from bs4 import BeautifulSoup

from ...reader.utils.html_to_markdown import HtmlToMarkdown
from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class HTMLTagSplitter(BaseSplitter):
    """
    HTMLTagSplitter splits HTML content into chunks based on a specified tag.
    Supports batching and optional Markdown conversion.

    Behavior:
      - When `tag` is specified (e.g., tag="div"), finds all matching elements.
      - When `tag` is None, splits by the most frequent and shallowest tag.

    Args:
        chunk_size (int): Maximum chunk size in characters (only used when `batch=True`).
        tag (str | None): HTML tag to split on. If None, auto-detects the best tag.
        batch (bool): If True (default), groups multiple tags into a chunk, not exceeding `chunk_size`.
            If False, returns one chunk per tag, ignoring chunk_size.
        to_markdown (bool): If True, converts each chunk to Markdown using HtmlToMarkdown.

    Example:
        >>> reader_output = ReaderOutput(text="<div>A</div><div>B</div>")
        >>> splitter = HTMLTagSplitter(tag="div", batch=False)
        >>> splitter.split(reader_output).chunks
        ['<html><body><div>A</div></body></html>', '<html><body><div>B</div></body></html>']
        >>> splitter = HTMLTagSplitter(tag="div", batch=True, chunk_size=100)
        >>> splitter.split(reader_output).chunks
        ['<html><body><div>A</div><div>B</div></body></html>']
        >>> splitter = HTMLTagSplitter(tag="div", batch=False, to_markdown=True)
        >>> splitter.split(reader_output).chunks
        ['A', 'B']

    Attributes:
        chunk_size (int): Maximum chunk size.
        tag (Optional[str]): Tag to split on.
        batch (bool): Whether to group elements into chunks.
        to_markdown (bool): Whether to convert each chunk to Markdown.
    """

    def __init__(
        self,
        chunk_size: int = 1,
        tag: Optional[str] = None,
        *,
        batch: bool = True,
        to_markdown: bool = True,
    ):
        """
        Initialize HTMLTagSplitter.

        Args:
            chunk_size (int): Maximum chunk size, in characters (only for batching).
            tag (str | None): Tag to split on. If None, auto-detects.
            batch (bool): If True (default), groups tags up to `chunk_size`.
            to_markdown (bool): If True (default), convert each chunk to Markdown.
        """
        super().__init__(chunk_size)
        self.tag = tag
        self.batch = batch
        self.to_markdown = to_markdown

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits HTML using the specified tag and batching, with optional Markdown conversion.

        Semantics:
        - Tables:
            * batch=False -> one chunk per requested element. If splitting by a row-level tag
                (e.g. 'tr'), emit a mini-table per row: <thead> once + that row in <tbody>.
            * batch=True and chunk_size in (0, 1, None) -> all tables in one chunk.
            * batch=True and chunk_size > 1 -> split each table into multiple chunks
                by batching <tr> rows (copying a <thead> into every chunk and
                skipping the header row from the body).
        - Non-table tags:
            * batch=False -> one chunk per element.
            * batch=True and chunk_size in (0, 1, None) -> all elements in one chunk.
            * batch=True and chunk_size > 1 -> batch by total HTML length.

        Args:
            reader_output: ReaderOutput containing at least `text`.

        Returns:
            SplitterOutput
        """
        html = getattr(reader_output, "text", "") or ""
        soup = BeautifulSoup(html, "html.parser")
        tag = self.tag or self._auto_tag(soup)

        # Locate elements for the chosen tag.
        try:
            elements = soup.find_all(tag)
            table_children = {"tr", "thead", "tbody", "th", "td"}
            # Only escalate to table when batching is enabled. For non-batch,
            # keep the exact tag so we can emit one chunk per element.
            if self.batch and tag in table_children:
                seen = set()
                parent_tables = []
                for el in elements:
                    table = el.find_parent("table")
                    if table and id(table) not in seen:
                        seen.add(id(table))
                        parent_tables.append(table)
                if parent_tables:
                    elements = parent_tables
                    tag = "table"
        except Exception:
            elements = []

        # -------- helpers -------- #

        def build_doc_with_children(children: List) -> str:
            """Wrap a list of top-level nodes into <html><body>…</body></html>."""
            doc = BeautifulSoup("", "html.parser")
            html_tag = doc.new_tag("html")
            body_tag = doc.new_tag("body")
            html_tag.append(body_tag)
            doc.append(html_tag)
            for c in children:
                body_tag.append(copy.deepcopy(c))
            return str(doc)

        def extract_table_header_and_rows(table_tag):
            """
            Return (header_thead, data_rows, header_row_src) where:
            - header_thead is a <thead> (deep-copied) or None
            - data_rows is a list of original <tr> nodes that are NOT header rows
            - header_row_src is the original <tr> used to synthesize <thead> (if any)
            """
            header = table_tag.find("thead")
            header_row_src = None

            if header is not None:
                data_rows = []
                for tr in table_tag.find_all("tr"):
                    if tr.find_parent("thead") is not None:
                        continue
                    data_rows.append(tr)
                return copy.deepcopy(header), data_rows, None

            first_tr = table_tag.find("tr")
            header_thead = None
            if first_tr is not None:
                tmp = BeautifulSoup("", "html.parser")
                thead = tmp.new_tag("thead")
                thead.append(copy.deepcopy(first_tr))
                header_thead = thead
                header_row_src = first_tr

            data_rows = []
            for tr in table_tag.find_all("tr"):
                if header_row_src is not None and tr is header_row_src:
                    continue
                if tr.find_parent("thead") is not None:
                    continue
                data_rows.append(tr)

            return header_thead, data_rows, header_row_src

        def build_table_chunk(table_tag, rows_subset: List) -> str:
            """
            Build a <html><body><table>… chunk with:
            - original table attributes
            - a <thead> (original or synthesized)
            - a <tbody> containing rows_subset
            """
            header_thead, _, _ = extract_table_header_and_rows(table_tag)
            doc = BeautifulSoup("", "html.parser")
            html_tag = doc.new_tag("html")
            body_tag = doc.new_tag("body")
            html_tag.append(body_tag)
            doc.append(html_tag)

            new_table = doc.new_tag("table", **table_tag.attrs)
            if header_thead is not None:
                new_table.append(copy.deepcopy(header_thead))

            tbody = doc.new_tag("tbody")
            for r in rows_subset:
                tbody.append(copy.deepcopy(r))
            new_table.append(tbody)

            body_tag.append(new_table)
            return str(doc)

        # -------- main chunking -------- #

        chunks: List[str] = []

        if tag == "table":
            # TABLES: custom batching
            if not self.batch:
                # one chunk per table (full)
                chunks = [build_doc_with_children([el]) for el in elements]

            elif self.chunk_size in (0, 1, None):
                # all tables together
                chunks = [build_doc_with_children(elements)] if elements else [""]

            else:
                # batch rows within each table
                for table_el in elements:
                    header_thead, rows, _ = extract_table_header_and_rows(table_el)
                    if not rows:
                        chunks.append(build_doc_with_children([table_el]))
                        continue

                    buf: List = []
                    for row in rows:
                        test_buf = buf + [row]
                        test_html = build_table_chunk(table_el, test_buf)
                        if len(test_html) > self.chunk_size and buf:
                            chunks.append(build_table_chunk(table_el, buf))
                            buf = [row]
                        else:
                            buf = test_buf
                    if buf:
                        chunks.append(build_table_chunk(table_el, buf))

        else:
            # NON-TABLE (including table children when batch=False)
            table_children = {"tr", "thead", "tbody", "th", "td"}

            if not self.batch:
                if tag in table_children:
                    # one chunk per row-like element, but keep header context
                    for el in elements:
                        table_el = el.find_parent("table")
                        if not table_el:
                            # Fallback: wrap the element as-is
                            chunks.append(build_doc_with_children([el]))
                            continue
                        # skip header-only rows
                        if el.name == "tr" and el.find_parent("thead") is not None:
                            continue
                        if el.name in {"thead", "th"}:
                            continue
                        chunks.append(build_table_chunk(table_el, [el]))
                else:
                    for el in elements:
                        chunks.append(build_doc_with_children([el]))

            elif self.chunk_size in (0, 1, None):
                chunks = [build_doc_with_children(elements)] if elements else [""]

            else:
                buffer = []
                for el in elements:
                    test_buffer = buffer + [el]
                    test_chunk_str = build_doc_with_children(test_buffer)
                    if len(test_chunk_str) > self.chunk_size and buffer:
                        chunks.append(build_doc_with_children(buffer))
                        buffer = [el]
                    else:
                        buffer = test_buffer
                if buffer:
                    chunks.append(build_doc_with_children(buffer))

        if not chunks:
            chunks = [""]

        if self.to_markdown:
            md = HtmlToMarkdown()
            chunks = [md.convert(chunk) for chunk in chunks]

        chunk_ids = self._generate_chunk_ids(len(chunks))
        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="html_tag_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "tag": tag,
                "batch": self.batch,
                "to_markdown": self.to_markdown,
            },
            metadata=self._default_metadata(),
        )

    def _auto_tag(self, soup: BeautifulSoup) -> str:
        """
        Auto-detect the most repeated tag with the highest (shallowest) level of hierarchy.
        If no repeated tags are found, return the first tag found in <body> or fallback to 'div'.
        """
        from collections import Counter, defaultdict

        body = soup.find("body")
        if not body:
            return "div"

        # Traverse all tags in body, tracking tag: (count, min_depth)
        tag_counter = Counter()
        tag_min_depth = defaultdict(lambda: float("inf"))

        def traverse(el, depth=0):
            for child in el.children:
                if getattr(child, "name", None):
                    tag_counter[child.name] += 1
                    tag_min_depth[child.name] = min(tag_min_depth[child.name], depth)
                    traverse(child, depth + 1)

        traverse(body)

        if not tag_counter:
            # fallback to first tag
            for t in body.find_all(True, recursive=True):
                return t.name
            return "div"

        # Find tags with the maximum count
        max_count = max(tag_counter.values())
        candidates = [t for t, cnt in tag_counter.items() if cnt == max_count]
        # Of the most frequent, pick the one with the minimum depth (shallowest)
        chosen = min(candidates, key=lambda t: tag_min_depth[t])
        return chosen
