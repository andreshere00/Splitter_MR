import re
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ...reader.utils import HtmlToMarkdown
from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class HeaderSplitter(BaseSplitter):
    """
    Split HTML or Markdown documents into chunks by header levels (H1–H6).

    - If the input looks like HTML, it is first converted to Markdown using the
      project's HtmlToMarkdown utility, which emits ATX-style headings (`#`, `##`, ...).
    - If the input is Markdown, Setext-style headings (underlines with `===` / `---`)
      are normalized to ATX so headers are reliably detected.
    - Splitting is performed with LangChain's MarkdownHeaderTextSplitter.
    - If no headers are detected after conversion/normalization, a safe fallback
      splitter (RecursiveCharacterTextSplitter) is used to avoid returning a single,
      excessively large chunk.

    Args:
        chunk_size (int, optional): Size hint for fallback splitting; not used by
            header splitting itself. Defaults to 1000.
        headers_to_split_on (Optional[List[str]]): Semantic header names like
            ["Header 1", "Header 2"]. If None, all levels 1–6 are enabled.
        group_header_with_content (bool, optional): If True (default), headers are
            kept with their following content (strip_headers=False). If False,
            headers are stripped from chunks (strip_headers=True).

    Example:
        ```python
        from splitter_mr.splitter import HeaderSplitter

        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2", "Header 3"])
        output = splitter.split(reader_output)  # reader_output.text may be HTML or MD
        for idx, chunk in enumerate(output.chunks):
            print(f"--- Chunk {idx+1} ---")
            print(chunk)
        ```
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        headers_to_split_on: Optional[List[str]] = None,
        *,
        group_header_with_content: bool = True,
    ):
        """
        Initialize the HeaderSplitter.

        Args:
            chunk_size (int): Used by fallback character splitter if no headers are found.
            headers_to_split_on (Optional[List[str]]): Semantic headers, e.g. ["Header 1", "Header 2"].
                Defaults to all levels 1–6.
            group_header_with_content (bool): Keep headers attached to following content if True.
        """
        super().__init__(chunk_size)
        # Default to all 6 levels for robust splitting unless caller narrows it.
        self.headers_to_split_on = headers_to_split_on or [
            f"Header {i}" for i in range(1, 7)
        ]
        self.group_header_with_content = bool(group_header_with_content)

    def _make_tuples(self, filetype: str) -> List[Tuple[str, str]]:
        """
        Convert semantic header names (e.g., "Header 2") into Markdown tokens.

        Args:
            filetype (str): Only "md" is supported (HTML is converted to MD first).

        Returns:
            List[Tuple[str, str]]: Tuples of (header_token, semantic_name), e.g. ("##", "Header 2").
        """
        tuples: List[Tuple[str, str]] = []
        for header in self.headers_to_split_on:
            lvl = self._header_level(header)
            if filetype == "md":
                tuples.append(("#" * lvl, header))
            else:
                raise ValueError(f"Unsupported filetype: {filetype!r}")
        return tuples

    @staticmethod
    def _header_level(header: str) -> int:
        """
        Extract numeric level from a header name like "Header 2".

        Raises:
            ValueError: If the header string is not of the expected form.
        """
        m = re.match(r"header\s*(\d+)", header.lower())
        if not m:
            raise ValueError(f"Invalid header: {header}")
        return int(m.group(1))

    @staticmethod
    def _guess_filetype(reader_output: ReaderOutput) -> str:
        """
        Heuristically determine whether the input is HTML or Markdown.

        Checks filename extensions first, then looks for HTML elements as a hint.
        """
        name = (reader_output.document_name or "").lower()
        if name.endswith((".html", ".htm")):
            return "html"
        if name.endswith((".md", ".markdown")):
            return "md"

        soup = BeautifulSoup(reader_output.text or "", "html.parser")
        if soup.find("html") or soup.find(re.compile(r"^h[1-6]$")) or soup.find("div"):
            return "html"
        return "md"

    @staticmethod
    def _normalize_setext(md_text: str) -> str:
        """
        Normalize Setext-style headings to ATX so MarkdownHeaderTextSplitter can detect them.

        H1:  Title\\n====  →  # Title
        H2:  Title\\n----  →  ## Title
        """
        # H1 underlines
        md_text = re.sub(r"^(?P<t>[^\n]+)\n=+\s*$", r"# \g<t>", md_text, flags=re.M)
        # H2 underlines
        md_text = re.sub(r"^(?P<t>[^\n]+)\n-+\s*$", r"## \g<t>", md_text, flags=re.M)
        return md_text

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Perform header-based splitting with HTML→Markdown conversion and safe fallback.

        Steps:
          1) Detect filetype (HTML/MD).
          2) If HTML, convert to Markdown with HtmlToMarkdown (emits ATX headings).
          3) If Markdown, normalize Setext headings to ATX.
          4) Split by headers via MarkdownHeaderTextSplitter.
          5) If no headers found, fallback to RecursiveCharacterTextSplitter.
        """
        if not reader_output.text:
            raise ValueError("reader_output.text is empty or None")

        filetype = self._guess_filetype(reader_output)
        tuples = self._make_tuples("md")  # Always work in Markdown space.

        text = reader_output.text

        # HTML → Markdown using the project's converter
        if filetype == "html":
            text = HtmlToMarkdown().convert(text)
        else:
            # Normalize Setext headings if already Markdown
            text = self._normalize_setext(text)

        # Detect presence of ATX headers (after conversion/normalization)
        has_headers = bool(re.search(r"(?m)^\s*#{1,6}\s+\S", text))

        # Configure header splitter. group_header_with_content -> strip_headers False
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=tuples,
            return_each_line=False,
            strip_headers=not self.group_header_with_content,
        )

        docs = splitter.split_text(text) if has_headers else []
        # Fallback if no headers were found
        if not docs:
            rc = RecursiveCharacterTextSplitter(
                chunk_size=max(1, int(self.chunk_size) or 1000),
                chunk_overlap=min(200, max(0, int(self.chunk_size) // 10)),
            )
            docs = rc.create_documents([text])

        chunks = [doc.page_content for doc in docs]

        return SplitterOutput(
            chunks=chunks,
            chunk_id=self._generate_chunk_ids(len(chunks)),
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="header_splitter",
            split_params={
                "headers_to_split_on": self.headers_to_split_on,
                "group_header_with_content": self.group_header_with_content,
            },
            metadata=self._default_metadata(),
        )
