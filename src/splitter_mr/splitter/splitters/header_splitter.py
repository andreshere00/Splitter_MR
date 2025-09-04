import re
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter
from .utils import HtmlToMarkdown


class HeaderSplitter(BaseSplitter):
    """
    Splits an HTML or Markdown document into chunks based on header levels.
    HTML is always converted to Markdown before splitting for consistency.

    Args:
        chunk_size (int, optional): Kept for compatibility. Defaults to 1000.
        headers_to_split_on (Optional[List[str]]): List of semantic header names such as
            ["Header 1", "Header 2"]. If None, all levels 1–6 are enabled.
        group_header_with_content (bool, optional): If True (default), keeps each header with
            its following block(s). If False, falls back to line/element splitting.

    Example:
        ```python
        from splitter_mr.splitter import HeaderSplitter

        reader_output = ReaderOutput(
            text = '<h1>Main Title</h1><h2>Section 1</h2>...',
            ...
        )
        splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
        output = splitter.split(reader_output)
        print(output.chunks)
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
        Initializes the HeaderSplitter with header configuration.

        Args:
            chunk_size (int): Unused, for API compatibility.
            headers_to_split_on (Optional[List[str]]): List of header names (e.g., ["Header 2"]).
            group_header_with_content (bool): If True, group header with body. Default True.
        """
        super().__init__(chunk_size)
        self.headers_to_split_on = headers_to_split_on or [
            f"Header {i}" for i in range(1, 7)
        ]
        self.group_header_with_content = bool(group_header_with_content)

    def _make_tuples(self, filetype: str) -> List[Tuple[str, str]]:
        """
        Converts semantic header names into tuples for Markdown splitter.

        Args:
            filetype (str): "md" for Markdown (always used here).

        Returns:
            List[Tuple[str, str]]: Tuples with (header_token, semantic_name).

        Raises:
            ValueError: If filetype is unknown.
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
        Extracts the numeric level from a header name like "Header 2".

        Args:
            header (str): Header string, e.g. "Header 2".

        Returns:
            int: Level of the header (e.g., 2 for "Header 2").

        Raises:
            ValueError: If header string is not of expected format.
        """
        m = re.match(r"header\s*(\d+)", header.lower())
        if not m:
            raise ValueError(f"Invalid header: {header}")
        return int(m.group(1))

    @staticmethod
    def _guess_filetype(reader_output: ReaderOutput) -> str:
        """
        Guesses if the document is HTML or Markdown based on filename or content.

        Args:
            reader_output (ReaderOutput): Reader output with text and metadata.

        Returns:
            str: "html" or "md".
        """
        name = (reader_output.document_name or "").lower()
        if name.endswith((".html", ".htm")):
            return "html"
        if name.endswith((".md", ".markdown")):
            return "md"

        soup = BeautifulSoup(reader_output.text, "html.parser")
        if soup.find("html") or soup.find(re.compile(r"^h[1-6]$")) or soup.find("div"):
            return "html"
        return "md"

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the document into chunks using the configured header levels.
        HTML input is always converted to Markdown before splitting.

        Args:
            reader_output (ReaderOutput): Input object with document text and metadata.

        Returns:
            SplitterOutput: Output dataclass with chunked text and metadata.

        Raises:
            ValueError: If reader_output.text is empty.
        """
        if not reader_output.text:
            raise ValueError("reader_output.text is empty or None")

        filetype = self._guess_filetype(reader_output)
        tuples = self._make_tuples("md")  # Always markdown now

        text = reader_output.text
        # Convert HTML to Markdown if needed
        if filetype == "html":
            text = HtmlToMarkdown().convert(text)

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=tuples, return_each_line=False, strip_headers=False
        )
        docs = splitter.split_text(text)
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
