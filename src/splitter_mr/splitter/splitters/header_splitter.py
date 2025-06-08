import re
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class HeaderSplitter(BaseSplitter):
    """
    Splits a Markdown or HTML document into chunks based on header levels.

    This splitter automatically adapts the provided list of semantic header names
    (e.g., ["Header 1", "Header 2"]) into the appropriate header tokens for either
    Markdown (e.g., "#", "##") or HTML (e.g., "h1", "h2"), depending on the document type.
    It then uses Langchain's `MarkdownHeaderTextSplitter` or `HTMLHeaderTextSplitter`
    under the hood.

    Args:
        chunk_size (int, optional): Unused, kept for compatibility with BaseSplitter API.
        headers_to_split_on (Optional[List[str]]):
            List of semantic header names to split on. For example: ["Header 1", "Header 2", "Header 3"]

    Notes:
        - See [Langchain MarkdownHeaderTextSplitter](https://api.python.langchain.com/en/latest/markdown/langchain_text_splitters.markdown.MarkdownHeaderTextSplitter.html)
        - See [Langchain HTMLHeaderTextSplitter](https://python.langchain.com/api_reference/text_splitters/html/langchain_text_splitters.html.HTMLHeaderTextSplitter.html)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        headers_to_split_on: Optional[List[str]] = ["Header 1", "Header 2", "Header 3"],
    ):
        """
        Initializes a HeaderSplitter instance.

        Args:
            chunk_size (int, optional): Unused parameter. Present for API compatibility.
            headers_to_split_on (Optional[List[str]]):
                List of header names such as ["Header 1", "Header 2"]. Defaults to first three levels.
        """
        super().__init__(chunk_size)
        self.headers_to_split_on = headers_to_split_on

    @staticmethod
    def _header_level(header: str) -> int:
        """
        Extracts the numeric level from a header name like "Header 2".

        Args:
            header (str): Semantic header name (e.g., "Header 2").

        Returns:
            int: The numeric level of the header (e.g., 2 for "Header 2").

        Raises:
            ValueError: If the header string is not in the expected format.
        """
        m = re.match(r"header\s*(\d+)", header.lower())
        if not m:
            raise ValueError(f"Invalid header: {header}")
        return int(m.group(1))

    def _make_tuples(self, filetype: str) -> List[Tuple[str, str]]:
        """
        Converts semantic header names into tuples for Markdown or HTML splitters.

        Args:
            filetype (str): "md" for Markdown, "html" for HTML.

        Returns:
            List[Tuple[str, str]]:
                A list of tuples where the first element is the filetype-specific header
                token (e.g., "#", "##", "h1", "h2"), and the second is the header's semantic name.

        Raises:
            ValueError: If filetype is unknown or headers are not in the expected format.
        """
        result = []
        for header in self.headers_to_split_on:
            level = self._header_level(header)
            if filetype == "md":
                token = "#" * level
            elif filetype == "html":
                token = f"h{level}"
            else:
                raise ValueError(f"Incompatible file extension: {filetype}")
            result.append((token, header))
        return result

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits a document (Markdown or HTML) into chunks using the configured header levels.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary with a 'text' field containing the document content,
                plus optional document-level metadata such as 'document_name' and 'document_path'.

        Returns:
            Dict[str, Any]: Dictionary following the SplitterOutput schema with keys:
                - 'chunks': List of chunked document strings.
                - 'chunk_id': List of unique IDs for each chunk.
                - 'document_name': Optional source document name.
                - 'document_path': Source document path.
                - 'document_id': Optional document identifier.
                - 'conversion_method': Optional conversion method.
                - 'ocr_method': Optional OCR method used.
                - 'split_method': Name of the split method ("header_splitter").
                - 'split_params': Parameters used for splitting.
                - 'metadata': Document-level metadata dictionary.

        Raises:
            ValueError: If 'text' is missing from reader_output or is empty.
            TypeError: If the document is not Markdown or HTML.

        Example:
            ```python
            from splitter_mr.splitter import HeaderSplitter

            reader_output = {
                "text": "# Title\\n\\n## Subtitle\\nText...",
                "document_name": "doc.md",
                "document_path": "/path/doc.md"
            }
            splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2"])
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
        """
        text = reader_output.text
        if not text:
            raise ValueError("reader_output must contain non-empty 'text' field.")

        # Detect file type and configure header tuples
        if bool(BeautifulSoup(text, "html.parser").find()):
            # HTML
            tuples = self._make_tuples("html")
            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=tuples, return_each_element=True
            )
        else:
            # Markdown
            tuples = self._make_tuples("md")
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=tuples, return_each_line=True
            )

        docs = splitter.split_text(text)
        chunks = [doc.page_content for doc in docs]
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="header_splitter",
            split_params={
                "headers_to_split_on": self.headers_to_split_on,
            },
            metadata=metadata,
        )
        return output
