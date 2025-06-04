from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter

from ...schema.schemas import SplitterOutput
from ..base_splitter import BaseSplitter


class HeaderSplitter(BaseSplitter):
    """
    HeaderSplitter splits a Markdown or HTML document into chunks using header levels.

    This splitter uses Langchain's MarkdownHeaderTextSplitter or HTMLHeaderTextSplitter
    depending on the file extension. You can configure which headers are used and their
    semantic names (e.g., ("##", "Header 2")).

    Args:
        headers_to_split_on (List[Tuple[str, str]]):
            List of tuples with Markdown or HTML header tokens and their semantic names.
        chunk_size (int):
            Unused but present for compatibility with the BaseSplitter API.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("####", "Header 3"),
        ],
    ):
        super().__init__(chunk_size)
        self.headers_to_split_on = headers_to_split_on

    def split(self, reader_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Splits the input document (Markdown or HTML) found in reader_output["text"] using
            the configured headers.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary with the structure defined in the ReaderOutput schema.
                Must contain a 'text' field (the Markdown or HTML to split) and may contain
                document-level metadata fields such as 'document_name', 'document_path', etc.

        Returns:
            Dict[str, Any]:
                A dictionary matching the SplitterOutput schema with the following keys:
                    - 'chunks': List[str], the resulting text chunks.
                    - 'chunk_id': List[str], unique IDs for each chunk.
                    - 'document_name': Optional[str], source document name.
                    - 'document_path': str, source document path.
                    - 'document_id': Optional[str], unique document identifier.
                    - 'conversion_method': Optional[str], conversion method used.
                    - 'ocr_method': Optional[str], OCR method used (if any).
                    - 'split_method': str, the name of the split method ("header_splitter").
                    - 'split_params': Dict[str, Any], parameters used for splitting.
                    - 'metadata': Dict, document-level metadata dictionary.

        Raises:
            ValueError: If 'text' is not present in reader_output or is empty.
            TypeError: If 'text' comes from a non-compatible document type.

        Example:
            ```python
            from splitter_mr.splitter import HeaderSplitter

            # This dictionary has been obtained as the output from a Reader object.
            reader_output = {
                "text": "# Title\\n\\n## Subtitle\\nText...",
                "document_name": "doc.md",
                "document_path": "/path/doc.md"
            }
            splitter = HeaderSplitter(
                headers_to_split_on = [
                    ("#", "Header 1"), ("##", "Header 2")
                    ]
                )
            output = splitter.split(reader_output)
            print(output["chunks"][0])
            ```
            ```python
            >>> ["# Title \\n \\n", "## Subtitle\\nText..."]
            ```
        """
        # Initialize variables
        markdown_text = reader_output.get("text", "")

        if bool(BeautifulSoup(markdown_text, "html.parser").find()):
            try:
                splitter = HTMLHeaderTextSplitter(
                    headers_to_split_on=self.headers_to_split_on
                )
            except ValueError as e:
                raise TypeError(
                    "Incorrect Header selection for HTML splitter. Please ensure headers_to_split_on contains only HTML header tags like [('h1', 'Header 1'), ('h2', 'Header 2')], etc."  # noqa: E501
                ) from e
        else:
            try:
                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=self.headers_to_split_on
                )
            except ValueError as e:
                raise ValueError(
                    "Incorrect Header selection for Markdown Splitter. Please ensure headers_to_split_on contains only Markdown header tags like [('#', 'Header 1'), ('##', 'Header 2')], etc."  # noqa: E501
                ) from e

        docs = splitter.split_text(markdown_text)
        chunks = [doc.page_content for doc in docs]
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.get("document_name"),
            document_path=reader_output.get("document_path", ""),
            document_id=reader_output.get("document_id"),
            conversion_method=reader_output.get("conversion_method"),
            ocr_method=reader_output.get("ocr_method"),
            split_method="header_splitter",
            split_params={
                "headers_to_split_on": self.headers_to_split_on,
            },
            metadata=metadata,
        )
        return output.__dict__
