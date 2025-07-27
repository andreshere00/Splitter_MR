from typing import List

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class PagedSplitter(BaseSplitter):
    """
    Splits a multi-page document into page-based or multi-page chunks using a placeholder marker.

    Supports overlap in characters between consecutive chunks.

    Args:
        chunk_size (int): Number of pages per chunk.
        chunk_overlap (int): Number of overlapping characters to include from the end of the previous chunk.

    Raises:
        ValueError: If chunk_size is less than 1.
    """

    def __init__(self, chunk_size: int = 1, chunk_overlap: int = 0):
        """
        Args:
            chunk_size (int): Number of pages per chunk.
            chunk_overlap (int): Number of overlapping characters to include from the end of the previous chunk.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be ≥ 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be ≥ 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text into chunks using the page_placeholder in the ReaderOutput.
        Optionally adds character overlap between chunks.

        Args:
            reader_output (ReaderOutput): The output from a reader containing text and metadata.

        Returns:
            SplitterOutput: The result with chunks and related metadata.

        Raises:
            ValueError: If the reader_output does not contain a valid page_placeholder.

        Example:
            ```python
            from splitter_mr.splitter import PagedSplitter

            reader_output = ReaderOutput(
                text: "<!-- page --> Page 1 <!-- page --> This is the page 2.",
                document_name: "test.md",
                document_path: "tmp/test.md",
                page_placeholder: "<!-- page -->",
                ...
            )
            splitter = PagedSplitter(chunk_size = 1)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            [" Page 1 ", " This is the page 2."]
            ```
        """
        page_placeholder: str = reader_output.page_placeholder

        if not bool(page_placeholder):
            raise ValueError(
                "The specified file does not contain page placeholders. "
                "Please, use a compatible file extension (pdf, docx, xlsx, pptx) "
                "or read the file using any BaseReader by pages and try again"
            )

        # Split the document into pages using the placeholder.
        pages: List[str] = [
            page.strip()  # Normalize spacing
            for page in reader_output.text.split(page_placeholder)
            if page.strip()
        ]

        chunks: List[str] = []
        for i in range(0, len(pages), self.chunk_size):
            chunk = "\n".join(pages[i : i + self.chunk_size])
            if self.chunk_overlap > 0 and i > 0 and chunks:
                # Add character overlap from previous chunk
                overlap_text = chunks[-1][-self.chunk_overlap :]
                chunk = overlap_text + chunk
            chunks.append(chunk)

        # Generate chunk_id and append metadata
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
            split_method="paged_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            metadata=metadata,
        )
        return output
