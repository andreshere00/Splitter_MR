from typing import Any, Dict, Union

from ...schema.schemas import SplitterOutput
from ..base_splitter import BaseSplitter


class WordSplitter(BaseSplitter):
    """
    WordSplitter splits a given text into overlapping or non-overlapping chunks
    based on a specified number of words per chunk.

    This splitter is configurable with a maximum chunk size (`chunk_size`, in words)
    and an overlap between consecutive chunks (`chunk_overlap`). The overlap can be
    specified either as an integer (number of words) or as a float between 0 and 1
    (fraction of chunk size). Useful for NLP tasks where word-based boundaries are
    important for context preservation.

    Args:
        chunk_size (int): Maximum number of words per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping words between chunks.
    """

    def __init__(self, chunk_size: int = 5, chunk_overlap: Union[int, float] = 0):
        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap

    def split(self, reader_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Splits the input text from the reader_output dictionary into word-based chunks.

        Each chunk contains at most `chunk_size` words, and adjacent chunks can overlap
        by a specified number or percentage of words, according to the `chunk_overlap`
        parameter set at initialization.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary containing at least a 'text' key (str) and optional document metadata
                (e.g., 'document_name', 'document_path', etc.).

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - 'chunks': List[str], the resulting text chunks.
                - 'chunk_id': List[str], unique IDs for each chunk.
                - 'document_name': Optional[str], source document name.
                - 'document_path': str, source document path.
                - 'document_id': Optional[str], unique document identifier.
                - 'conversion_method': Optional[str], conversion method used.
                - 'ocr_method': Optional[str], OCR method used (if any).
                - 'split_method': str, the name of the split method ("word_splitter").
                - 'split_params': Dict[str, Any], parameters used for splitting.
                - 'metadata': Dict[str, Any], document-level metadata.

        Raises:
            ValueError: If chunk_overlap is greater than or equal to chunk_size.

        Example:
            ```python
            from splitter_mr.splitter import WordSplitter

            reader_output = {
                "text": (
                    "The quick brown fox jumps over the lazy dog. "
                    "Pack my box with five dozen liquor jugs. "
                    "Sphinx of black quartz, judge my vow."
                ),
                "document_name": "pangrams.txt",
                "document_path": "/data/pangrams.txt",
            }

            # Split into chunks of 5 words, overlapping by 2 words
            splitter = WordSplitter(chunk_size=5, chunk_overlap=2)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```bash
            ["The quick brown fox jumps", "fox jumps over the lazy", "over the lazy dog. Pack", ...]
            ```
        """
        # Initialize variables
        text = reader_output.get("text", "")
        chunk_size = self.chunk_size

        # Split text into words (using simple whitespace tokenization)
        words = text.split()
        total_words = len(words)

        # Determine overlap in characters
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            overlap = int(chunk_size * self.chunk_overlap)
        else:
            overlap = int(self.chunk_overlap)
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        # Split into chunks
        chunks = []
        start = 0
        step = chunk_size - overlap if (chunk_size - overlap) > 0 else 1
        while start < total_words:
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start += step

        # Generate chunk_id and append metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        # Return output
        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.get("document_name"),
            document_path=reader_output.get("document_path", ""),
            document_id=reader_output.get("document_id"),
            conversion_method=reader_output.get("conversion_method"),
            ocr_method=reader_output.get("ocr_method"),
            split_method="word_splitter",
            split_params={
                "chunk_size": chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            metadata=metadata,
        )
        return output.__dict__
