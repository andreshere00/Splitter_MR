import json
import warnings

from ...schema.exceptions import InvalidChunkException, SplitterOutputException
from ...schema.models import ReaderOutput, SplitterOutput
from ...schema.warnings import SplitterInputWarning
from ..base_splitter import BaseSplitter


class CharacterSplitter(BaseSplitter):
    """
    Splits textual input into fixed-size character chunks with optional overlap.

    The ``CharacterSplitter`` is a simple and robust splitter that divides text into
    overlapping or non-overlapping chunks, based on the specified number of characters
    per chunk. It is commonly used in document-processing or NLP pipelines where
    preserving context between chunks is important.

    The splitter can be configured to use:
      - ``chunk_size``: maximum number of characters per chunk.
      - ``chunk_overlap``: the number (or fraction) of overlapping characters
        between consecutive chunks.

    Args:
        chunk_size (int, optional): Maximum number of characters per chunk. Must be >= 1.
        chunk_overlap (Union[int, float], optional): Number or percentage of overlapping
            characters between chunks. If float, must be in [0.0, 1.0).

    Raises:
        ValueError: If either ``chunk_size`` or ``chunk_overlap`` are invalid.

    Example:
        ```python
        from splitter_mr.splitter import CharacterSplitter
        from splitter_mr.schema.models import ReaderOutput

        reader_output = ReaderOutput(
            text="abcdefghijklmnopqrstuvwxyz",
            document_name="doc.txt",
            document_path="/path/doc.txt",
        )

        splitter = CharacterSplitter(chunk_size=5, chunk_overlap=2)
        output = splitter.split(reader_output)
        print(output.chunks)
        # ['abcde', 'defgh', 'ghijk', ..., 'yz']
        ```
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be an integer >= 1")

        if isinstance(chunk_overlap, int):
            if chunk_overlap < 0:
                raise ValueError("chunk_overlap (int) must be >= 0")
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap (int) must be smaller than chunk_size")
        elif isinstance(chunk_overlap, float):
            if not (0.0 <= chunk_overlap < 1.0):
                raise ValueError("chunk_overlap (float) must be in [0.0, 1.0)")
        else:
            raise ValueError("chunk_overlap must be int or float")

        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap

    def _coerce_overlap(self, chunk_size: int) -> int:
        """
        Convert the ``chunk_overlap`` parameter into an absolute number of characters.

        Args:
            chunk_size (int): The configured chunk size.

        Returns:
            int: The computed overlap value (in characters).
        """
        if isinstance(self.chunk_overlap, float):
            return int(chunk_size * self.chunk_overlap)
        return int(self.chunk_overlap)

    def _check_input(self, reader_output: ReaderOutput, text: str) -> None:
        """
        Validate and warn about potential input issues.

        This helper method emits warnings instead of raising exceptions for the following cases:
          - Empty or whitespace-only text.
          - Declared JSON input (``conversion_method='json'``) that cannot be parsed as JSON.

        Args:
            reader_output (ReaderOutput): Input reader output containing text and metadata.
            text (str): The textual content to check.

        Warnings:
            SplitterInputWarning: Emitted if text is empty or non-parseable JSON.
        """
        if text.strip() == "":
            warnings.warn(
                SplitterInputWarning(
                    "ReaderOutput.text is empty or whitespace-only. "
                    "Proceeding; this will yield a single empty chunk."
                )
            )

        if (reader_output.conversion_method or "").lower() == "json":
            try:
                json.loads(text or "")
            except Exception:
                warnings.warn(
                    SplitterInputWarning(
                        "ReaderOutput.conversion_method is 'json' but text is not valid JSON. "
                        "Proceeding as plain text."
                    )
                )

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Split the provided text into character-based chunks with optional overlap.

        The method iterates through the text and produces fixed-size chunks that can
        optionally overlap. Each chunk is accompanied by automatically generated
        unique identifiers and metadata inherited from the original document.

        Input validity is checked and warnings may be emitted for empty or invalid text.

        Args:
            reader_output (ReaderOutput): A validated input object containing at least
                a ``text`` field and optional document metadata.

        Returns:
            SplitterOutput: Structured splitter output including:
                - ``chunks``: list of text segments.
                - ``chunk_id``: unique identifier per chunk.
                - document metadata.
                - ``split_params`` reflecting the splitter configuration.

        Raises:
            ValueError: If initialization parameters are invalid.
            InvalidChunkException: If chunks cannot be properly created (e.g., all empty).
            SplitterOutputException: If the final SplitterOutput cannot be validated or built.

        Warnings:
            SplitterInputWarning: If text is empty or cannot be parsed as JSON.

        Example:
            ```python
            reader_output = ReaderOutput(
                text="Hello world! This is a test text for splitting.",
                document_name="example.txt",
                document_path="/path/example.txt"
            )
            splitter = CharacterSplitter(chunk_size=10, chunk_overlap=0.2)
            output = splitter.split(reader_output)
            print(output.chunks)
            # ['Hello worl', 'world! Thi', 'is is a te', ...]
            ```
        """
        text = reader_output.text
        chunk_size = self.chunk_size

        self._check_input(reader_output, text)

        overlap = self._coerce_overlap(chunk_size)

        chunks = []
        start = 0
        step = max(1, chunk_size - overlap)

        try:
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += step

            if len(text) == 0:
                chunks = [""]

            if not isinstance(chunks, list) or len(chunks) == 0:
                raise InvalidChunkException("No chunks were produced.")
            if any(c is None for c in chunks):
                raise InvalidChunkException("A produced chunk is None.")
            if len(text) > 0 and all(c == "" for c in chunks):
                raise InvalidChunkException(
                    "All produced chunks are empty for non-empty text."
                )
        except InvalidChunkException:
            raise
        except Exception as e:
            raise InvalidChunkException(
                f"Unexpected error while building chunks: {e}"
            ) from e

        try:
            chunk_ids = self._generate_chunk_ids(len(chunks))
            metadata = self._default_metadata()
            output = SplitterOutput(
                chunks=chunks,
                chunk_id=chunk_ids,
                document_name=reader_output.document_name,
                document_path=reader_output.document_path or "",
                document_id=reader_output.document_id,
                conversion_method=reader_output.conversion_method,
                reader_method=reader_output.reader_method,
                ocr_method=reader_output.ocr_method,
                split_method="character_splitter",
                split_params={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
                metadata=metadata,
            )
            return output
        except Exception as e:
            raise SplitterOutputException(f"Failed to build SplitterOutput: {e}") from e
