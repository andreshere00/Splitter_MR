from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class CharacterSplitter(BaseSplitter):
    """
    CharacterSplitter splits a given text into overlapping or non-overlapping chunks
    based on a specified number of characters per chunk.

    This splitter is configurable with a maximum chunk size (`chunk_size`) and an overlap
    between consecutive chunks (`chunk_overlap`). The overlap can be specified either as
    an integer (number of characters) or as a float between 0 and 1 (fraction of chunk size).
    This is particularly useful for downstream NLP tasks where context preservation between
    chunks is important.

    Args:
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping characters
            between chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text from the reader_output dictionary into character-based chunks.

        Each chunk contains at most `chunk_size` characters, and adjacent chunks can overlap
        by a specified number or percentage of characters, according to the `chunk_overlap`
        parameter set at initialization. Returns a dictionary with the same document metadata,
        unique chunk identifiers, and the split parameters used.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary containing at least a 'text' key (str) and optional document metadata
                (e.g., 'document_name', 'document_path', etc.).

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If chunk_overlap is greater than or equal to chunk_size.

        Example:
            ```python
            from splitter_mr.splitter import CharacterSplitter

            # This dictionary has been obtained as the output from a Reader object.
            reader_output = ReaderOutput(
                text: "abcdefghijklmnopqrstuvwxyz",
                document_name: "doc.txt",
                document_path: "/path/doc.txt",
            )
            splitter = CharacterSplitter(chunk_size=5, chunk_overlap=2)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['abcde', 'defgh', 'ghijk', ..., 'yz']
            ```
        """
        # Initialize variables
        text = reader_output.text
        chunk_size = self.chunk_size

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
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap if (chunk_size - overlap) > 0 else 1

        # Generate chunk_id and append metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        # Return output
        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="character_splitter",
            split_params={
                "chunk_size": chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            metadata=metadata,
        )
        return output
