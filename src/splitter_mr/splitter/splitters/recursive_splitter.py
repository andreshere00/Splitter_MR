from typing import List, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class RecursiveCharacterSplitter(BaseSplitter):
    """
    RecursiveCharacterSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk is created repeatedly breaking down the text until it reaches the
    desired chunk size. This class implements the Langchain RecursiveCharacterTextSplitter.

    Args:
        chunk_size (int): Approximate chunk size, in characters.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping characters between
            chunks.
        separators (Union[str, List[str]]): Character(s) to recursively split sentences.

    Notes:
        More info about the RecursiveCharacterTextSplitter:
        [Langchain Docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: Union[int, float] = 0.1,
        separators: Union[str, List[str]] = [
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    ):
        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.separators = separators if isinstance(separators, list) else [separators]

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text into character-based chunks using a recursive splitting strategy
        (via Langchain's `RecursiveCharacterTextSplitter`), supporting configurable separators,
        chunk size, and overlap.

        Args:
            reader_output (Dict[str, Any]): Dictionary containing at least a 'text' key (str)
                and optional document metadata (e.g., 'document_name', 'document_path', etc.).

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If 'text' is missing in `reader_output` or is not a string.

        Example:
            ```python
            from splitter_mr.splitter import RecursiveCharacterSplitter

            # This dictionary has been obtained as the output from a Reader object.
            reader_output = ReaderOutput(
                text: "This is a long document.
                It will be recursively split into smaller chunks using the specified separators.
                Each chunk will have some overlap with the next.",
                document_name: "sample.txt",
                document_path: "/tmp/sample.txt"
            )

            splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=5)
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['This is a long document. It will be', 'be recursively split into smaller chunks', ...]
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

        # Split text into sentences
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        texts = splitter.create_documents([text])
        chunks = [doc.page_content for doc in texts]

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
            split_method="recursive_character_splitter",
            split_params={
                "chunk_size": chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "separators": self.separators,
            },
            metadata=metadata,
        )
        return output
