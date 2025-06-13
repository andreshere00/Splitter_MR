import uuid
from abc import ABC, abstractmethod
from typing import List

from ..schema import ReaderOutput, SplitterOutput


class BaseSplitter(ABC):
    """
    Abstract base class for all splitter implementations.

    This class defines the common interface and utility methods for splitters that
    divide text or data into smaller chunks, typically for downstream natural language
    processing tasks or information retrieval. Subclasses should implement the `split`
    method, which takes in a dictionary (typically from a document reader) and returns
    a structured output with the required chunking.

    Attributes:
        chunk_size (int): The maximum number of units (e.g., characters, words, etc.) per chunk.

    Methods:
        split: Abstract method. Should be implemented by all subclasses to perform the actual
            splitting logic.

        _generate_chunk_ids: Generates a list of unique chunk IDs using UUID4, for use in the output.

        _default_metadata: Returns a default (empty) metadata dictionary, which can be extended by subclasses.
    """

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    @abstractmethod
    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Abstract method to split input data into chunks.

        Args:
            reader_output (ReaderOutput): Input data, typically from a document reader,
                including the text to split and any relevant metadata.

        Returns:
            SplitterOutput: A dictionary containing split chunks and associated metadata.
        """
        pass

    def _generate_chunk_ids(self, num_chunks: int) -> List[str]:
        """
        Generate a list of unique chunk identifiers.

        Args:
            num_chunks (int): Number of chunk IDs to generate.

        Returns:
            List[str]: List of unique string IDs (UUID4).
        """
        return [str(uuid.uuid4()) for _ in range(num_chunks)]

    def _default_metadata(self) -> dict:
        """
        Return a default metadata dictionary.

        Returns:
            dict: An empty dictionary; subclasses may override to provide additional metadata.
        """
        return {}
