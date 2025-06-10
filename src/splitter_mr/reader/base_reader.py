import json
import os
from abc import ABC, abstractmethod
from typing import Any, Union
from urllib.parse import urlparse

from ..schema import ReaderOutput


class BaseReader(ABC):
    """
    Abstract base class for all document readers.

    This interface defines the contract for file readers that process documents and return
    a standardized dictionary containing the extracted text and document-level metadata.
    Subclasses must implement the `read` method to handle specific file formats or reading
    strategies.

    Methods:
        read(file_path: str, **kwargs) -> dict:
            Reads the input file and returns a dictionary with text and metadata.
    """

    @staticmethod
    def is_valid_file_path(path: str) -> bool:
        """
        Checks if the provided string is a valid file path.

        Args:
            path (str): The string to check.

        Returns:
            bool: True if the string is a valid file path to an existing file, False otherwise.

        Example:
            ```python
            BaseReader.is_valid_file_path("/tmp/myfile.txt")
            ```
            ```bash
            True
            ```
        """
        return os.path.isfile(path)

    @staticmethod
    def is_url(string: str) -> bool:
        """
        Determines whether the given string is a valid HTTP or HTTPS URL.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the string is a valid URL with HTTP or HTTPS scheme, False otherwise.

        Example:
            ```python
            BaseReader.is_url("https://example.com")
            ```
            ```bash
            True
            ```
            ```python
            BaseReader.is_url("not_a_url")
            ```
            ```bash
            False
            ```
        """
        try:
            result = urlparse(string)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    @staticmethod
    def parse_json(obj: Union[dict, str]) -> dict:
        """
        Attempts to parse the provided object as JSON.

        Args:
            obj (Union[dict, str]): The object to parse. If a dict, returns it as-is.
                If a string, attempts to parse it as a JSON string.

        Returns:
            dict: The parsed JSON object.

        Raises:
            ValueError: If a string is provided that cannot be parsed as valid JSON.
            TypeError: If the provided object is neither a dict nor a string.

        Example:
            ```python
            BaseReader.try_parse_json('{"a": 1}')
            ```
            ```bash
            {'a': 1}
            ```
            ```python
            BaseReader.try_parse_json({'b': 2})
            ```
            ```bash
            {'b': 2}
            ```
            ```python
            BaseReader.try_parse_json('[not valid json]')
            ```
            ValueError: String could not be parsed as JSON: ...
            ```
        """
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception as e:
                raise ValueError(f"String could not be parsed as JSON: {e}")
        raise TypeError("Provided object is not a string or dictionary")

    @abstractmethod
    @abstractmethod
    def read(self, file_path: str, **kwargs: Any) -> ReaderOutput:
        """
        Reads input and returns a ReaderOutput with text content and standardized metadata.

        Args:
            file_path (str): Path to the input file, a URL, raw string, or dictionary.
            **kwargs: Additional keyword arguments for implementation-specific options.

        Returns:
            ReaderOutput: An object with these keys:
                - text (str)
                - document_name (Optional[str])
                - document_path (Optional[str])
                - document_id (Optional[str])
                - conversion_method (Optional[str])
                - ocr_method (Optional[str])
                - metadata (Optional[dict])

        Raises:
            ValueError: If the provided string is not valid file path, URL, or parsable content.
            TypeError: If input type is unsupported.

        Example:
            ```python
            class MyReader(BaseReader):
                def read(self, file_path: str, **kwargs) -> ReaderOutput:
                    return ReaderOutput(
                        text="example",
                        document_name="example.txt",
                        document_path=file_path,
                        document_id=kwargs.get("document_id"),
                        conversion_method="custom",
                        ocr_method=None,
                        metadata={}
                    )
            ```
        """
        pass
