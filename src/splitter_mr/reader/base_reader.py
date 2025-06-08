import json
import os
from abc import ABC, abstractmethod
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
    def parse_json(obj):
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
    def read(self, file_path: str, **kwargs) -> ReaderOutput:
        """
        Reads input and returns a dictionary with its text content and standardized metadata.

        This method should support reading from:
            - File paths (if the string is a valid path)
            - URLs (if the string is a valid HTTP/HTTPS URL)
            - Raw string content (if the string is not a file or URL)
            - JSON/dict input (if the input is a dictionary or a valid JSON string)

        Implementations should extract the main text from the file or source, and populate
        all metadata fields to enable downstream processing and traceability.

        Args:
            file_path (str | dict): Path to the input file, a URL, a raw string, or a dictionary.
            **kwargs: Additional keyword arguments for implementation-specific options, such as:
                - document_id (Optional[str]): Unique identifier for the document.
                - conversion_method (Optional[str]): Method used for document conversion.
                - ocr_method (Optional[str]): OCR method used, if any.
                - metadata (Optional[dict]): Additional metadata as a dictionary.

        Returns:
            dict: Dictionary with the following keys:
                - text (str): The extracted text content.
                - document_name (Optional[str]): The base name of the file, if available.
                - document_path (Optional[str]): The absolute path to the file or URL, if available.
                - document_id (Optional[str]): Unique identifier for the document, if provided.
                - conversion_method (Optional[str]): The method used for conversion, if provided.
                - ocr_method (Optional[str]): The OCR method applied, if any.
                - metadata (Optional[dict]): Additional document-level metadata, if provided.

        Raises:
            ValueError: If the provided string is not a valid file path, URL, or parsable content.
            TypeError: If input type is unsupported.

        Example:
            ```python
            class MyReader(BaseReader):
                def read(self, file_path: str, **kwargs) -> dict:
                    return {
                        "text": "example",
                        "document_name": "example.txt",
                        "document_path": file_path,
                        "document_id": kwargs.document_id,
                        "conversion_method": "custom",
                        "ocr_method": None,
                        "metadata": {}
                    }
            ```
        """
        pass
