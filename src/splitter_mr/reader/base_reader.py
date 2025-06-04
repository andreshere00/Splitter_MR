from abc import ABC, abstractmethod


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

    @abstractmethod
    def read(self, file_path: str, **kwargs) -> dict:
        """
        Reads a file and returns a dictionary with its text content and standardized metadata.

        Implementations should extract the main text from the file and populate all metadata
        fields to enable downstream processing and traceability.

        Args:
            file_path (str): Path to the input file to be read.
            **kwargs: Additional keyword arguments for implementation-specific options, such as:
                - document_id (Optional[str]): Unique identifier for the document.
                - conversion_method (Optional[str]): Method used for document conversion.
                - ocr_method (Optional[str]): OCR method used, if any.
                - metadata (Optional[dict]): Additional metadata as a list of strings.

        Returns:
            dict: Dictionary with the following keys:
                - text (str): The extracted text content of the file.
                - document_name (Optional[str]): The base name of the file.
                - document_path (str): The absolute path to the file.
                - document_id (Optional[str]): Unique identifier for the document.
                - conversion_method (Optional[str]): The method used for conversion.
                - ocr_method (Optional[str]): The OCR method applied (if any).
                - metadata (Optional[dict]): Additional document-level metadata.

        Example:
            ```python
            class MyReader(BaseReader):
                def read(self, file_path: str, **kwargs) -> dict:
                    return {
                        "text": "example",
                        "document_name": "example.txt",
                        "document_path": file_path,
                        "document_id": kwargs.get("document_id"),
                        "conversion_method": "custom",
                        "ocr_method": None,
                        "metadata": {}
                    }
            ```
        """
        pass
