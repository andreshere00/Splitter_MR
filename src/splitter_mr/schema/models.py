import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ReaderOutput(BaseModel):
    """Pydantic model defining the output structure for all readers.

    Attributes:
        text: The textual content extracted by the reader.
        document_name: The name of the document.
        document_path: The path to the document.
        document_id: A unique identifier for the document.
        conversion_method: The method used for document conversion.
        reader_method: The method used for reading the document.
        ocr_method: The OCR method used, if any.
        metadata: Additional metadata associated with the document.
    """

    text: Optional[str] = ""
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("document_id", mode="before")
    def default_document_id(cls, v: str):
        """Generate a default UUID for document_id if not provided.

        Args:
            v (str): The provided document_id value.

        Returns:
            document_id (str): The provided document_id or a newly generated UUID string.
        """
        document_id = v or str(uuid.uuid4())
        return document_id

    @classmethod
    def from_text(cls, s: str) -> "ReaderOutput":
        """Create a ReaderOutput from a text string, with all other fields set to their defaults.

        Args:
            s (str): The text content to store in the ReaderOutput.

        Returns:
            ReaderOutput: An instance of ReaderOutput with the given text.
        """
        return cls(text=s)


class SplitterOutput(BaseModel):
    """Pydantic model defining the output structure for all splitters.

    Attributes:
        chunks: List of text chunks produced by splitting.
        chunk_id: List of unique IDs corresponding to each chunk.
        document_name: The name of the document.
        document_path: The path to the document.
        document_id: A unique identifier for the document.
        conversion_method: The method used for document conversion.
        reader_method: The method used for reading the document.
        ocr_method: The OCR method used, if any.
        split_method: The method used to split the document.
        split_params: Parameters used during the splitting process.
        metadata: Additional metadata associated with the splitting.
    """

    chunks: List[str] = Field(default_factory=list)
    chunk_id: Optional[List[str]] = None
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    split_method: str = ""
    split_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_and_set_defaults(self):
        """Validates and sets defaults for the SplitterOutput instance.

        Raises:
            ValueError: If `chunks` is empty or if `chunk_id` length does not match `chunks` length.

        Returns:
            self (SplitterOutput): The validated and updated instance.
        """
        if not self.chunks:
            raise ValueError("Chunks list cannot be empty.")

        if self.chunk_id is not None:
            if len(self.chunk_id) != len(self.chunks):
                raise ValueError(
                    f"chunk_id length ({len(self.chunk_id)}) does not match chunks length ({len(self.chunks)})."
                )
        else:
            self.chunk_id = [str(uuid.uuid4()) for _ in self.chunks]

        if not self.document_id:
            self.document_id = str(uuid.uuid4())

        return self

    @classmethod
    def from_chunks(cls, chunks: List[str]) -> "SplitterOutput":
        """Create a SplitterOutput from a list of chunks, with all other fields set to their defaults.

        Args:
            chunks (List[str]): A list of text chunks.

        Returns:
            SplitterOutput: An instance of SplitterOutput with the given chunks.
        """
        return cls(chunks=chunks)
