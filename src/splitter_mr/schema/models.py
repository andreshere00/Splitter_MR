import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

try:
    import torch

    TorchDevice = torch.device
except ImportError:
    TorchDevice = object

# ------- #
# READERS #
# ------- #


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
        page_placeholder: The placeholder use to identify each page, if used.
        metadata: Additional metadata associated with the document.
    """

    text: Optional[str] = ""
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    page_placeholder: Optional[str] = None
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

    def from_variable(
        self, variable: Union[str, Dict[str, Any]], variable_name: str
    ) -> "ReaderOutput":
        """
        Generate a new ReaderOutput object from a variable (str or dict).

        Args:
            variable (Union[str, Dict[str, Any]]): The variable to use as text.
            variable_name (str): The name for document_name.

        Returns:
            ReaderOutput: The new ReaderOutput object.
        """
        if isinstance(variable, dict):
            text = json.dumps(variable, ensure_ascii=False, indent=2)
            conversion_method = "json"
            metadata = {"details": "Generated from a json variable"}
        elif isinstance(variable, str):
            text = variable
            conversion_method = "txt"
            metadata = {"details": "Generated from a str variable"}
        else:
            raise ValueError("Variable must be either a string or a dictionary.")

        return ReaderOutput(
            text=text,
            document_name=variable_name,
            document_path="",
            conversion_method=conversion_method,
            reader_method="vanilla",
            ocr_method=None,
            page_placeholder=None,
            metadata=metadata,
        )

    def append_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Append (update) the metadata dictionary with new key-value pairs.

        Args:
            metadata (Dict[str, Any]): The metadata to add or update.
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(metadata)


# --------- #
# SPLITTERS #
# --------- #


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
    chunk_id: List[str] = Field(default_factory=list)
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

    def append_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Append (update) the metadata dictionary with new key-value pairs.

        Args:
            metadata (Dict[str, Any]): The metadata to add or update.
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(metadata)


# ------ #
# MODELS #
# ------ #

# -------------------------- #
# ---- API-Based Models ---- #
# -------------------------- #

# ---- OpenAI models ---- #


class OpenAIClientTextContent(BaseModel):
    """Text content block for chat payloads.

    Attributes:
        type: Constant literal `"text"`.
        text: The textual prompt or instruction.
    """

    type: Literal["text"]
    text: str


class OpenAIClientImageUrl(BaseModel):
    """Image URL container for data-URI images.

    Attributes:
        url: A data URI string (e.g., "data:image/png;base64,<...>").
        detail: Optional level of detail for vision models that support it
            (e.g., Grok). Valid values: "low", "high", "auto".
            If not provided, the field is omitted from the payload.
    """

    url: str
    detail: Optional[Literal["low", "high", "auto"]] = Field(
        default=None, description="Optional detail level for compatible VLMs."
    )


class OpenAIClientImageContent(BaseModel):
    """Image content block for chat payloads.

    Attributes:
        type: Constant literal `"image_url"`.
        image_url: The image URL wrapper containing a data URI.
    """

    type: Literal["image_url"]
    image_url: OpenAIClientImageUrl


class OpenAIClientPayload(BaseModel):
    """Top-level chat message payload sent to the model.

    Attributes:
        role: The role of the message author, one of "user", "system", or "assistant".
        content: Ordered list of content blocks (text and/or image).
    """

    role: Literal["user", "system", "assistant"]
    content: List[OpenAIClientTextContent | OpenAIClientImageContent]


# ---- HuggingFace Models ---- #


class HFChatImageContent(BaseModel):
    """
    TODO: Add docstrings using Google docstyle
    """

    type: Literal["image"]
    image: str


class HFChatTextContent(BaseModel):
    """
    TODO: Add docstrings using Google docstyle
    """

    type: Literal["text"]
    text: str


class HFChatMessage(BaseModel):
    """
    TODO: Add docstrings using Google docstyle
    """

    role: Literal["user", "system", "assistant"]
    content: List[Union[HFChatTextContent, HFChatImageContent]]


class HFClient(BaseModel):
    """
    Lightweight client holder for vision models.
    """

    model: Any
    processor: Any
    tokenizer: Optional[Any] = None
    device: TorchDevice

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("device", mode="before")
    @classmethod
    def _coerce_device(cls, v):
        # Only coerce if torch is available
        try:
            import torch

            if isinstance(v, torch.device):
                return v
            return torch.device(str(v))
        except ImportError:
            return v  # Don't coerce if torch isn't installed

    @field_serializer("device")
    def _serialize_device(self, v) -> str:
        return str(v)
