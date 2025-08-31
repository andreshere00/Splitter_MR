from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseVisionModel(ABC):
    """
    Abstract base for vision models that extract text from images.

    Subclasses encapsulate local or API-backed implementations (e.g., OpenAI,
    Azure OpenAI, or on-device models). Implementations should handle encoding,
    request construction, and response parsing while exposing a uniform
    interface for clients of the library.
    """

    @abstractmethod
    def __init__(self, model_name) -> Any:
        """Initialize the model.

        Args:
            model_name (Any): Identifier of the underlying model. For hosted APIs
                this could be a model name or deployment name; for local models,
                it could be a path or configuration object.

        Raises:
            ValueError: If required configuration or credentials are missing.
        """

    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client or handle.

        Returns:
            Any: A client/handle that the implementation uses to perform
                inference (e.g., an SDK client instance, session object, or
                lightweight wrapper). May be ``None`` for pure-local implementations.
        """

    @abstractmethod
    def analyze_content(
        self,
        prompt: str,
        file: Optional[bytes],
        file_ext: Optional[str],
        **parameters: Dict[str, Any],
    ) -> str:
        """Extract text from an image using the provided prompt.

        Encodes the image (provided as base64 **without** the
        ``data:<mime>;base64,`` prefix), sends it with an instruction prompt to
        the underlying vision model, and returns the model's textual output.

        Args:
            prompt (str): Instruction or task description guiding the extraction
                (e.g., *"Read all visible text"* or *"Summarize the receipt"*).
            file (Optional[bytes]): Base64-encoded image bytes **without** the
                header/prefix. Must not be ``None`` for remote/API calls that
                require an image payload.
            file_ext (Optional[str]): File extension (e.g., ``"png"``, ``"jpg"``)
                used to infer the MIME type when required by the backend.
            **parameters (Dict[str, Any]): Additional backend-specific options
                forwarded to the implementation (e.g., timeouts, user tags,
                temperature, etc.).

        Returns:
            str: The extracted text or the model's textual response.

        Raises:
            ValueError: If ``file`` is ``None`` when required, or if the file
                type is unsupported by the implementation.
            RuntimeError: If the inference call fails or returns an unexpected
                response shape.
        """
