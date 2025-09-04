import mimetypes
import os
from typing import Any, Optional

from openai import OpenAI

from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    OPENAI_MIME_BY_EXTENSION,
    SUPPORTED_OPENAI_MIME_TYPES,
    OpenAIClientImageContent,
    OpenAIClientImageUrl,
    OpenAIClientPayload,
    OpenAIClientTextContent,
)
from ..base_model import BaseVisionModel


class OpenAIVisionModel(BaseVisionModel):
    """
    Implementation of BaseModel leveraging OpenAI's Chat Completions API.

    Uses the `client.chat.completions.create()` method to send base64-encoded
    images along with text prompts in a single multimodal request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o"),
    ) -> None:
        """
        Initialize the OpenAIVisionModel.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses the
                ``OPENAI_API_KEY`` environment variable.
            model_name (str): Vision-capable model name (e.g., ``"gpt-4o"``).

        Raises:
            ValueError: If no API key is provided or ``OPENAI_API_KEY`` is not set.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided or 'OPENAI_API_KEY' env var is not set."
                )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        """
        Get the underlying OpenAI client instance.

        Returns:
            OpenAI: The initialized API client.
        """
        return self.client

    def analyze_content(
        self,
        file: Optional[bytes],
        prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT,
        *,
        file_ext: Optional[str] = "png",
        **parameters: Any,
    ) -> str:
        """
        Extract text from an image using OpenAI's Chat Completions API.

        Encodes the provided image bytes as a base64 data URI and sends it
        along with a textual prompt to the specified vision-capable model.
        The model processes the image and returns extracted text.

        Args:
            file (bytes, optional): Base64-encoded image content **without** the
                ``data:image/...;base64,`` prefix. Must not be None.
            prompt (str, optional): Instruction text guiding the extraction.
                Defaults to ``DEFAULT_IMAGE_CAPTION_PROMPT``.
            file_ext (str, optional): File extension (e.g., ``"png"``, ``"jpg"``,
                ``"jpeg"``, ``"webp"``, ``"gif"``) used to determine the MIME type.
                Defaults to ``"png"``.
            **parameters (Any): Additional keyword arguments passed directly to
                the OpenAI client ``chat.completions.create()`` method. Consult documentation
                [here](https://platform.openai.com/docs/api-reference/chat/create).

        Returns:
            str: Extracted text returned by the model.

        Raises:
            ValueError: If ``file`` is None or the file extension is not compatible.
            openai.OpenAIError: If the API request fails.

        Example:
            ```python
            from splitter_mr.model import OpenAIVisionModel
            import base64

            model = OpenAIVisionModel(api_key="sk-...")
            with open("example.png", "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            text = model.analyze_content(img_b64, prompt="Describe the content of this image.")
            print(text)
            ```
        """
        if file is None:
            raise ValueError("No file content provided for text extraction.")

        ext = (file_ext or "png").lower()
        mime_type = (
            OPENAI_MIME_BY_EXTENSION.get(ext)  # noqa: W503
            or mimetypes.types_map.get(f".{ext}")  # noqa: W503
            or "image/png"  # noqa: W503
        )

        if mime_type not in SUPPORTED_OPENAI_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {mime_type}")

        payload_obj = OpenAIClientPayload(
            role="user",
            content=[
                OpenAIClientTextContent(type="text", text=prompt),
                OpenAIClientImageContent(
                    type="image_url",
                    image_url=OpenAIClientImageUrl(
                        url=f"data:{mime_type};base64,{file}"
                    ),
                ),
            ],
        )
        payload = payload_obj.model_dump(exclude_none=True)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[payload],
            **parameters,
        )
        return response.choices[0].message.content
