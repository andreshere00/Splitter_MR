import mimetypes
import os
from typing import Any, Optional

from openai import Client

from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    GROK_MIME_BY_EXTENSION,
    SUPPORTED_GROK_MIME_TYPES,
    OpenAIClientImageContent,
    OpenAIClientImageUrl,
    OpenAIClientPayload,
    OpenAIClientTextContent,
)
from ..base_model import BaseVisionModel


class GrokVisionModel(BaseVisionModel):
    """
    Implementation of BaseModel for Grok Vision using the xAI API.

    Provides methods to interact with Grok’s multimodal models that support
    base64-encoded images and natural language instructions. This class is
    designed to extract structured text descriptions or captions from images.
    """

    def __init__(
        self,
        api_key: Optional[str] = os.getenv("XAI_API_KEY"),
        model_name: str = os.getenv("XAI_MODEL", "grok-4"),
    ) -> None:
        """
        Initializes the GrokVisionModel.

        Args:
            api_key (str, optional): Grok API key. If not provided, uses the
                ``XAI_API_KEY`` environment variable.
            model_name (str, optional): Model identifier to use. If not provided,
                defaults to ``XAI_MODEL`` environment variable or ``"grok-4"``.

        Raises:
            ValueError: If ``api_key`` is not provided or cannot be resolved
                from environment variables.
        """
        api_key = api_key or os.getenv("XAI_API_KEY")
        model_name = model_name or os.getenv("XAI_MODEL") or "grok-4"

        if not api_key:
            raise ValueError(
                "Grok API key not provided or 'XAI_API_KEY' env var is not set."
            )

        self.model_name = model_name
        self.client = Client(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )  # TODO: Change to xAI SDK

    def get_client(self) -> Client:
        """
        Returns the underlying Grok API client.

        Returns:
            Client: The initialized Grok ``Client`` instance.
        """
        return self.client

    def analyze_content(
        self,
        file: Optional[bytes],
        prompt: Optional[str] = None,
        *,
        file_ext: Optional[str] = "png",
        detail: str = "auto",
        **parameters: Any,
    ) -> str:
        """
        Extract text from an image using the Grok Vision model.

        Encodes the given image as a data URI with an appropriate MIME type based on
        ``file_ext`` and sends it along with a prompt to the Grok API. The API
        processes the image and returns extracted text in the response.

        Args:
            file (bytes, optional): Base64-encoded image content **without** the
                ``data:image/...;base64,`` prefix. Must not be None.
            prompt (str, optional): Instruction text guiding the extraction.
                Defaults to ``DEFAULT_IMAGE_CAPTION_PROMPT``.
            file_ext (str, optional): File extension (e.g., ``"png"``, ``"jpg"``)
                used to determine the MIME type for the image. Defaults to ``"png"``.
            detail (str, optional): Level of detail to request for the image
                analysis. Options typically include ``"low"``, ``"high"`` or ``"auto"``.
                Defaults to ``"auto"``.
            **parameters (Any): Additional keyword arguments passed directly to
                the Grok client ``chat.completions.create()`` method.

        Returns:
            str: The extracted text returned by the vision model.

        Raises:
            ValueError: If ``file`` is None or the file extension is not compatible.
            openai.OpenAIError: If the API request fails.

        Example:
            ```python
            from splitter_mr.model import GrokVisionModel

            model = GrokVisionModel()
            with open("image.jpg", "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            text = model.analyze_content(
                img_b64, prompt="What's in this image?", file_ext="jpg", detail="high"
            )
            print(text)
            ```
        """
        if file is None:
            raise ValueError("No file content provided for text extraction.")

        ext = (file_ext or "png").lower()
        mime_type = (
            GROK_MIME_BY_EXTENSION.get(ext)  # noqa: W503
            or mimetypes.types_map.get(f".{ext}")  # noqa: W503
            or "image/png"  # noqa: W503
        )

        if mime_type not in SUPPORTED_GROK_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {mime_type}")

        prompt = prompt or DEFAULT_IMAGE_CAPTION_PROMPT

        payload_obj = OpenAIClientPayload(
            role="user",
            content=[
                OpenAIClientTextContent(type="text", text=prompt),
                OpenAIClientImageContent(
                    type="image_url",
                    image_url=OpenAIClientImageUrl(
                        url=f"data:{mime_type};base64,{file}",
                        detail=detail,
                    ),
                ),
            ],
        )

        response = self.client.chat.completions.create(
            model=self.model_name, messages=[payload_obj], **parameters
        )

        return response.choices[0].message.content
