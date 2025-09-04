import mimetypes
import os
from typing import Any, Dict, Optional

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


class AnthropicVisionModel(BaseVisionModel):
    """
    Implementation of BaseVisionModel using Anthropic's Claude Vision API via OpenAI SDK.

    Sends base64-encoded images + prompts to the Claude multimodal endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    ) -> None:
        """
        Initialize the AnthropicVisionModel.

        Args:
            api_key (str, optional): Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
            model_name (str): Vision-capable Claude model name.

        Raises:
            ValueError: If no API key provided or found in environment.
        """
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not provided and 'ANTHROPIC_API_KEY' env var not set."
                )

        base_url: str = ("https://api.anthropic.com/v1/",)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        """
        Get the underlying Anthropic API client instance.

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
        **parameters: Dict[str, Any],
    ) -> str:
        """
        Extract text from an image using Anthropic's Claude Vision API.

        Args:
            prompt (str): Task or instruction (e.g. "Describe the image contents").
            file (bytes): Base64-encoded image content, no prefix/header.
            file_ext (str, optional): File extension (e.g. "png", "jpg").
            **parameters: Extra arguments to client.chat.completions.create().

        Returns:
            str: Extracted text or model response.

        Raises:
            ValueError: If file is None or unsupported file type.
            RuntimeError: For failed/invalid responses.
        """
        if file is None:
            raise ValueError("No file content provided for vision model.")

        ext = (file_ext or "png").lower()
        mime_type = (
            OPENAI_MIME_BY_EXTENSION.get(ext)
            or mimetypes.types_map.get(f".{ext}")  # noqa: W503
            or "image/png"  # noqa: W503
        )
        if mime_type not in SUPPORTED_OPENAI_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type for Anthropic: {mime_type}")

        # Build multimodal payload in OpenAI/Anthropic-compatible format
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
        try:
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to extract response: {e}")
