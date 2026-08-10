import mimetypes
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DEFAULT_IMAGE_EXTENSION,
    DEFAULT_OPENROUTER_ENTRYPOINT,
    DEFAULT_OPENROUTER_MODEL,
    OPENAI_MIME_BY_EXTENSION,
    SUPPORTED_OPENAI_MIME_TYPES,
    OpenAIClientImageContent,
    OpenAIClientImageUrl,
    OpenAIClientPayload,
    OpenAIClientTextContent,
)
from ..base_model import BaseVisionModel


class OpenRouterVisionModel(BaseVisionModel):
    """
    Implementation of BaseVisionModel using OpenRouter's universal API via OpenAI SDK.

    Sends base64-encoded images and prompts to any vision-capable model slug
    exposed through OpenRouter's OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        site_url: Optional[str] = os.getenv("OPENROUTER_SITE_URL"),
        app_name: Optional[str] = os.getenv("OPENROUTER_APP_NAME"),
    ) -> None:
        """
        Initialize the OpenRouterVisionModel.

        Args:
            api_key (str, optional): OpenRouter API key. Uses OPENROUTER_API_KEY env
                var if not provided.
            model_name (str): OpenRouter model slug (e.g. ``openai/gpt-4o``).
            site_url (str, optional): Optional site URL for OpenRouter app attribution
                (``HTTP-Referer`` header). Uses OPENROUTER_SITE_URL if not provided.
            app_name (str, optional): Optional app name for OpenRouter attribution
                (``X-OpenRouter-Title`` header). Uses OPENROUTER_APP_NAME if not
                provided.

        Raises:
            ValueError: If no API key provided or found in environment.
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not provided and 'OPENROUTER_API_KEY' "
                    "env var not set."
                )

        default_headers: Dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-OpenRouter-Title"] = app_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=DEFAULT_OPENROUTER_ENTRYPOINT,
            default_headers=default_headers or None,
        )
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        """
        Get the underlying OpenRouter API client instance.

        Returns:
            OpenAI: The initialized API client.
        """
        return self.client

    def analyze_content(
        self,
        file: Optional[bytes],
        prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT,
        *,
        file_ext: Optional[str] = DEFAULT_IMAGE_EXTENSION,
        **parameters: Dict[str, Any],
    ) -> str:
        """
        Extract text from an image using a vision model via OpenRouter.

        Args:
            file (bytes): Base64-encoded image content, no prefix/header.
            prompt (str): Task or instruction (e.g. "Describe the image contents").
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

        ext = (file_ext or DEFAULT_IMAGE_EXTENSION).lower()
        mime_type = (
            OPENAI_MIME_BY_EXTENSION.get(ext)
            or mimetypes.types_map.get(f".{ext}")  # noqa: W503
            or "image/png"  # noqa: W503
        )
        if mime_type not in SUPPORTED_OPENAI_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type for OpenRouter: {mime_type}")

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
