import mimetypes
import os
from typing import Any, Optional

from openai import OpenAI

from ...schema import DEFAULT_IMAGE_CAPTION_PROMPT
from ..base_model import BaseModel


class OpenAIVisionModel(BaseModel):
    """
    Implementation of BaseModel leveraging OpenAI's Chat Completions API.

    Uses the `client.chat.completions.create()` method to send base64-encoded
    images along with text prompts in a single multimodal request.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4.1"):
        """
        Initialize the OpenAIVisionModel.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses the
                ``OPENAI_API_KEY`` environment variable.
            model_name (str): Vision-capable model name (e.g., ``"gpt-4.1"``).

        Raises:
            ValueError: If no API key is provided and ``OPENAI_API_KEY`` is not set.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided and 'OPENAI_API_KEY' env var is not set."
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

    def extract_text(
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
            **parameters: Extra keyword args forwarded to
                ``client.chat.completions.create()`` (e.g., ``temperature``).

        Returns:
            str: Extracted text returned by the model.

        Raises:
            ValueError: If ``file`` is None.
            openai.OpenAIError: If the API request fails.

        Example:
            ```python
            from splitter_mr.model import OpenAIVisionModel
            import base64

            model = OpenAIVisionModel(api_key="sk-...")
            with open("example.png", "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            text = model.extract_text(img_b64, prompt="Describe the content of this image.")
            print(text)
            ```
        """
        if file is None:
            raise ValueError("No file content provided for text extraction.")

        # Resolve MIME type from extension (fallback is octet-stream).
        mime_type = mimetypes.types_map.get(
            f".{(file_ext or 'png').lower()}", "application/octet-stream"
        )

        payload = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{file}"},
                },
            ],
        }
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[payload], **parameters
        )
        return response.choices[0].message.content
