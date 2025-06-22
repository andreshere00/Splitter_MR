import os
from typing import Any, Optional

from openai import OpenAI

from ..base_model import BaseModel


class OpenAIVisionModel(BaseModel):
    """
    Implementation of BaseModel leveraging OpenAI's Responses API.

    Uses the `client.responses.create()` method to send base64-encoded images
    along with text prompts in a single multimodal request.
    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-4.1"):
        """
        Initializes the OpenAIVisionModel.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses environment variable 'OPENAI_API_KEY'.
            model_name (str): Vision-capable model name (e.g., "gpt-4.1").
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
        """Returns the underlying OpenAI client instance."""
        return self.client

    def extract_text(
        self,
        file: Optional[bytes],
        prompt: str = "Extract the text from this resource in the original language. Return the result in markdown code format.",
        **parameters: Any,
    ) -> str:
        """
        Extracts text from a base64-encoded image using OpenAI's Responses API.

        Args:
            file (bytes): Base64-encoded image string.
            prompt (str): Instructions for text extraction.
            **parameters: Additional parameters for `client.responses.create()`.

        Returns:
            str: The extracted text from the image.

        Example:
            ```python
            from splitter_mr.model import OpenAIVisionModel

            # Initialize with your OpenAI API key (set as env variable or pass directly)
            model = OpenAIVisionModel(api_key="sk-...")

            with open("example.png", "rb") as f:
                image_bytes = f.read()

            markdown = model.extract_text(image_bytes)
            print(markdown)
            ```
            ```python
            This picture shows ...
            ```
        """
        payload = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{file}"},
                },
            ],
        }
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[payload], **parameters
        )
        return response.choices[0].message.content
