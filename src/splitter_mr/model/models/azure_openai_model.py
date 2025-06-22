import os
from typing import Any, Optional

from openai import AzureOpenAI

from ..base_model import BaseModel


class AzureOpenAIVisionModel(BaseModel):
    """
    Implementation of BaseModel for Azure OpenAI Vision using the Responses API.

    Utilizes Azure’s preview `responses` API, which supports
    base64-encoded images and stateful multimodal calls.
    """

    def __init__(
        self,
        api_key: str = None,
        azure_endpoint: str = None,
        azure_deployment: str = None,
        api_version: str = None,
    ):
        """
        Initializes the AzureOpenAIVisionModel.

        Args:
            api_key (str, optional): Azure OpenAI API key.
                If not provided, uses 'AZURE_OPENAI_API_KEY' env var.
            azure_endpoint (str, optional): Azure endpoint.
                If not provided, uses 'AZURE_OPENAI_ENDPOINT' env var.
            azure_deployment (str, optional): Azure deployment name.
                If not provided, uses 'AZURE_OPENAI_DEPLOYMENT' env var.
            api_version (str, optional): API version string.
                If not provided, uses 'AZURE_OPENAI_API_VERSION' env var or defaults to '2025-04-14-preview'.
        """
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI API key not provided and 'AZURE_OPENAI_API_KEY' env var is not set."
                )
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError(
                    "Azure endpoint not provided and 'AZURE_OPENAI_ENDPOINT' env var is not set."
                )
        if azure_deployment is None:
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not azure_deployment:
                raise ValueError(
                    "Azure deployment name not provided and 'AZURE_OPENAI_DEPLOYMENT' env var is not set."
                )
        if api_version is None:
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
        )
        self.model_name = azure_deployment

    def get_client(self) -> AzureOpenAI:
        """Returns the AzureOpenAI client instance."""
        return self.client

    def extract_text(
        self,
        file: Optional[bytes],
        prompt: str = "Extract the text from this resource in the original language. Return the result in markdown code format.",
        **parameters: Any,
    ) -> str:
        """
        Extracts text from a base64 image using Azure's Responses API.

        Args:
            file (bytes): Base64‑encoded image string.
            prompt (str): Instruction prompt for text extraction.
            **parameters: Extra params passed to client.responses.create().

        Returns:
            str: Extracted text from the image.

        Example:
            ```python
            from splitter_mr.model import AzureOpenAIVisionModel

            # Ensure required Azure environment variables are set, or pass parameters directly
            model = AzureOpenAIVisionModel(
                api_key="...",
                azure_endpoint="https://...azure.com/",
                azure_deployment="deployment-name"
            )

            with open("example.png", "rb") as f:
                image_bytes = f.read()

            markdown = model.extract_text(image_bytes)
            print(markdown)
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
            model=self.get_client()._azure_deployment, messages=[payload], **parameters
        )
        return response.choices[0].message.content
