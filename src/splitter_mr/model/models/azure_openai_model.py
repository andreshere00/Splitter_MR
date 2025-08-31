import mimetypes
import os
from typing import Any, Optional

from openai import AzureOpenAI

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


class AzureOpenAIVisionModel(BaseVisionModel):
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
    ) -> None:
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

        Raises:
            ValueError: If no connection details are provided or environment variables
                are not set.
        """
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI API key not provided or 'AZURE_OPENAI_API_KEY' env var is not set."
                )
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError(
                    "Azure endpoint not provided or 'AZURE_OPENAI_ENDPOINT' env var is not set."
                )
        if azure_deployment is None:
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            if not azure_deployment:
                raise ValueError(
                    "Azure deployment name not provided or 'AZURE_OPENAI_DEPLOYMENT' env var is not set."
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

    def analyze_content(
        self,
        file: Optional[bytes],
        prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT,
        file_ext: Optional[str] = "png",
        **parameters: Any,
    ) -> str:
        """
        Extract text from an image using the Azure OpenAI Vision model.

        Encodes the given image as a data URI with an appropriate MIME type based on
        ``file_ext`` and sends it along with a prompt to the Azure OpenAI Vision API.
        The API processes the image and returns extracted text in the response.

        Args:
            file (bytes, optional): Base64-encoded image content **without** the
                ``data:image/...;base64,`` prefix. Must not be None.
            prompt (str, optional): Instruction text guiding the extraction.
                Defaults to ``DEFAULT_IMAGE_CAPTION_PROMPT``.
            file_ext (str, optional): File extension (e.g., ``"png"``, ``"jpg"``)
                used to determine the MIME type for the image. Defaults to ``"png"``.
            **parameters (Any): Additional keyword arguments passed directly to
                the Azure OpenAI client ``chat.completions.create()`` method. Consult
                documentation [here](https://platform.openai.com/docs/api-reference/chat/create).

        Returns:
            str: The extracted text returned by the vision model.

        Raises:
            ValueError: If ``file`` is None or the file extension is not compatible.
            openai.OpenAIError: If the API request fails.

        Example:
            ```python
            model = AzureOpenAIVisionModel(...)
            with open("image.jpg", "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            text = model.analyze_content(img_b64, prompt="Describe this image", file_ext="jpg")
            print(text)
            ```
        """
        if file is None:
            raise ValueError("No file content provided to be analyzed with the VLM.")

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
            model=self.get_client()._azure_deployment,
            messages=[payload],
            **parameters,
        )
        return response.choices[0].message.content
