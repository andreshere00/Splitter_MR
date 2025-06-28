from typing import Any, Tuple
from urllib.parse import urlencode, urljoin

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    PdfPipelineOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from openai import AzureOpenAI, OpenAI


class DoclingUtils:
    """
    Utility class to configure and return Docling pipelines and VLM endpoints.
    """

    @staticmethod
    def get_vlm_url_and_headers(client: Any) -> Tuple[str, dict]:
        """
        Build URL and headers for AzureOpenAI or OpenAI VLM endpoints using urllib.parse.

        Args:
            client: An instance of AzureOpenAI or OpenAI.

        Returns:
            Tuple containing the full endpoint URL and headers dictionary.

        Raises:
            ValueError: If required Azure parameters are missing or client type is unsupported.
        """
        headers = {"Authorization": f"Bearer {client.api_key}"}

        if isinstance(client, AzureOpenAI):
            endpoint = client._azure_endpoint
            deployment = client._azure_deployment
            version = client._api_version
            if not all([endpoint, deployment, version]):
                raise ValueError(
                    "Missing Azure VLM config: endpoint, deployment, or api_version"
                )
            base = str(endpoint).rstrip("/") + "/"
            path = f"openai/deployments/{deployment}/chat/completions"
            url = urljoin(base, path)
            query = urlencode({"api-version": version})
            full_url = f"{url}?{query}"
            return full_url, headers

        if isinstance(client, OpenAI):
            base = "https://api.openai.com/"
            path = "v1/chat/completions"
            full_url = urljoin(base, path)
            return full_url, headers

        raise ValueError(f"Unsupported client type: {type(client)}")

    def get_pdf_pipeline(
        self,
        mode: str,
        client: Any = None,
        model_name: str = "",
        prompt: str = "",
        images_scale: float = 2.0,
        generate_page_images: bool = True,
        generate_picture_images: bool = True,
        timeout: int = 60,
    ) -> DocumentConverter:
        """
        Create a DocumentConverter for PDF processing.

        Args:
            mode: 'vlm' for vision-language page analysis, 'image' for image extraction.
            client: VLM client for 'vlm' mode.
            model_name: Model identifier for the VLM.
            prompt: Caption prompt for VLM.
            images_scale: Scaling factor for extracted images.
            generate_page_images: Include page screenshots.
            generate_picture_images: Include embedded pictures.
            timeout: Request timeout in seconds for VLM.

        Returns:
            Configured DocumentConverter instance.

        Raises:
            ValueError: If mode is not 'vlm' or 'image'.
        """
        if mode == "vlm":
            if client is None:
                raise ValueError("Client required for VLM mode")
            url, headers = self.get_vlm_url_and_headers(client)
            vlm_opts = ApiVlmOptions(
                url=url,
                params={"model": model_name},
                headers=headers,
                prompt=prompt,
                timeout=timeout,
                response_format=ResponseFormat.MARKDOWN,
            )
            vlm_pipeline_opts = VlmPipelineOptions(
                enable_remote_services=True,
                vlm_options=vlm_opts,
            )
            fmt_opts = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_pipeline_opts,
                )
            }
            return DocumentConverter(format_options=fmt_opts)

        if mode == "image":
            pdf_opts = PdfPipelineOptions(
                images_scale=images_scale,
                generate_page_images=generate_page_images,
                generate_picture_images=generate_picture_images,
            )
            fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
            return DocumentConverter(format_options=fmt_opts)

        raise ValueError(f"Unknown PDF pipeline mode: '{mode}'")
