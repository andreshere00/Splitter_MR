import os
import uuid
from typing import Any, Dict, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from openai import AzureOpenAI, OpenAI

from ...model import BaseModel
from ...schema import ReaderOutput
from ..base_reader import BaseReader
from .vanilla_reader import VanillaReader


class DoclingReader(BaseReader):
    """
    Read multiple file types using IBM's Docling library, and convert the documents
    into markdown or JSON format.
    """

    SUPPORTED_EXTENSIONS = (
        "pdf",
        "docx",
        "html",
        "md",
        "markdown",
        "htm",
        "pptx",
        "xlsx",
        "odt",
        "rtf",
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "gif",
        "tiff",
    )

    def __init__(self, model: Optional[BaseModel] = None):
        self.model = model
        self.model_name = None
        if self.model is not None:
            self.client = self.model.get_client()
            for attr in ["_azure_deployment", "_azure_endpoint", "_api_version"]:
                setattr(self, attr, getattr(self.client, attr, None))
            self.api_key = self.client.api_key
            self.model_name = self.model.model_name

    def _get_vlm_url_and_headers(self, client: Any) -> Tuple[str, Dict[str, str]]:
        """
        Returns VLM API URL and headers based on model type.
        """
        if isinstance(client, AzureOpenAI):
            url = f"{self._azure_endpoint}/openai/deployments/{self._azure_deployment}/chat/completions?api-version={self._api_version}"
            headers = {"Authorization": f"Bearer {client.api_key}"}
        elif isinstance(client, OpenAI):
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {client.api_key}"}
        else:
            raise ValueError(f"Unknown client type: {type(client)}")
        return url, headers

    def _make_docling_reader(self, prompt: str, timeout: int = 60) -> DocumentConverter:
        """
        Returns a configured DocumentConverter with VLM pipeline options for OpenAI or Azure.
        """
        url, headers = self._get_vlm_url_and_headers(self.client)
        vlm_options = ApiVlmOptions(
            url=url,
            params={"model": self.model_name},
            headers=headers,
            prompt=prompt,
            timeout=timeout,
            response_format=ResponseFormat.MARKDOWN,
        )
        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True,
            vlm_options=vlm_options,
        )
        reader = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
        return reader

    def read(
        self,
        file_path: str,
        prompt: str = "Analyze the following resource in the original language. Be concise but comprehensive, according to the image context. Return the content in markdown format",
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Reads and converts a document to Markdown format using the
        [Docling](https://github.com/docling-project/docling) library, supporting a wide range
        of file types including PDF, DOCX, HTML, and images.

        This method leverages Docling's advanced document parsing capabilities—including layout
        and table detection, code and formula extraction, and integrated OCR—to produce clean,
        markdown-formatted output for downstream processing. The output includes standardized
        metadata and can be easily integrated into generative AI or information retrieval pipelines.

        Args:
            file_path (str): Path to the input file to be read and converted.
            **kwargs:
                document_id (Optional[str]): Unique document identifier.
                    If not provided, a UUID will be generated.
                conversion_method (Optional[str]): Name or description of the
                    conversion method used. Default is None.
                ocr_method (Optional[str]): OCR method applied (if any).
                    Default is None.
                metadata (Optional[List[str]]): Additional metadata as a list of strings.
                    Default is an empty list.

        Returns:
            ReaderOutput: Dataclass defining the output structure for all readers.

        Example:
            ```python
            from splitter_mr.readers import DoclingReader

            reader = DoclingReader()
            result = reader.read(file_path = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.pdf")
            print(result.text)
            ```
            ```bash
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """
        # Check if the extension is valid
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(
                f"Warning: File extension not compatible: {ext}. Fallback to VanillaReader."
            )
            return VanillaReader().read(file_path=file_path, **kwargs)

        if self.model is not None:
            reader = self._make_docling_reader(prompt)
        else:
            reader = DocumentConverter()

        # Read and convert to markdown
        text = reader.convert(file_path)
        markdown_text = text.document.export_to_markdown()

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id") or str(uuid.uuid4()),
            conversion_method="markdown",
            reader_method="docling",
            ocr_method=self.model_name,
            metadata=kwargs.get("metadata"),
        )
