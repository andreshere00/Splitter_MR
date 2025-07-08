import os
import re
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

from ...model import BaseModel
from ...schema import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DOCLING_SUPPORTED_EXTENSIONS,
    ReaderOutput,
)
from ..base_reader import BaseReader
from ..utils import DoclingPipelineFactory
from .vanilla_reader import VanillaReader


class DoclingReader(BaseReader):
    """
    High-level document reader leveraging IBM Docling for flexible document-to-Markdown conversion,
    with optional image captioning or VLM-based PDF processing. Supports automatic pipeline selection,
    seamless integration with custom vision-language models, and configurable output for both PDF
    and non-PDF files.
    """

    SUPPORTED_EXTENSIONS = DOCLING_SUPPORTED_EXTENSIONS

    _IMAGE_PATTERN = re.compile(
        r"!\[(?P<alt>[^\]]*?)\]"
        r"\((?P<uri>data:image/[a-zA-Z0-9.+-]+;base64,(?P<b64>[A-Za-z0-9+/=]+))\)"
    )

    def __init__(self, model: Optional[BaseModel] = None) -> None:
        self.model = model
        self.client = None
        self.model_name: Optional[str] = None
        if model:
            self.client = model.get_client()
            self.model_name = model.model_name
            for attr in ("_azure_deployment", "_azure_endpoint", "_api_version"):
                setattr(self, attr, getattr(self.client, attr, None))

    def read(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Reads a document, automatically selecting the appropriate Docling pipeline for extraction.
        Supports PDFs (per-page VLM or standard extraction), as well as other file types.

        Args:
            file_path (str | Path): Path or URL to the document file.
            **kwargs: Keyword arguments to control extraction, including:
                - prompt (str): Prompt for image captioning or VLM-based PDF extraction.
                - scan_pdf_pages (bool): If True (and model provided), analyze each PDF page via VLM.
                - show_base64_images (bool): If True, embed base64 images in Markdown; if False, use image placeholders.
                - page_placeholder (str): Placeholder for page breaks in output Markdown.
                - image_placeholder (str): Placeholder for image locations in output Markdown.
                - image_resolution (float): Resolution scaling factor for image extraction.
                - document_id (Optional[str]): Optional document ID for metadata.
                - metadata (Optional[dict]): Optional metadata dictionary.

        Returns:
            ReaderOutput: Extracted document in Markdown format and associated metadata.

        Raises:
            Warning: If a file extension is unsupported, falls back to VanillaReader and emits a warning.
            ValueError: If PDF pipeline requirements are not satisfied (e.g., neither model nor show_base64_images provided).
        """

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext not in self.SUPPORTED_EXTENSIONS:
            warnings.warn(f"Unsupported extension '{ext}'. Using VanillaReader.")
            return VanillaReader().read(file_path=file_path, **kwargs)

        # Pipeline selection and execution
        pipeline_name, pipeline_args = self._select_pipeline(file_path, ext, **kwargs)
        md = DoclingPipelineFactory.run(pipeline_name, file_path, **pipeline_args)

        # TODO: Image post-processing, if needed.
        text = md

        return ReaderOutput(
            text=text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method="markdown",
            reader_method="docling",
            ocr_method=self.model_name,
            metadata=kwargs.get("metadata", {}),
        )

    def _select_pipeline(self, file_path: str, ext: str, **kwargs) -> tuple[str, dict]:
        """
        Decides which pipeline to use and prepares arguments for it.

        Args:
            file_path (str): Path to the input document.
            ext (str): File extension.
            **kwargs: Extraction and pipeline control options, including:
                - prompt (str)
                - scan_pdf_pages (bool)
                - show_base64_images (bool)
                - page_placeholder (str)
                - image_placeholder (str)
                - image_resolution (float)

        Returns:
            tuple[str, dict]: Name of the selected pipeline and the dictionary of arguments for that pipeline.

        Pipeline selection logic:
            - For PDFs:
                - If scan_pdf_pages is True: uses per-page VLM/image pipeline.
                - Else if model is provided: uses VLM pipeline.
                - Else: uses default Markdown pipeline.
            - For other extensions: always uses Markdown pipeline.
        """
        # Defaults
        show_base64_images = kwargs.get("show_base64_images", False)
        page_placeholder = kwargs.get("page_placeholder", "<!-- page -->")
        image_placeholder = kwargs.get("image_placeholder", "<!-- image -->")
        image_resolution = kwargs.get("image_resolution", 1.0)
        scan_pdf_pages = kwargs.get("scan_pdf_pages", False)

        # --- PDF logic ---
        if ext == "pdf":
            if scan_pdf_pages:
                # Scan pages as images and extract their content
                pipeline_args = {
                    "model": self.model,
                    "prompt": kwargs.get("prompt", DEFAULT_EXTRACTION_PROMPT),
                    "image_resolution": image_resolution,
                    "page_placeholder": page_placeholder,
                    "show_base64_images": show_base64_images,
                }
                pipeline_name = "page_image"
            else:
                if self.model:
                    if show_base64_images:
                        warnings.warn(
                            "When using a model, base64 images are not rendered. So, deactivate the `show_base64_images` option or don't provide the model in the class constructor."
                        )
                    # Use VLM pipeline for the whole PDF
                    pipeline_args = {
                        "model": self.model,
                        "prompt": kwargs.get("prompt", DEFAULT_IMAGE_CAPTION_PROMPT),
                        "page_placeholder": page_placeholder,
                        "image_placeholder": image_placeholder,
                    }
                    pipeline_name = "vlm"
                else:
                    # No model: use markdown pipeline (default docling, base64 or placeholders)
                    pipeline_args = {
                        "show_base64_images": show_base64_images,
                        "page_placeholder": page_placeholder,
                        "image_placeholder": image_placeholder,
                        "image_resolution": image_resolution,
                        "ext": ext,
                    }
                    pipeline_name = "markdown"
        else:
            # For non-PDF: use markdown pipeline
            pipeline_args = {
                "show_base64_images": show_base64_images,
                "page_placeholder": page_placeholder,
                "image_placeholder": image_placeholder,
                "ext": ext,
            }
            pipeline_name = "markdown"

        return pipeline_name, pipeline_args
