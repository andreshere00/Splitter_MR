import os
import re
import uuid
from typing import Any, Optional

from docling.document_converter import DocumentConverter

from ...model import BaseModel
from ...schema import ReaderOutput
from ..base_reader import BaseReader
from ..utils import DoclingUtils
from .vanilla_reader import VanillaReader


class DoclingReader(BaseReader):
    """
    High-level reader leveraging IBM Docling for text extraction and optional image captioning.

    Supports Markdown conversion for various formats, with fine-grained control over
    image embedding, captioning, and placeholders. Automatically falls back to
    VanillaReader for unsupported extensions.
    """

    SUPPORTED_EXTENSIONS = {
        "md",
        "markdown",
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "html",
        "htm",
        "odt",
        "rtf",
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "gif",
        "tiff",
    }

    _IMAGE_PATTERN = re.compile(
        r"!\[(?P<alt>[^\]]*?)\]"
        r"\((?P<uri>data:image/[a-zA-Z0-9.+-]+;base64,(?P<b64>[A-Za-z0-9+/=]+))\)"
    )

    def __init__(self, model: Optional[BaseModel] = None) -> None:
        """
        Initialize the DoclingReader.

        Args:
            model: Optional vision-language model for image captioning and VLM-based PDF analysis.
        """
        self.model = model
        self.client = None
        self.model_name: Optional[str] = None
        if model:
            self.client = model.get_client()
            self.model_name = model.model_name
            for attr in ("_azure_deployment", "_azure_endpoint", "_api_version"):
                setattr(self, attr, getattr(self.client, attr, None))
        self._utils = DoclingUtils()

    def read(
        self,
        file_path: str,
        *,
        prompt: str = (
            "Provide a short, descriptive caption for this image. "
            "Return only the caption, in emphasis markdown (e.g., *A cat sitting*)."
        ),
        scan_pdf_pages: bool = False,
        show_base64_images: bool = False,
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Convert a document to Markdown, optionally captioning or placeholdering images.

        Args:
            file_path: Path or URL to the document.
            prompt: Prompt for image captioning or page-level VLM.
            scan_pdf_pages: If True and model provided, use VLM on full pages.
            show_base64_images: If True, embed images; if False, replace with captions or placeholders.
            **kwargs: Additional metadata (e.g., document_id).

        Returns:
            ReaderOutput with Markdown text and metadata.
        """
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"Warning: unsupported extension '{ext}'. Using VanillaReader.")
            return VanillaReader().read(file_path=file_path, **kwargs)

        if ext == "pdf":
            md = self._read_pdf(
                file_path=file_path, prompt=prompt, scan_pdf_pages=scan_pdf_pages
            )
        else:
            md = self._read_non_pdf(file_path=file_path, prompt=prompt)

        # Post-process images
        return ReaderOutput(
            text=self._process_images(md, prompt, show_base64_images),
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method="markdown",
            reader_method="docling",
            ocr_method=self.model_name,
            metadata=kwargs.get("metadata"),
        )

    def _read_pdf(
        self,
        file_path: str,
        scan_pdf_pages: bool,
        prompt: str = "Extract all the elements detected in the page, orderly. Return only all the extracted content, always in markdown format.",
    ) -> str:
        """
        Extract Markdown from a PDF, using VLM or image pipeline.

        Args:
            file_path: Path to the PDF file.
            prompt: Prompt for page-level VLM if used.
            scan_pdf_pages: Whether to send full pages to the VLM.

        Returns:
            Raw Markdown with embedded base64 images.
        """
        if scan_pdf_pages and self.model:
            pipeline = self._utils.get_pdf_pipeline(
                mode="vlm",
                client=self.client,
                model_name=self.model_name,
                prompt=prompt,
            )
            return pipeline.convert(str(file_path)).document.export_to_markdown()

        pipeline = self._utils.get_pdf_pipeline(mode="image")
        return pipeline.convert(str(file_path)).document.export_to_markdown(
            image_mode="embedded"
        )

    def _read_non_pdf(
        self,
        file_path: str,
        prompt: str = "Extract all the elements detected in the page, orderly. Return only all the extracted content, always in markdown format.",
    ) -> str:
        """
        Convert non-PDF documents via Docling or fallback converter.

        Args:
            file_path: Path to the document.
            prompt: Prompt for VLM if model provided.

        Returns:
            Raw Markdown output.
        """
        if self.model:
            reader = self._utils.get_pdf_pipeline(
                mode="vlm",
                client=self.client,
                model_name=self.model_name,
                prompt=prompt,
            )
        else:
            reader = DocumentConverter()
        return reader.convert(file_path).document.export_to_markdown()

    def _process_images(
        self,
        markdown: str,
        show_base64_images: bool,
        prompt: str = "Provide a caption for the following image. Return the result as emphasis in markdown code format (e.g., *Description of the image*).",
    ) -> str:
        """
        Handle embedded base64 images: caption, embed, or placeholder.

        Args:
            markdown: Markdown text with embedded images.
            prompt: Caption prompt for VLM.
            show_base64_images: Flag to embed images or replace them.

        Returns:
            Processed Markdown.
        """
        # Caption images if model available
        if self.model:
            markdown = self._IMAGE_PATTERN.sub(
                lambda m: self._format_caption(m, prompt, show_base64_images), markdown
            )

        # Replace remaining images if hiding
        if not show_base64_images:
            markdown = self._IMAGE_PATTERN.sub("<!-- image -->", markdown)

        return markdown

    def _format_caption(
        self,
        match: re.Match,
        show_base64_images: bool,
        prompt: str = "Provide a caption for the following image. Return the result as emphasis in markdown code format (e.g., *Description of the image*).",
    ) -> str:
        """
        Generate caption for a base64 image match.

        Args:
            match: Regex match with groups 'alt', 'uri', 'b64'.
            prompt: Prompt for caption model.
            show_base64_images: Whether to keep embedded images.

        Returns:
            Markdown string with image (optional) and caption comment.
        """
        alt = match.group("alt")
        uri = match.group("uri")
        b64 = match.group("b64")
        caption = (
            f"\n<!-- image -->\n{self.model.extract_text(file=b64, prompt=prompt)}\n"
        )
        if show_base64_images:
            return f"![{alt}]({uri})\n{caption}"  # keep image then caption
        return caption  # caption only
