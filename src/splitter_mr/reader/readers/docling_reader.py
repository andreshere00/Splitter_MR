import os
import re
import uuid
from typing import Any, Optional

from docling.document_converter import DocumentConverter

from ...model import BaseModel
from ...schema import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DOCLING_SUPPORTED_EXTENSIONS,
    ReaderOutput,
)
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

    SUPPORTED_EXTENSIONS = DOCLING_SUPPORTED_EXTENSIONS

    _IMAGE_PATTERN = re.compile(
        r"!\[(?P<alt>[^\]]*?)\]"
        r"\((?P<uri>data:image/[a-zA-Z0-9.+-]+;base64,(?P<b64>[A-Za-z0-9+/=]+))\)"
    )

    def __init__(self, model: Optional[BaseModel] = None) -> None:
        """
        Initialize the DoclingReader.

        Args:
            model: Optional vision-language model for image captioning and VLM-based PDF analyzis.
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
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Convert a document to Markdown, with advanced image and PDF handling. Use `scan_pdf_pages=True` (with a model)
        to extract each PDF page with a VLM. Use a custom `prompt` for VLM-based image captioning or PDF extraction.
        Set `show_base64_images=True` to embed images; `False` to replace with captions/placeholders. Note that it fallbacks
        to VanillaReader for unsupported formats.

        Args:
            file_path (str): Path or URL to the document.
            **kwargs:
                - `document_id (Optional[str])`: Unique document identifier.
                    If not provided, a UUID will be generated.
                - `metadata (Optional[Dict[str, Any]])`: Additional metadata, given in dictionary format.
                    If not provided, no metadata is returned.
                - `prompt (Optional[str])`: Prompt for image captioning or VLM extraction.
                - `scan_pdf_pages (Optional[bool])`: If True (and model provided), extract PDFs page-by-page with VLM.
                - `show_base64_images (Optional[bool])`: If True, embed base64 images in Markdown.
                    If False, replace with captions or comments.
                - `placeholder (Optional[str])`: placeholder to indicate where is located an image in the document.

        Returns:
            ReaderOutput: Contains Markdown text and metadata.

        Example:
            ```python
            # Basic Markdown extraction:
            reader = DoclingReader()
            result = reader.read(file_path="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/lorem_ipsum.pdf")
            print(output.text)
            ```
            ```python
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"Warning: unsupported extension '{ext}'. Using VanillaReader.")
            return VanillaReader().read(file_path=file_path, **kwargs)

        if ext == "pdf":
            md = self._read_pdf(
                file_path=file_path,
                prompt=kwargs.get("prompt", DEFAULT_IMAGE_CAPTION_PROMPT),
                scan_pdf_pages=kwargs.get("scan_pdf_pages", False),
            )
        else:
            md = self._read_non_pdf(
                file_path=file_path,
                prompt=kwargs.get("prompt", DEFAULT_EXTRACTION_PROMPT),
            )

        # Image post-processing
        text = self._process_images(
            markdown=md,
            prompt=kwargs.get("prompt", DEFAULT_IMAGE_CAPTION_PROMPT),
            show_base64_images=kwargs.get("show_base64_images", False),
            placeholder=kwargs.get("placeholder", "<!-- image -->"),
        )

        # Return output
        return ReaderOutput(
            text=text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method="markdown",
            reader_method="docling",
            ocr_method=self.model_name,
            metadata=kwargs.get("metadata"),
        )

    def _read_pdf(self, file_path: str, scan_pdf_pages: bool, prompt: str) -> str:
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

        pipeline = self._utils.get_pdf_pipeline(
            mode="image",
            client=self.client,
            model_name=self.model_name,
            prompt=prompt,
        )
        return pipeline.convert(str(file_path)).document.export_to_markdown(
            image_mode="embedded"
        )

    def _read_non_pdf(
        self, file_path: str, prompt: str = DEFAULT_EXTRACTION_PROMPT
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
        prompt: str,
        placeholder: str = "<!-- image -->",
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
                lambda m: self._format_caption(
                    match=m,
                    prompt=prompt,
                    show_base64_images=show_base64_images,
                    placeholder=placeholder,
                ),
                markdown,
            )

        # Replace remaining images if hiding
        if not show_base64_images:
            markdown = self._IMAGE_PATTERN.sub(placeholder, markdown)

        return markdown

    def _format_caption(
        self,
        match: re.Match,
        show_base64_images: bool,
        prompt: str,
        placeholder: str = "<!-- image -->",
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

        caption = f"\n{self.model.extract_text(
            file=b64, 
            prompt=prompt
        )}"
        if show_base64_images:
            return f"![{alt}]({uri})\n{caption}"
        return f"{placeholder}\n{caption}"
