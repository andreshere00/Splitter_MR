import io
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Set

import fitz
from markitdown import MarkItDown
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from ...model import BaseVisionModel
from ...schema import DEFAULT_IMAGE_EXTRACTION_PROMPT, ReaderOutput
from ..base_reader import BaseReader


class MarkItDownReader(BaseReader):
    """
    Read multiple file types using Microsoft's MarkItDown library, and convert
    the documents using markdown format.

    This reader supports both standard MarkItDown conversion and the use of Vision Language Models (VLMs)
    for LLM-based OCR when extracting text from images or scanned documents.
    """

    def __init__(self, model: BaseVisionModel = None) -> None:
        """
        Initializer method for MarkItDownReader

        Args:
            model (Optional[BaseVisionModel], optional): An optional vision-language
                model instance used for PDF pipelines that require image captioning
                or per-page analysis. If provided, the model’s client and metadata
                (e.g., Azure deployment settings) are stored for use in downstream
                processing. Defaults to None.
        """
        self.model = model
        self.model_name = model.model_name if self.model else None

    def _convert_to_pdf(self, file_path: str) -> str:
        """
        Converts DOCX, PPTX, or XLSX to PDF using LibreOffice (headless mode).

        Args:
            file_path (str): Path to the Office file.
            ext (str): File extension (lowercase, no dot).

        Returns:
            str: Path to the converted PDF.

        Raises:
            RuntimeError: If conversion fails or LibreOffice is not installed.
        """
        if not shutil.which("soffice"):
            raise RuntimeError(
                "LibreOffice (soffice) is required for Office to PDF conversion but was not found in PATH. "
                "Please install LibreOffice or set split_by_pages=False. "
                "How to install: https://www.libreoffice.org/get-help/install-howto/"
            )

        outdir = tempfile.mkdtemp()
        # Use soffice (LibreOffice) in headless mode
        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            outdir,
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to convert {file_path} to PDF: {result.stderr.decode()}"
            )
        pdf_name = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        pdf_path = os.path.join(outdir, pdf_name)
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"PDF was not created: {pdf_path}")
        return pdf_path

    def _pdf_pages_to_streams(self, pdf_path: str) -> List[io.BytesIO]:
        """
        Convert each PDF page to a PNG and wrap in a BytesIO stream.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[io.BytesIO]: List of PNG image streams for each page.
        """
        doc = fitz.open(pdf_path)
        streams = []
        for idx in range(len(doc)):
            pix = doc.load_page(idx).get_pixmap()
            buf = io.BytesIO(pix.tobytes("png"))
            buf.name = f"page_{idx + 1}.png"
            buf.seek(0)
            streams.append(buf)
        return streams

    def _split_pdf_to_temp_pdfs(self, pdf_path: str) -> List[str]:
        """
        Split a PDF file into single-page temporary PDF files.

        Args:
            pdf_path (str): Path to the PDF file to split.

        Returns:
            List[str]: List of file paths for the temporary single-page PDFs.

        Example:
            temp_files = self._split_pdf_to_temp_pdfs("document.pdf")
            # temp_files = ["/tmp/tmpa1b2c3.pdf", "/tmp/tmpd4e5f6.pdf", ...]
        """
        temp_files = []
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                writer.write(tmp)
                temp_files.append(tmp.name)
        return temp_files

    def _pdf_pages_to_markdown(
        self, file_path: str, md: MarkItDown, prompt: str, page_placeholder: str
    ) -> str:
        """
        Convert each scanned PDF page to markdown using the provided MarkItDown instance.

        Args:
            file_path (str): Path to PDF.
            md (MarkItDown): The MarkItDown converter instance.
            prompt (str): The LLM prompt for OCR.
            page_placeholder (str): Page break placeholder for markdown.

        Returns:
            str: Markdown of the entire PDF (one page per placeholder).
        """
        page_md = []
        for idx, page_stream in enumerate(
            self._pdf_pages_to_streams(file_path), start=1
        ):
            page_md.append(page_placeholder.replace("{page}", str(idx)))
            result = md.convert(page_stream, llm_prompt=prompt)
            page_md.append(result.text_content)
        return "\n".join(page_md)

    def _pdf_file_per_page_to_markdown(
        self, file_path: str, md: "MarkItDown", prompt: str, page_placeholder: str
    ) -> str:
        """
        Convert each page of a PDF to markdown by splitting the PDF into temporary single-page files,
        extracting text from each page using MarkItDown, and joining the results with a page placeholder.

        Args:
            file_path (str): Path to the PDF file.
            md (MarkItDown): The MarkItDown converter instance.
            prompt (str): The LLM prompt for extraction.
            page_placeholder (str): Markdown placeholder for page breaks; supports '{page}' for numbering.

        Returns:
            str: Concatenated markdown content for the entire PDF, separated by page placeholders.

        Raises:
            Any exception raised by MarkItDown or file I/O will propagate.

        Example:
            markdown = self._pdf_file_per_page_to_markdown("doc.pdf", md, prompt, "<!-- page {page} -->")
        """
        temp_files = self._split_pdf_to_temp_pdfs(pdf_path=file_path)
        page_md = []
        try:
            for idx, temp_pdf in enumerate(temp_files, start=1):
                page_md.append(page_placeholder.replace("{page}", str(idx)))
                result = md.convert(temp_pdf, llm_prompt=prompt)
                page_md.append(result.text_content)
            return "\n".join(page_md)
        finally:
            # Clean up temp files
            for temp_pdf in temp_files:
                os.remove(temp_pdf)

    def _get_markitdown(self) -> tuple:
        """
        Returns a MarkItDown instance and OCR method name depending on model presence.

        Returns:
            tuple[MarkItDown, Optional[str]]: MarkItDown instance, OCR method or None.

        Raises:
            ValueError: If provided model is not supported.
        """
        if self.model:
            self.client = self.model.get_client()
            if not isinstance(self.client, OpenAI):
                raise ValueError(
                    "Incompatible client. Only models that use the OpenAI client are supported."
                )
            return (
                MarkItDown(llm_client=self.client, llm_model=self.model.model_name),
                self.model.model_name,
            )
        else:
            return MarkItDown(), None

    def read(self, file_path: Path | str = None, **kwargs: Any) -> ReaderOutput:
        """
        Reads a file and converts its contents to Markdown using MarkItDown.

        Features:
            - Standard file-to-Markdown conversion for most formats.
            - LLM-based OCR (if a Vision model is provided) for images and scanned PDFs.
            - Optional PDF page-wise OCR with fine-grained control and custom LLM prompt.

        Args:
            file_path (str): Path to the input file to be read and converted.
            **kwargs:
                - `document_id (Optional[str])`: Unique document identifier.
                    If not provided, a UUID will be generated.
                - `metadata (Dict[str, Any], optional)`: Additional metadata, given in dictionary format.
                    If not provided, no metadata is returned.
                - `prompt (Optional[str])`: Prompt for image captioning or VLM extraction.
                - `page_placeholder (str)`: Markdown placeholder string for pages (default: "<!-- page -->").
                - split_by_pages (bool): If True and the input is a PDF, split the PDF by pages and process
                    each page separately. Default is False.

        Returns:
            ReaderOutput: Dataclass defining the output structure for all readers.

        Example:
            ```python
            from splitter_mr.model import OpenAIVisionModel
            from splitter_mr.reader import MarkItDownReader

            model = AzureOpenAIVisionModel()
            reader = MarkItDownReader(model=model)
            output = reader.read(file_path="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/lorem_ipsum.pdf")
            print(output.text)
            ```
            ```python
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """
        # Initialize MarkItDown reader
        file_path: str | Path = os.fspath(file_path)
        ext: str = os.path.splitext(file_path)[1].lower().lstrip(".")
        prompt: str = kwargs.get("prompt", DEFAULT_IMAGE_EXTRACTION_PROMPT)
        page_placeholder: str = kwargs.get("page_placeholder", "<!-- page -->")
        split_by_pages: bool = kwargs.get("split_by_pages", False)
        conversion_method: str = None
        md, ocr_method = self._get_markitdown()

        PDF_CONVERTIBLE_EXT: Set[str] = {"docx", "pptx", "xlsx"}

        if split_by_pages and ext != "pdf":
            if ext in PDF_CONVERTIBLE_EXT:
                file_path = self._convert_to_pdf(file_path)

        md, ocr_method = self._get_markitdown()

        # Process text
        if split_by_pages:
            markdown_text = self._pdf_file_per_page_to_markdown(
                file_path=file_path,
                md=md,
                prompt=prompt,
                page_placeholder=page_placeholder,
            )
            conversion_method = "markdown"
        elif self.model is not None:
            markdown_text = self._pdf_pages_to_markdown(
                file_path=file_path,
                md=md,
                prompt=prompt,
                page_placeholder=page_placeholder,
            )
            conversion_method = "markdown"
        else:
            markdown_text = md.convert(file_path, llm_prompt=prompt).text_content
            conversion_method = "json" if ext == "json" else "markdown"

        page_placeholder_value = (
            page_placeholder
            if page_placeholder and page_placeholder in markdown_text
            else None
        )

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method=conversion_method,
            reader_method="markitdown",
            ocr_method=ocr_method,
            page_placeholder=page_placeholder_value,
            metadata=kwargs.get("metadata", {}),
        )
