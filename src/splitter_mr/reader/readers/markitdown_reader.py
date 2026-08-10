import io
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Optional

import fitz
from markitdown import MarkItDown
from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from ...model import BaseVisionModel
from ...schema import (
    DEFAULT_IMAGE_EXTRACTION_PROMPT,
    DEFAULT_PAGE_PLACEHOLDER,
    ReaderOutput,
)
from ...schema.exceptions import (
    MarkItDownReaderException,
    ReaderConfigException,
    ReaderOutputException,
)
from ..base_reader import BaseReader


class MarkItDownReader(BaseReader):
    """Reads multiple file types using Microsoft's MarkItDown library and converts them to Markdown.

    This reader serves as a bridge between standard document formats (PDF, DOCX, PPTX, XLSX)
    and Markdown. It supports two modes of operation:
    1.  **Standard Conversion:** Uses native parsers for high-fidelity text extraction.
    2.  **VLM-Enhanced Conversion:** Integrates with Vision Language Models (VLMs) via the
        `BaseVisionModel` interface to perform LLM-based OCR on images or scanned documents.

    Attributes:
        model (BaseVisionModel): The vision model instance used for OCR/Captioning tasks.
        model_name (str): The identifier of the model (e.g., 'gpt-5.6-luna'), used for metadata.
        client (OpenAI): The OpenAI-compatible client extracted from the vision model.

    Raises:
        ReaderConfigException: If the provided model uses an incompatible client (non-OpenAI).
    """

    def __init__(self, model: BaseVisionModel = None) -> None:
        """Initializes the MarkItDownReader.

        Args:
            model (Optional[BaseVisionModel], optional): An optional vision-language model
                wrapper. If provided, its underlying client is injected into the MarkItDown
                instance to enable image description and optical character recognition.

        Raises:
            ReaderConfigException: If the `model` provided does not expose an `OpenAI`
                compatible client, or if initialization fails unexpectedly.
        """
        try:
            self.model: BaseVisionModel = model
            self.model_name: str = model.model_name if self.model else None

            # Pre-validate client compatibility if model is provided
            if self.model:
                client = self.model.get_client()
                if not isinstance(client, OpenAI):
                    raise ReaderConfigException(
                        f"Incompatible client type: {type(client)}. "
                        "MarkItDownReader currently only supports models using the OpenAI client."
                    )
        except Exception as e:
            if isinstance(e, ReaderConfigException):
                raise
            raise ReaderConfigException(
                f"Failed to initialize MarkItDownReader: {str(e)}"
            ) from e

    def _convert_to_pdf(self, file_path: str | Path) -> str:
        """Converts Office documents (DOCX, PPTX, XLSX) to PDF using headless LibreOffice.

        This method acts as a pre-processing step when `split_by_pages=True` is requested for
        office formats. It delegates conversion to the system's installed `soffice` binary.

        Args:
            file_path (str | Path): The path to the source Office file.

        Returns:
            str: The absolute path to the newly created PDF file located in a temporary directory.

        Raises:
            MarkItDownReaderException:
                - If the `soffice` binary is not found in the system PATH.
                - If the subprocess returns a non-zero exit code.
                - If the expected output PDF file was not created.
        """
        if not shutil.which("soffice"):
            raise MarkItDownReaderException(
                "LibreOffice (soffice) is required for Office to PDF conversion but was not found in PATH. "
                "Please install LibreOffice or set split_by_pages=False. "
                "How to install: https://www.libreoffice.org/get-help/install-howto/"
            )

        try:
            outdir: str = tempfile.mkdtemp()
            # Use soffice (LibreOffice) in headless mode
            cmd: list[str] = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                outdir,
                str(file_path),
            ]

            result: CompletedProcess[bytes] = subprocess.run(
                cmd, capture_output=True, check=False
            )

            if result.returncode != 0:
                raise MarkItDownReaderException(
                    f"LibreOffice conversion failed for {file_path}.\n"
                    f"Stderr: {result.stderr.decode() if result.stderr else 'Unknown error'}"
                )

            filename = os.path.basename(file_path)
            pdf_name = os.path.splitext(filename)[0] + ".pdf"
            pdf_path = os.path.join(outdir, pdf_name)

            if not os.path.exists(pdf_path):
                raise MarkItDownReaderException(
                    f"PDF was not created at expected path: {pdf_path}"
                )

            return pdf_path

        except subprocess.SubprocessError as e:
            raise MarkItDownReaderException(
                f"Subprocess error during PDF conversion: {str(e)}"
            ) from e
        except OSError as e:
            raise MarkItDownReaderException(
                f"I/O error during PDF conversion: {str(e)}"
            ) from e

    def _pdf_pages_to_streams(self, pdf_path: str | Path) -> list[io.BytesIO]:
        """Rasterizes PDF pages into in-memory PNG streams using PyMuPDF (fitz).

        This method is preferred when processing speed is prioritized over memory usage.
        It avoids writing intermediate image files to disk.

        Args:
            pdf_path (str | Path): The path to the PDF file.

        Returns:
            list[io.BytesIO]: A list of byte streams, where each stream contains a PNG
                representation of a single PDF page.

        Raises:
            MarkItDownReaderException: If PyMuPDF encounters a corrupted file or fails to render.
        """
        try:
            doc = fitz.open(pdf_path)
            streams: list[io.BytesIO] = []
            for idx in range(len(doc)):
                pix = doc.load_page(idx).get_pixmap()
                buf = io.BytesIO(pix.tobytes("png"))
                buf.name = f"page_{idx + 1}.png"
                buf.seek(0)
                streams.append(buf)
            return streams
        except Exception as e:
            raise MarkItDownReaderException(
                f"Failed to convert PDF pages to image streams via PyMuPDF: {str(e)}"
            ) from e

    def _split_pdf_to_temp_pdfs(self, pdf_path: str | Path) -> list[str]:
        """Splits a multi-page PDF into multiple single-page PDF files on disk.

        This approach is safer than in-memory streams for extremely large documents or
        when the downstream converter requires a physical file path rather than a byte stream.

        Args:
            pdf_path (str | Path): The path to the source PDF.

        Returns:
            list[str]: A list of absolute file paths to the temporary single-page PDFs.

        Note:
            The caller is responsible for cleaning up these temporary files.
            (Handled automatically in `_pdf_file_per_page_to_markdown`).

        Raises:
            MarkItDownReaderException: If `pypdf` fails to read or split the document.
        """
        temp_files: list[str] = []
        try:
            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    writer.write(tmp)
                    temp_files.append(tmp.name)
            return temp_files
        except Exception as e:
            # Clean up any files created before the failure
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            raise MarkItDownReaderException(f"Failed to split PDF: {str(e)}") from e

    def _pdf_pages_to_markdown(
        self, file_path: str, md: "MarkItDown", prompt: str, page_placeholder: str
    ) -> str:
        """Processes a PDF by converting each page to an image stream and then to Markdown.

        This method is typically used when OCR is needed on a per-page basis without
        creating intermediate PDF files.

        Args:
            file_path (str): Path to the source PDF.
            md (MarkItDown): The configured MarkItDown instance.
            prompt (str): The LLM prompt used for image extraction/OCR.
            page_placeholder (str): The string used to separate pages (e.g., '').

        Returns:
            str: The concatenated Markdown content of all pages.

        Raises:
            MarkItDownReaderException: If conversion fails for a specific page.
        """
        # Exceptions here are caught by the calling methods or bubble up as MarkItDownReaderException
        # from the _pdf_pages_to_streams call.
        page_md: list[str] = []
        streams = self._pdf_pages_to_streams(file_path)

        try:
            for idx, page_stream in enumerate(streams, start=1):
                page_md.append(page_placeholder.replace("{page}", str(idx)))
                try:
                    result = md.convert(page_stream, llm_prompt=prompt)
                    page_md.append(result.text_content)
                except Exception as e:
                    raise MarkItDownReaderException(
                        f"MarkItDown conversion failed on page {idx} of PDF image stream: {str(e)}"
                    ) from e
            return "\n".join(page_md)
        finally:
            # Close streams
            for s in streams:
                s.close()

    def _pdf_file_per_page_to_markdown(
        self, file_path: str, md: "MarkItDown", prompt: str, page_placeholder: str
    ) -> str:
        """Processes a PDF by splitting it into temp files and converting each individually.

        This method provides robust isolation: if one page crashes the converter due to
        file corruption, it allows for easier debugging (though currently it fails fast).

        Args:
            file_path (str): Path to the source PDF.
            md (MarkItDown): The configured MarkItDown instance.
            prompt (str): The LLM prompt used for extraction.
            page_placeholder (str): The string used to separate pages.

        Returns:
            str: The concatenated Markdown content.

        Raises:
            MarkItDownReaderException: If splitting or conversion fails.
        """
        temp_files: list[str] = self._split_pdf_to_temp_pdfs(pdf_path=file_path)
        page_md: list[str] = []

        try:
            for idx, temp_pdf in enumerate(temp_files, start=1):
                page_md.append(page_placeholder.replace("{page}", str(idx)))
                try:
                    result = md.convert(temp_pdf, llm_prompt=prompt)
                    page_md.append(result.text_content)
                except Exception as e:
                    raise MarkItDownReaderException(
                        f"MarkItDown conversion failed on page {idx} (temp file {temp_pdf}): {str(e)}"
                    ) from e
            return "\n".join(page_md)
        finally:
            # Clean up temp files
            for temp_pdf in temp_files:
                try:
                    if os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
                except OSError:
                    pass  # Best effort cleanup

    def _get_markitdown(self) -> tuple["MarkItDown", Optional[str]]:
        """Configures and returns the MarkItDown instance based on available models.

        Returns:
            tuple[MarkItDown, Optional[str]]:
                - A configured `MarkItDown` instance.
                - The name of the OCR model used (if any), or None.

        Raises:
            ReaderConfigException: If the OpenAI client cannot be retrieved from the model.
        """
        if self.model:
            try:
                self.client = self.model.get_client()
                # Double check client in case it changed or wasn't checked in init
                if not isinstance(self.client, OpenAI):
                    raise ValueError("Client must be an instance of OpenAI.")

                return (
                    MarkItDown(llm_client=self.client, llm_model=self.model.model_name),
                    self.model.model_name,
                )
            except Exception as e:
                raise ReaderConfigException(
                    f"Failed to configure MarkItDown with model: {str(e)}"
                ) from e
        else:
            return MarkItDown(), None

    def read(self, file_path: Path | str = None, **kwargs: Any) -> ReaderOutput:
        """Orchestrates the file reading and conversion process.

        This method handles file existence checks, format detection, optional
        Office-to-PDF conversion, and the final Markdown extraction.

        Args:
            file_path (Path | str): The absolute or relative path to the input file.
            **kwargs: Additional configuration parameters:
                - document_id (str, optional): A unique ID for the document. Defaults to UUID.
                - metadata (dict, optional): Metadata to attach to the output.
                - prompt (str, optional): Custom prompt for VLM-based extraction.
                - page_placeholder (str, optional): String to delimit pages (default: '').
                - split_by_pages (bool, optional): If True, splits PDFs/Office docs by page
                  before processing. This is useful for granular OCR control but requires
                  LibreOffice for Office files. Defaults to False.

        Returns:
            ReaderOutput: A standardized object containing the extracted text, metadata,
            and processing details.

        Raises:
            ReaderConfigException: If `file_path` is missing.
            MarkItDownReaderException: For file not found, conversion failures, or internal library errors.
            ReaderOutputException: If the final output object cannot be constructed.
        """
        if not file_path:
            raise ReaderConfigException("file_path must be provided.")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise MarkItDownReaderException(f"File not found: {file_path}")

        file_path_str: str = os.fspath(file_path_obj)
        ext: str = file_path_obj.suffix.lower().lstrip(".")

        prompt: str = kwargs.get("prompt", DEFAULT_IMAGE_EXTRACTION_PROMPT)
        page_placeholder: str = kwargs.get("page_placeholder", DEFAULT_PAGE_PLACEHOLDER)
        split_by_pages: bool = kwargs.get("split_by_pages", False)

        # Determine conversion strategy
        try:
            md, ocr_method = self._get_markitdown()
        except Exception as e:
            raise MarkItDownReaderException(
                f"Failed to initialize MarkItDown instance: {str(e)}"
            ) from e

        PDF_CONVERTIBLE_EXT: set[str] = {"docx", "pptx", "xlsx"}

        # Handle Office -> PDF conversion
        if split_by_pages and ext in PDF_CONVERTIBLE_EXT:
            try:
                file_path_str = self._convert_to_pdf(file_path_str)
            except Exception as e:
                raise MarkItDownReaderException(
                    f"Pre-conversion of {ext} to PDF failed: {str(e)}"
                ) from e

        # Process text
        try:
            if split_by_pages:
                markdown_text: str = self._pdf_file_per_page_to_markdown(
                    file_path=file_path_str,
                    md=md,
                    prompt=prompt,
                    page_placeholder=page_placeholder,
                )
            else:
                result = md.convert(file_path_str, llm_prompt=prompt)
                markdown_text: str = result.text_content
        except MarkItDownReaderException:
            raise  # Re-raise already wrapped exceptions
        except Exception as e:
            raise MarkItDownReaderException(
                f"MarkItDown processing failed for file {file_path_str}: {str(e)}"
            ) from e

        conversion_method = "json" if ext == "json" else "markdown"

        page_placeholder_value: str = (
            page_placeholder
            if (page_placeholder and page_placeholder in markdown_text)
            else None
        )

        # Return output
        try:
            return ReaderOutput(
                text=markdown_text,
                document_name=os.path.basename(file_path_str),
                document_path=file_path_str,
                document_id=kwargs.get("document_id", str(uuid.uuid4())),
                conversion_method=conversion_method,
                reader_method="markitdown",
                ocr_method=ocr_method,
                page_placeholder=page_placeholder_value,
                metadata=kwargs.get("metadata", {}),
            )
        except Exception as e:
            raise ReaderOutputException(
                f"Failed to construct ReaderOutput: {str(e)}"
            ) from e
