import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence

import fitz
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from PIL import Image

from ...schema import (
    DEFAULT_PAGE_PLACEHOLDER,
    TEXTRACT_IMAGE_EXTENSIONS,
    TEXTRACT_OCR_METHOD,
    TEXTRACT_OFFICE_EXTENSIONS,
    TEXTRACT_PDF_EXTENSIONS,
    TEXTRACT_SUPPORTED_FILE_EXTENSIONS,
    TEXTRACT_SYNC_MAX_BYTES,
    TEXTRACT_TEXT_EXTENSIONS,
    ReaderConfigException,
    ReaderOutput,
    TextractReaderException,
)
from ..base_reader import BaseReader
from .vanilla_reader import VanillaReader

try:
    import boto3
except ImportError as exc:  # pragma: no cover - guarded by optional extra
    boto3 = None
    _BOTO3_IMPORT_ERROR = exc
else:
    _BOTO3_IMPORT_ERROR = None


class TextractReader(BaseReader):
    """
    Read documents using AWS Textract synchronous text detection.

    Text-native formats (MD, JSON, YAML, TXT) are delegated to ``VanillaReader``.
    Visual formats are normalized to PNG pages and processed with
    ``detect_document_text`` one page at a time.
    """

    def __init__(
        self,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        client: Any = None,
    ) -> None:
        """
        Initialize the TextractReader.

        Args:
            region_name (Optional[str]): AWS region override for the Textract client.
            profile_name (Optional[str]): AWS shared-credentials profile name.
            client (Any): Optional preconfigured boto3 Textract client for testing.
        """
        if client is not None:
            self._client = client
            return

        if boto3 is None:
            raise ModuleNotFoundError(
                "TextractReader requires the 'textract' extra. "
                "Install with: pip install 'splitter-mr[textract]'"
            ) from _BOTO3_IMPORT_ERROR

        session_kwargs: dict[str, str] = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        if region_name:
            session_kwargs["region_name"] = region_name

        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "textract",
            config=Config(retries={"mode": "standard", "max_attempts": 10}),
        )

    def read(self, file_path: str | Path, **kwargs: Any) -> ReaderOutput:
        """
        Read a local document and return extracted text.

        Args:
            file_path (str | Path): Path to the input file.
            **kwargs: Optional arguments:
                document_id (str): Explicit document identifier.
                metadata (dict): Metadata attached to the output.
                page_placeholder (str): Page delimiter inserted between OCR pages.
                resolution (int): DPI used when rasterizing PDF pages (default 300).

        Returns:
            ReaderOutput: Standardized reader output.

        Raises:
            ReaderConfigException: If the file path or extension is invalid.
            TextractReaderException: If conversion or Textract processing fails.
        """
        if file_path is None:
            raise ReaderConfigException("file_path must be provided.")

        path_str = os.fspath(file_path)
        if not self.is_valid_file_path(path_str):
            raise ReaderConfigException(f"File not found: {path_str}")

        ext = Path(path_str).suffix.lower().lstrip(".")
        if ext not in TEXTRACT_SUPPORTED_FILE_EXTENSIONS:
            raise ReaderConfigException(
                f"Unsupported file extension: .{ext}. "
                "See TextractReader documentation for supported formats."
            )

        page_placeholder = kwargs.get("page_placeholder", DEFAULT_PAGE_PLACEHOLDER)
        document_id = kwargs.get("document_id", str(uuid.uuid4()))
        metadata = kwargs.get("metadata", {})

        if ext in TEXTRACT_TEXT_EXTENSIONS:
            return self._read_text_native(
                path_str,
                document_id=document_id,
                metadata=metadata,
                page_placeholder=page_placeholder,
            )

        try:
            rel_path = os.path.relpath(path_str)
        except ValueError:
            rel_path = path_str

        page_pngs = self._prepare_page_pngs(
            path_str,
            ext=ext,
            resolution=int(kwargs.get("resolution", 300)),
        )
        if not page_pngs:
            raise TextractReaderException("No pages available for Textract processing.")

        text = self._extract_text_from_pages(page_pngs, page_placeholder)
        if "{page}" in page_placeholder:
            page_ph_out: str | None = page_placeholder
        else:
            page_ph_out = (
                page_placeholder
                if page_placeholder and page_placeholder in text
                else None
            )

        return ReaderOutput(
            text=text,
            document_name=os.path.basename(path_str),
            document_path=rel_path,
            document_id=document_id,
            conversion_method="png",
            reader_method="textract",
            ocr_method=TEXTRACT_OCR_METHOD,
            page_placeholder=page_ph_out,
            metadata=metadata,
        )

    def _read_text_native(
        self,
        path_str: str,
        *,
        document_id: str,
        metadata: dict[str, Any],
        page_placeholder: str,
    ) -> ReaderOutput:
        """Delegate text-native files to VanillaReader and expose Textract identity."""
        output = VanillaReader().read(file_path=path_str, document_id=document_id)
        output.reader_method = "textract"
        output.ocr_method = None
        if metadata:
            output.append_metadata(metadata)
        if page_placeholder and page_placeholder in (output.text or ""):
            output.page_placeholder = page_placeholder
        return output

    def _prepare_page_pngs(
        self,
        path_str: str,
        *,
        ext: str,
        resolution: int,
    ) -> List[bytes]:
        """Normalize supported visual inputs into in-memory PNG page bytes."""
        if ext in TEXTRACT_OFFICE_EXTENSIONS:
            with TemporaryDirectory(prefix="textract_office2pdf_") as tmp_dir:
                pdf_path = self._convert_office_to_pdf(path_str, tmp_dir)
                return self._pdf_to_png_pages(pdf_path, resolution=resolution)

        if ext in TEXTRACT_PDF_EXTENSIONS:
            return self._pdf_to_png_pages(path_str, resolution=resolution)

        if ext in TEXTRACT_IMAGE_EXTENSIONS:
            return self._image_to_png_pages(path_str)

        raise ReaderConfigException(f"Unsupported visual extension: .{ext}")

    def _convert_office_to_pdf(self, file_path: str, outdir: str) -> str:
        """Convert Office documents to PDF using headless LibreOffice."""
        if not shutil.which("soffice"):
            raise TextractReaderException(
                "LibreOffice/soffice is required for Office-to-PDF conversion "
                "but was not found in PATH."
            )

        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            outdir,
            file_path,
        ]
        try:
            proc: CompletedProcess[bytes] = subprocess.run(
                cmd, capture_output=True, check=False
            )
        except Exception as exc:
            raise TextractReaderException(
                f"Subprocess failed when executing LibreOffice: {exc}"
            ) from exc

        if proc.returncode != 0:
            err_msg = proc.stderr.decode() if proc.stderr else "Unknown error"
            raise TextractReaderException(
                f"LibreOffice failed converting {file_path} -> PDF. "
                f"Exit code {proc.returncode}. Error: {err_msg}"
            )

        pdf_name = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        pdf_path = os.path.join(outdir, pdf_name)
        if not os.path.exists(pdf_path):
            raise TextractReaderException(
                f"LibreOffice finished, but expected PDF was not found at: {pdf_path}"
            )
        return pdf_path

    def _pdf_to_png_pages(self, pdf_path: str, *, resolution: int) -> List[bytes]:
        """Rasterize each PDF page to PNG bytes using PyMuPDF."""
        if resolution <= 0:
            raise ReaderConfigException("resolution must be a positive integer.")

        pages: list[bytes] = []
        try:
            doc = fitz.open(pdf_path)
            zoom = resolution / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for idx in range(len(doc)):
                try:
                    pix = doc.load_page(idx).get_pixmap(matrix=matrix, alpha=False)
                    pages.append(pix.tobytes("png"))
                except Exception as exc:
                    raise TextractReaderException(
                        f"Failed to rasterize PDF page {idx + 1} from '{pdf_path}': {exc}"
                    ) from exc
            doc.close()
        except TextractReaderException:
            raise
        except Exception as exc:
            raise TextractReaderException(
                f"Failed to open or rasterize PDF '{pdf_path}': {exc}"
            ) from exc

        if not pages:
            raise TextractReaderException(f"No pages found in PDF '{pdf_path}'.")
        return pages

    def _image_to_png_pages(self, file_path: str) -> List[bytes]:
        """Normalize raster or SVG images into one or more PNG page bytes."""
        ext = Path(file_path).suffix.lower().lstrip(".")

        if ext == "svg":
            return self._svg_to_png_pages(file_path)

        pages: list[bytes] = []
        try:
            with Image.open(file_path) as img:
                frame_count = getattr(img, "n_frames", 1)
                for frame_idx in range(frame_count):
                    img.seek(frame_idx)
                    frame = img.convert("RGBA")
                    buf = io.BytesIO()
                    frame.save(buf, format="PNG")
                    pages.append(buf.getvalue())
        except Exception as exc:
            raise TextractReaderException(
                f"Failed to normalize image '{file_path}' to PNG: {exc}"
            ) from exc

        if not pages:
            raise TextractReaderException(f"No image content found in '{file_path}'.")
        return pages

    def _svg_to_png_pages(self, file_path: str) -> List[bytes]:
        """Convert SVG to a temporary PDF and rasterize it to PNG pages."""
        try:
            doc = fitz.open(file_path)
            pdf_bytes = doc.convert_to_pdf()
            doc.close()
        except Exception as exc:
            raise TextractReaderException(
                f"Failed to convert SVG '{file_path}' to PDF: {exc}"
            ) from exc

        with TemporaryDirectory(prefix="textract_svg2pdf_") as tmp_dir:
            pdf_path = os.path.join(tmp_dir, "converted.pdf")
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)
            return self._pdf_to_png_pages(pdf_path, resolution=300)

    def _extract_text_from_pages(
        self,
        page_pngs: Sequence[bytes],
        page_placeholder: str,
    ) -> str:
        """Run Textract on each PNG page and join page text."""
        page_texts: list[str] = []
        for page_idx, png_bytes in enumerate(page_pngs, start=1):
            if len(png_bytes) > TEXTRACT_SYNC_MAX_BYTES:
                raise TextractReaderException(
                    f"Page {page_idx} exceeds Textract synchronous size limit "
                    f"({TEXTRACT_SYNC_MAX_BYTES} bytes)."
                )
            page_text = self._detect_text(png_bytes)
            placeholder = page_placeholder.replace("{page}", str(page_idx))
            page_texts.append(f"{placeholder}\n\n{page_text}".strip())
        return "\n\n".join(page_texts).strip()

    def _detect_text(self, png_bytes: bytes) -> str:
        """Call Textract and extract LINE text in relationship order."""
        try:
            response = self._client.detect_document_text(Document={"Bytes": png_bytes})
        except NoCredentialsError as exc:
            raise ReaderConfigException(
                "AWS credentials were not found. Configure the standard boto3 "
                "credential chain (environment variables, shared config, or IAM role)."
            ) from exc
        except (ClientError, BotoCoreError) as exc:
            raise TextractReaderException(
                f"AWS Textract detect_document_text failed: {exc}"
            ) from exc
        except Exception as exc:
            raise TextractReaderException(
                f"Unexpected Textract client failure: {exc}"
            ) from exc

        return self._blocks_to_text(response.get("Blocks", []))

    @staticmethod
    def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
        """Extract LINE text from PAGE->CHILD relationships without duplicating WORDs."""
        block_map = {block["Id"]: block for block in blocks if "Id" in block}
        lines: list[str] = []

        page_blocks = [block for block in blocks if block.get("BlockType") == "PAGE"]
        for page in page_blocks:
            for rel in page.get("Relationships") or []:
                if rel.get("Type") != "CHILD":
                    continue
                for child_id in rel.get("Ids") or []:
                    child = block_map.get(child_id)
                    if child and child.get("BlockType") == "LINE":
                        text = child.get("Text")
                        if text:
                            lines.append(text)

        if not lines:
            for block in blocks:
                if block.get("BlockType") == "LINE" and block.get("Text"):
                    lines.append(block["Text"])

        return "\n".join(lines).strip()
