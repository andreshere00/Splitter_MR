import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import requests
import yaml

from ...model import BaseVisionModel
from ...schema import (
    DEFAULT_IMAGE_CAPTION_PROMPT,
    DEFAULT_IMAGE_EXTRACTION_PROMPT,
    SUPPORTED_PROGRAMMING_LANGUAGES,
    SUPPORTED_VANILLA_IMAGE_EXTENSIONS,
    ReaderOutput,
)
from ..base_reader import BaseReader
from ..utils import PDFPlumberReader
from ..utils.html_to_markdown import HtmlToMarkdown  # <-- NEW: project converter


class VanillaReader(BaseReader):
    """
    Read multiple file types using Python's built-in and standard libraries.
    Supported: .json, .html/.htm, .txt, .xml, .yaml/.yml, .csv, .tsv, .parquet, .pdf

    **NEW**: HTML handling (local files and URLs):
      - If ``html_to_markdown=True`` (kw arg), HTML is converted to Markdown using the
        project's HtmlToMarkdown utility, and the conversion method is reported as ``"md"``.
      - If ``html_to_markdown=False`` (default), raw HTML is returned without transformation,
        and the conversion method is ``"html"``.

    For PDFs, this reader uses PDFPlumberReader to extract text, tables, and images,
    with options to show or omit images, and to annotate images using a vision model.
    """

    def __init__(self, model: Optional[BaseVisionModel] = None):
        super().__init__()
        self.model = model
        self.pdf_reader = PDFPlumberReader()

    def read(
        self,
        file_path: str | Path = None,
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Read a document from various sources and return standardized output.

        This method supports:
        - Local file paths (``file_path`` or positional arg)
        - URLs (``file_url``)
        - JSON/dict objects (``json_document``)
        - Raw text strings (``text_document``)

        If multiple sources are provided, the priority is:
        ``file_path`` > ``file_url`` > ``json_document`` > ``text_document``.
        If only ``file_path`` is provided, auto-detects whether it is a path, URL,
        JSON, YAML, or plain text.

        Args:
            file_path (str | Path): Path to the input file (overridden by
                ``kwargs['file_path']`` if present).

            **kwargs: Optional arguments that adjust behavior:

                Source selection:
                    file_path (str): Path to the input file (overrides positional arg).
                    file_url (str): HTTPS/HTTP URL to read from.
                    json_document (dict | str): JSON-like document (dict or JSON string).
                    text_document (str): Raw text content.

                Identification/metadata:
                    document_id (str): Explicit document id. Defaults to a new UUID.
                    metadata (dict): Additional metadata to attach to the output.

                HTML handling:
                    html_to_markdown (bool): If True, convert HTML to Markdown before
                        returning. If False (default), return raw HTML as-is.

                PDF extraction:
                    scan_pdf_pages (bool): If True, rasterize and describe pages using a
                        vision model (VLM). If False (default), use element-wise extraction.
                    model (BaseVisionModel): Vision-capable model used for scanned PDFs and/or
                        image captioning (also used for image files).
                    prompt (str): Prompt for image captioning / page description. Defaults to
                        ``DEFAULT_IMAGE_CAPTION_PROMPT`` for element-wise PDFs and
                        ``DEFAULT_IMAGE_EXTRACTION_PROMPT`` for scanned PDFs/images.
                    resolution (int): DPI when rasterizing pages for VLM. Default: 300.
                    show_base64_images (bool): Include base64-embedded images in PDF output.
                        Default: False.
                    image_placeholder (str): Placeholder for omitted images in PDFs.
                        Default: ``"<!-- image -->"``.
                    page_placeholder (str): Placeholder inserted between PDF pages (only
                        surfaced when scanning or when the placeholder occurs in text).
                        Default: ``"<!-- page -->"``.
                    vlm_parameters (dict): Extra keyword args forwarded to
                        ``model.analyze_content(...)``.

                Excel / Parquet reading:
                    as_table (bool): For Excel (``.xlsx``/``.xls``), if True read as a table
                        using pandas and return CSV text. If False (default), convert to PDF
                        and run the PDF pipeline.
                    excel_engine (str): pandas Excel engine. Default: ``"openpyxl"``.
                    parquet_engine (str): pandas Parquet engine (e.g. ``"pyarrow"``,
                        ``"fastparquet"``). Default: pandas auto-selection.

        Returns:
            ReaderOutput: Unified result containing text, metadata, and extraction info.

        Raises:
            ValueError: If the source is invalid/unsupported, or if a VLM is required
                but not provided.
            TypeError: If provided arguments are of unsupported types.

        Notes:
            - HTML control via ``html_to_markdown`` applies to both local files and URLs.
            - For `.parquet` files, content is loaded via pandas and returned as CSV-formatted text.

        Example:
            ```python
            # Convert HTML to Markdown
            reader = VanillaReader()
            md_output = reader.read(file_path="page.html", html_to_markdown=True)

            # Keep raw HTML as-is
            html_output = reader.read(file_path="page.html", html_to_markdown=False)
            ```
        """

        source_type, source_val = _guess_source(kwargs, file_path)
        name, path, text, conv, ocr = self._dispatch_source(
            source_type, source_val, kwargs
        )

        page_ph: str = kwargs.get("page_placeholder", "<!-- page -->")
        page_ph_out = self._surface_page_placeholder(
            scan=bool(kwargs.get("scan_pdf_pages")),
            placeholder=page_ph,
            text=text,
        )

        return ReaderOutput(
            text=_ensure_str(text),
            document_name=name,
            document_path=path or "",
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method=conv,
            reader_method="vanilla",
            ocr_method=ocr,
            page_placeholder=page_ph_out,
            metadata=kwargs.get("metadata", {}),
        )

    def _dispatch_source(  # noqa: WPS231
        self,
        src_type: str,
        src_val: Any,
        kw: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Any, str, Optional[str]]:
        """
        Route the request to a specialised handler and return
        (document_name, document_path, text/content, conversion_method, ocr_method)
        """
        handlers = {
            "file_path": self._handle_local_path,
            "file_url": self._handle_url,
            "json_document": self._handle_explicit_json,
            "text_document": self._handle_explicit_text,
        }
        if src_type not in handlers:
            raise ValueError(f"Unrecognized document source: {src_type}")
        return handlers[src_type](src_val, kw)

    # ---- individual strategies below – each ~20 lines or fewer ---------- #

    # 1) Local / drive paths
    def _handle_local_path(
        self,
        path_like: str | Path,
        kw: Dict[str, Any],
    ) -> Tuple[str, str, Any, str, Optional[str]]:
        """Load from the filesystem (or, if it ‘looks like’ one, via HTTP)."""
        path_str = os.fspath(path_like) if isinstance(path_like, Path) else path_like
        if not isinstance(path_str, str):
            raise ValueError("file_path must be a string or Path object.")

        if not self.is_valid_file_path(path_str):
            if self.is_url(path_str):
                return self._handle_url(path_str, kw)
            return self._handle_fallback(path_str, kw)

        ext = os.path.splitext(path_str)[1].lower().lstrip(".")
        doc_name = os.path.basename(path_str)
        rel_path = os.path.relpath(path_str)

        # ---- type-specific branches ---- #
        # TODO: Refactor to sort the code and make it more readable
        if ext == "pdf":
            return (
                doc_name,
                rel_path,
                *self._process_pdf(path_str, kw),
            )
        if ext == "html" or ext == "htm":
            content, conv = _read_html_file(
                path_str, html_to_markdown=bool(kw.get("html_to_markdown", False))
            )
            return doc_name, rel_path, content, conv, None
        if ext in ("json", "txt", "xml", "csv", "tsv", "md", "markdown"):
            return doc_name, rel_path, _read_text_file(path_str, ext), ext, None
        if ext == "parquet":
            parquet_engine = kw.get(
                "parquet_engine"
            )  # e.g., "pyarrow" or "fastparquet"
            return (
                doc_name,
                rel_path,
                _read_parquet(path_str, engine=parquet_engine),
                "csv",
                None,
            )
        if ext in ("yaml", "yml"):
            return doc_name, rel_path, _read_text_file(path_str, ext), "json", None
        if ext in ("xlsx", "xls"):
            # When as_table=True, pass excel_engine
            if kw.get("as_table", False):
                excel_engine = kw.get("excel_engine", "openpyxl")
                return (
                    doc_name,
                    rel_path,
                    _read_excel(path_str, engine=excel_engine),
                    ext,
                    None,
                )
            # Otherwise convert workbook to PDF and reuse the PDF extractor
            pdf_path = self._convert_office_to_pdf(path_str)
            return (
                os.path.basename(pdf_path),
                os.path.relpath(pdf_path),
                *self._process_pdf(pdf_path, kw),
            )
        if ext in ("docx", "pptx"):
            pdf_path = self._convert_office_to_pdf(path_str)
            return (
                os.path.basename(pdf_path),
                os.path.relpath(pdf_path),
                *self._process_pdf(pdf_path, kw),
            )
        if ext in ("xlsx", "xls"):
            if kw.get("as_table", False):
                # direct spreadsheet → pandas → CSV
                return doc_name, rel_path, _read_excel(path_str), ext, None
            # otherwise convert workbook to PDF and reuse the PDF extractor
            pdf_path = self._convert_office_to_pdf(path_str)
            return (
                os.path.basename(pdf_path),
                os.path.relpath(pdf_path),
                *self._process_pdf(pdf_path, kw),
            )
        if ext in SUPPORTED_VANILLA_IMAGE_EXTENSIONS:
            model = kw.get("model", self.model)
            prompt = kw.get("prompt", DEFAULT_IMAGE_EXTRACTION_PROMPT)
            vlm_parameters = kw.get("vlm_parameters", {})
            return self._handle_image_to_llm(
                model, path_str, prompt=prompt, vlm_parameters=vlm_parameters
            )
        if ext in SUPPORTED_PROGRAMMING_LANGUAGES:
            return doc_name, rel_path, _read_text_file(path_str, ext), "txt", None

        raise ValueError(f"Unsupported file extension: {ext}. Use another Reader.")

    # 2) Remote URL
    def _handle_url(
        self,
        url: str,
        kw: Dict[str, Any],
    ) -> Tuple[str, str, Any, str, Optional[str]]:  # noqa: D401
        """Fetch via HTTP(S)."""
        if not isinstance(url, str) or not self.is_url(url):
            raise ValueError("file_url must be a valid URL string.")
        content, conv = _load_via_requests(
            url, html_to_markdown=bool(kw.get("html_to_markdown", False))
        )
        name = url.split("/")[-1] or "downloaded_file"
        return name, url, content, conv, None

    # 3) Explicit JSON (dict or str)
    def _handle_explicit_json(
        self,
        json_doc: Any,
        _kw: Dict[str, Any],
    ) -> Tuple[str, None, Any, str, None]:
        """JSON passed straight in."""
        return (
            _kw.get("document_name", None),
            None,
            self.parse_json(json_doc),
            "json",
            None,
        )

    # 4) Explicit raw text
    def _handle_explicit_text(
        self,
        txt: str,
        _kw: Dict[str, Any],
    ) -> Tuple[str, None, Any, str, None]:  # noqa: D401
        """Text (maybe JSON / YAML) passed straight in."""
        for parser, conv in ((self.parse_json, "json"), (yaml.safe_load, "json")):
            try:
                parsed = parser(txt)
                if isinstance(parsed, (dict, list)):
                    return _kw.get("document_name", None), None, parsed, conv, None
            except Exception:  # pragma: no cover
                pass
        return _kw.get("document_name", None), None, txt, "txt", None

    # ----- shared utilities ------------------------------------------------ #

    def _process_pdf(
        self,
        path: str,
        kw: Dict[str, Any],
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Process a PDF file and extract content.

        This method supports two modes:
        - Scanned PDF pages using a vision-capable model (image-based extraction).
        - Element-wise text and image extraction using PDFPlumber.

        Args:
            path (str): The path to the PDF file.
            kw (dict): Keyword arguments controlling extraction behavior. Recognized keys include:
                scan_pdf_pages (bool): If True, process the PDF as scanned images.
                model (BaseVisionModel, optional): Vision-capable model for scanned PDFs or image captioning.
                prompt (str, optional): Prompt for image captioning.
                show_base64_images (bool): Whether to include base64 images in the output.
                image_placeholder (str): Placeholder for omitted images.
                page_placeholder (str): Placeholder for page breaks.

        Returns:
            tuple: A tuple of:
                - content (Any): Extracted text/content from the PDF.
                - conv (str): Conversion method used (e.g., "pdf", "png").
                - ocr_method (str or None): OCR model name if applicable.

        Raises:
            ValueError: If `scan_pdf_pages` is True but no vision-capable model is provided.
        """
        if kw.get("scan_pdf_pages"):
            model = kw.get("model", self.model)
            if model is None:
                raise ValueError("scan_pdf_pages=True requires a vision-capable model.")
            joined = self._scan_pdf_pages(path, model=model, **kw)
            return joined, "png", model.model_name
        # element-wise extraction
        content = self.pdf_reader.read(
            path,
            model=kw.get("model", self.model),
            prompt=kw.get("prompt") or DEFAULT_IMAGE_CAPTION_PROMPT,
            show_base64_images=kw.get("show_base64_images", False),
            image_placeholder=kw.get("image_placeholder", "<!-- image -->"),
            page_placeholder=kw.get("page_placeholder", "<!-- page -->"),
        )
        ocr_name = (
            (kw.get("model") or self.model).model_name
            if kw.get("model") or self.model
            else None
        )
        return content, "pdf", ocr_name

    def _scan_pdf_pages(self, file_path: str, model: BaseVisionModel, **kw) -> str:
        """
        Describe each page of a PDF using a vision model.

        Args:
            file_path (str): The path to the PDF file.
            model (BaseVisionModel): Vision-capable model used for page description.
            **kw: Additional keyword arguments. Recognized keys include:
                prompt (str, optional): Prompt for describing PDF pages.
                resolution (int): DPI resolution for rasterizing pages (default: 300).
                vlm_parameters (dict): Extra parameters for the vision model.

        Returns:
            str: A string containing page descriptions separated by page placeholders.
        """
        page_ph = kw.get("page_placeholder", "<!-- page -->")
        pages = self.pdf_reader.describe_pages(
            file_path=file_path,
            model=model,
            prompt=kw.get("prompt") or DEFAULT_IMAGE_EXTRACTION_PROMPT,
            resolution=kw.get("resolution", 300),
            **kw.get("vlm_parameters", {}),
        )
        return "\n\n---\n\n".join(f"{page_ph}\n\n{md}" for md in pages)

    def _handle_fallback(self, raw: str, kw: Dict[str, Any]):
        """
        Handle unsupported or unknown sources.

        Attempts to parse the input as JSON, then as text.
        Falls back to returning the raw content as plain text.

        Args:
            raw (str): Raw string content to be processed.
            kw (dict): Additional keyword arguments, may include:
                document_name (str): Optional name of the document.

        Returns:
            tuple: A tuple of:
                - document_name (str or None)
                - document_path (None)
                - content (Any): Parsed or raw content
                - conversion_method (str)
                - ocr_method (None)
        """
        try:
            return self._handle_explicit_json(raw, kw)
        except Exception:
            try:
                return self._handle_explicit_text(raw, kw)
            except Exception:  # pragma: no cover
                return kw.get("document_name", None), None, raw, "txt", None

    def _handle_image_to_llm(
        self,
        model: BaseVisionModel,
        file_path: str,
        prompt: Optional[str] = None,
        vlm_parameters: Optional[dict] = None,
    ) -> Tuple[str, str, Any, str, str]:
        """
        Extract content from an image file using a vision model.

        Reads the image, encodes it in base64, and sends it to the given vision model
        with the provided prompt.

        Args:
            model (BaseVisionModel): Vision-capable model to process the image.
            file_path (str): Path to the image file.
            prompt (str, optional): Prompt for guiding the vision model.
            vlm_parameters (dict, optional): Additional parameters for the vision model.

        Returns:
            tuple: A tuple of:
                - document_name (str)
                - document_path (str)
                - extracted (Any): Extracted content from the image.
                - conversion_method (str): Always "image".
                - ocr_method (str): Model name.

        Raises:
            ValueError: If no vision model is provided.
        """
        if model is None:
            raise ValueError("No vision model provided for image extraction.")
        # Read image as bytes and encode as base64
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        ext = os.path.splitext(file_path)[1].lstrip(".").lower()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        prompt = prompt or DEFAULT_IMAGE_EXTRACTION_PROMPT
        vlm_parameters = vlm_parameters or {}
        extracted = model.analyze_content(
            img_b64, prompt=prompt, file_ext=ext, **vlm_parameters
        )
        doc_name = os.path.basename(file_path)
        rel_path = os.path.relpath(file_path)
        return doc_name, rel_path, extracted, "image", model.model_name

    @staticmethod
    def _surface_page_placeholder(
        scan: bool, placeholder: str, text: Any
    ) -> Optional[str]:
        """
        Decide whether to expose the page placeholder in output.

        Never exposes placeholders containing '%'. Returns the placeholder if
        scanning mode is enabled or if the placeholder is found in the text.

        Args:
            scan (bool): Whether the document was scanned.
            placeholder (str): Page placeholder string.
            text (Any): Extracted text or content.

        Returns:
            str or None: The placeholder string if it should be exposed, else None.
        """
        if "%" in placeholder:
            return None
        txt = _ensure_str(text)
        return placeholder if (scan or placeholder in txt) else None

    def _convert_office_to_pdf(self, file_path: str) -> str:
        """
        Convert a DOCX/XLSX/PPTX file to PDF using LibreOffice.

        Args:
            file_path: Absolute path to the Office document.

        Returns:
            Path to the generated PDF in a temporary directory.

        Raises:
            RuntimeError: If LibreOffice (``soffice``) is not in *PATH* or the
            conversion fails for any reason.
        """
        if not shutil.which("soffice"):
            raise RuntimeError(
                "LibreOffice/soffice is required for Office-to-PDF conversion "
                "but was not found in PATH.  Install LibreOffice or use a "
                "different reader."
            )

        outdir = tempfile.mkdtemp(prefix="vanilla_office2pdf_")
        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            outdir,
            file_path,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"LibreOffice failed converting {file_path} → PDF:\n{proc.stderr.decode()}"
            )

        pdf_name = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
        pdf_path = os.path.join(outdir, pdf_name)
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"Expected PDF not found: {pdf_path}")

        return pdf_path


# -------- Helpers --------- #


def _ensure_str(val: Any) -> str:
    """
    Convert an arbitrary value to a readable string.

    If ``val`` is a mapping or a sequence (dict/list), try JSON first and then
    YAML to preserve structure and readability. Otherwise, fall back to ``str(val)``.

    Args:
        val: Any Python object.

    Returns:
        str: Human-readable string representation.
    """
    if isinstance(val, (dict, list)):
        for dumper in (
            lambda v: json.dumps(v, indent=2, ensure_ascii=False),
            lambda v: yaml.safe_dump(v, allow_unicode=True),
        ):
            try:
                return dumper(val)
            except Exception:  # pragma: no cover – fall-through
                pass
    return "" if val is None else str(val)


def _guess_source(
    kwargs: Dict[str, Any], file_path: Union[str, Path]
) -> Tuple[str, Any]:
    """
    Determine the input source type and value.

    Checks explicit kwargs in priority order and falls back to the positional
    ``file_path`` if none are provided.

    Args:
        kwargs: Arbitrary keyword arguments that may include:
            - ``file_path``: Local path to a file.
            - ``file_url``: Remote URL to fetch.
            - ``json_document``: Dict or JSON string.
            - ``text_document``: Raw text.
        file_path: Optional positional file path argument.

    Returns:
        Tuple[str, Any]: A pair ``(source_type, value)`` where ``source_type`` is
        one of ``{"file_path", "file_url", "json_document", "text_document"}``
        and ``value`` is the corresponding payload.

    Raises:
        KeyError: Never raised here; unknown keys are ignored. (Validation
            happens later in dispatch.)
    """
    for key in ("file_path", "file_url", "json_document", "text_document"):
        if kwargs.get(key) is not None:
            return key, kwargs[key]
    return "file_path", file_path


def _read_text_file(path: Union[str, Path], ext: str) -> str:
    """
    Read a small text-like file from disk.

    YAML files are parsed into Python objects and then dumped to a string
    to keep output consistent with other structured formats.

    Args:
        path: Path to the file.
        ext: Lowercased file extension without the dot (e.g., ``"txt"``, ``"yaml"``).

    Returns:
        str: File contents (or a YAML-dumped string for YAML files).
    """
    with open(path, "r", encoding="utf-8") as fh:
        return (
            fh.read()
            if ext not in ("yaml", "yml")
            else yaml.safe_dump(yaml.safe_load(fh), allow_unicode=True)
        )


def _read_html_file(
    path: Union[str, Path], *, html_to_markdown: bool
) -> Tuple[str, str]:
    """
    Read an HTML file from disk, optionally converting to Markdown.

    Args:
        path: Path to the HTML file.
        html_to_markdown: If True, convert to Markdown; else return raw HTML.

    Returns:
        Tuple[str, str]: (content, conversion_method) where conversion_method is
        "md" if converted, otherwise "html".
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    if html_to_markdown:
        md = HtmlToMarkdown().convert(raw)
        return md, "md"
    return raw, "html"


def _read_parquet(path: Union[str, Path], *, engine: Optional[str] = None) -> str:
    """Read a Parquet file and return CSV-formatted text.

    Args:
        path: Path to the Parquet file.
        engine: Parquet engine to use. If ``None`` (default), pandas will auto-select
            an available engine (commonly ``"pyarrow"`` or ``"fastparquet"``).
            Provide a specific engine name to force one, e.g., ``"pyarrow"``.

    Returns:
        str: CSV string (header included, index excluded).

    Raises:
        ImportError: If the requested engine is not available.
        ValueError: If the file is malformed or unreadable.
    """
    if engine is None:
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path, engine=engine)
    return df.to_csv(index=False)


def _read_excel(path: Union[str, Path], *, engine: str = "openpyxl") -> str:
    """
    Read an Excel workbook and return CSV-formatted text of the first sheet.

    Args:
        path: Path to the Excel file.
        engine: Pandas Excel engine. Defaults to ``"openpyxl"`` for xlsx/xls support.

    Returns:
        str: CSV string (header included, index excluded).

    Raises:
        ImportError: If the requested engine is not installed.
        ValueError: If the file is malformed or unreadable.
    """
    df = pd.read_excel(path, engine=engine)
    return df.to_csv(index=False)


def _load_via_requests(url: str, *, html_to_markdown: bool = False) -> Tuple[Any, str]:
    """Fetch content via HTTP(S) and return a (payload, type) pair.

    The ``type`` loosely reflects a "conversion key" used elsewhere to decide
    how the content should be treated downstream.

    Behavior for content types:
      - application/json or ``*.json``: parsed JSON → (obj, "json")
      - text/html or ``*.html``/``*.htm``:
            * if ``html_to_markdown=True``: converted Markdown → (str, "md")
            * else: raw HTML text → (str, "html")
      - text/yaml or ``*.yaml``/``*.yml``: parsed YAML → (obj, "json")
      - Otherwise: raw text → (str, "txt")

    Args:
        url: Fully qualified HTTP/HTTPS URL.
        html_to_markdown: If True, convert HTML responses to Markdown.

    Returns:
        Tuple[Any, str]:
            - payload: Parsed JSON/YAML, Markdown (converted), raw HTML, or raw text.
            - conv: One of ``{"json", "md", "html", "txt"}``.

    Raises:
        requests.HTTPError: If the HTTP request fails (non-2xx).
    """
    resp = requests.get(url)
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type", "") or "").lower()

    # JSON
    if "application/json" in ctype or url.endswith(".json"):
        return resp.json(), "json"

    # HTML
    if "text/html" in ctype or url.endswith((".html", ".htm")):
        raw_html = resp.text
        if html_to_markdown:
            md = HtmlToMarkdown().convert(raw_html)
            return md, "md"
        return raw_html, "html"

    # YAML
    if "text/yaml" in ctype or url.endswith((".yaml", ".yml")):
        return yaml.safe_load(resp.text), "json"

    # covers csv & plain text and many other text/* types
    return resp.text, "txt"


class SimpleHTMLTextExtractor(HTMLParser):
    """Extract text from HTML by concatenating text nodes (legacy helper)."""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts).strip()
