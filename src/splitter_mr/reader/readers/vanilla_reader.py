import json
import os
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import yaml

from ...model import BaseModel
from ...schema import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_IMAGE_CAPTION_PROMPT,
    SUPPORTED_PROGRAMMING_LANGUAGES,
    ReaderOutput,
)
from ..base_reader import BaseReader
from ..utils import PDFPlumberReader


class VanillaReader(BaseReader):
    """
    Read multiple file types using Python's built-in and standard libraries.
    Supported: .json, .html, .txt, .xml, .yaml/.yml, .csv, .tsv, .parquet, .pdf

    For PDFs, this reader uses PDFPlumberReader to extract text, tables, and images,
    with options to show or omit images, and to annotate images using a vision model.
    """

    def __init__(self, model: Optional[BaseModel] = None):
        super().__init__()
        self.model = model
        self.pdf_reader = PDFPlumberReader()

    def read(
        self,
        file_path: str | Path = None,
        **kwargs: Any,
    ) -> ReaderOutput:
        """
        Reads a document from various sources and returns its text content along with standardized metadata.

        This method supports reading from:
            - Local file paths (file_path, or as a positional argument)
            - URLs (file_url)
            - JSON/dict objects (json_document)
            - Raw text strings (text_document)
        If multiple sources are provided, the following priority is used: file_path, file_url,
        json_document, text_document.
        If only file_path is provided, the method will attempt to automatically detect if the value is
        a path, URL, JSON, YAML, or plain text.

        Args:
            file_path (str | Path): Path to the input file.
            **kwargs:
                - `file_path (str, optional)`: Path to the input file (overrides positional argument).
                - `prompt (str, optional)`: Custom prompt for image captioning.
                - `show_base64_images (Optional[bool])`: If True (default), images in PDFs are shown inline as base64 PNG.
                    If False, images are omitted (or annotated if a model is provided).
                - `scan_pdf_pages (bool)`: If *True* and the source is a PDF, read the PDF by pages as scanned images.
                - `file_url (str, optional)`: URL to read the document from.
                - `json_document (dict or str, optional)`: Dictionary or JSON string containing document content.
                - `text_document (str, optional)`: Raw text or string content of the document.
                - `document_id (Optional[str])`: Unique document identifier. If not provided, an UUID will be generated.
                - `metadata (Optional[Dict[str, Any]])`: Additional metadata, given in dictionary format.
                    If not provided, no metadata is returned.
                - `vlm_parameters (Optional[Dict[str, Any]])`:
                    Extra kwargs forwarded verbatim to `model.extract_text`.
                - `resolution (Optional[int])`: DPI used when rasterising PDF pages for vision models. Default is 300.
                - `image_placeholder (Optional[str])`: Placeholder string to use for omitted images in PDFs. Default is `"<!-- image -->"`.
                - `page_placeholder (Optional[str])`: Placeholder string for PDF page breaks. Default is `"<!-- page -->"`.

        Returns:
            ReaderOutput: Dataclass defining the output structure for all readers.

        Raises:
            ValueError: If the provided source is not valid or supported, or if file/URL/JSON detection fails.
            TypeError: If provided arguments are of unsupported types.

        Notes:
            - PDF extraction now supports image captioning/omission indicators.
            - For `.parquet` files, content is loaded via pandas and returned as CSV-formatted text.

        Example:
            ```python
            from splitter_mr.readers import VanillaReader
            from splitter_mr.models import AzureOpenAIVisionModel

            model = AzureOpenAIVisionModel()
            reader = VanillaReader(model=model)
            output = reader.read(file_path="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/lorem_ipsum.pdf")
            print(output.text)
            ```
            ```bash
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
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

    # ---- individual strategies below – each ~20 lines or fewer ---------- #

    # 1) Local / drive paths
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

        # ---- type‑specific branches ---- #
        if ext == "pdf":
            return (
                doc_name,
                rel_path,
                *self._process_pdf(path_str, kw),
            )
        if ext in ("json", "html", "txt", "xml", "csv", "tsv", "md", "markdown"):
            return doc_name, rel_path, _read_text_file(path_str, ext), ext, None
        if ext == "parquet":
            return doc_name, rel_path, _read_parquet(path_str), "csv", None
        if ext in ("yaml", "yml"):
            return doc_name, rel_path, _read_text_file(path_str, ext), "json", None
        if ext in ("xlsx", "xls"):
            return doc_name, rel_path, _read_excel(path_str), ext, None
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
        content, conv = _load_via_requests(url)
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
        """Handle the two PDF modes, returning (content, conv, ocr_method)."""
        if kw.get("scan_pdf_pages"):
            model = kw.get("model", self.model)
            if model is None:
                raise ValueError("scan_pdf_pages=True requires a vision‑capable model.")
            joined = self._scan_pdf_pages(path, model=model, **kw)
            return joined, "png", model.model_name
        # element‑wise extraction
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

    def _scan_pdf_pages(self, file_path: str, model: BaseModel, **kw) -> str:
        """VLM: describe each page then join."""
        page_ph = kw.get("page_placeholder", "<!-- page -->")
        pages = self.pdf_reader.describe_pages(
            file_path=file_path,
            model=model,
            prompt=kw.get("prompt") or DEFAULT_EXTRACTION_PROMPT,
            resolution=kw.get("resolution", 300),
            **kw.get("vlm_parameters", {}),
        )
        return "\n\n---\n\n".join(f"{page_ph}\n\n{md}" for md in pages)

    def _handle_fallback(self, raw: str, kw: Dict[str, Any]):
        """
        Re‑use the logic from the original ‘else’ branch when *raw* is neither
        a valid path nor a recognised URL.
        """
        try:
            return self._handle_explicit_json(raw, kw)
        except Exception:
            try:
                return self._handle_explicit_text(raw, kw)
            except Exception:  # pragma: no cover
                return kw.get("document_name", None), None, raw, "txt", None

    @staticmethod
    def _surface_page_placeholder(
        scan: bool, placeholder: str, text: Any
    ) -> Optional[str]:
        """
        Decide whether to expose the page placeholder in `ReaderOutput`.
        Follows the rule introduced in the latest patch: never expose
        placeholders that contain '%'.
        """
        if "%" in placeholder:
            return None
        txt = _ensure_str(text)
        return placeholder if (scan or placeholder in txt) else None


# Helpers


def _ensure_str(val: Any) -> str:
    """Convert *val* (possibly a dict / list) to a readable string."""
    if isinstance(val, (dict, list)):
        for dumper in (
            lambda v: json.dumps(v, indent=2, ensure_ascii=False),
            lambda v: yaml.safe_dump(v, allow_unicode=True),
        ):
            try:
                return dumper(val)
            except Exception:  # pragma: no cover – fall‑through
                pass
    return "" if val is None else str(val)


def _guess_source(
    kwargs: Dict[str, Any], file_path: str | Path | None
) -> Tuple[str, Any]:
    """
    Decide where the content comes from based on the precedence defined in the
    original implementation.
    """
    for key in ("file_path", "file_url", "json_document", "text_document"):
        if kwargs.get(key) is not None:
            return key, kwargs[key]
    return "file_path", file_path


def _read_text_file(path: str, ext: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read() if ext != "yaml" and ext != "yml" else yaml.safe_load(fh)


def _read_parquet(path: str) -> str:
    return pd.read_parquet(path).to_csv(index=False)


def _read_excel(path: str) -> str:
    return pd.read_excel(path, engine="openpyxl").to_csv(index=False)


def _load_via_requests(url: str) -> Tuple[Any, str]:
    """Return content, mime‐like conversion key."""
    resp = requests.get(url)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "application/json" in ctype or url.endswith(".json"):
        return resp.json(), "json"
    if "text/html" in ctype or url.endswith(".html"):
        p = SimpleHTMLTextExtractor()
        p.feed(resp.text)
        return p.get_text(), "html"
    if "text/yaml" in ctype or url.endswith((".yaml", ".yml")):
        return yaml.safe_load(resp.text), "json"
    return resp.text, "txt"  # covers csv & plain text


class SimpleHTMLTextExtractor(HTMLParser):
    """Extract HTML Structures from a text"""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts).strip()
