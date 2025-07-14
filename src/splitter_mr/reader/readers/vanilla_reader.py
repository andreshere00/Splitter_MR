import json
import os
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        def _ensure_str(val):
            if isinstance(val, (dict, list)):
                try:
                    return json.dumps(val, indent=2, ensure_ascii=False)
                except Exception:
                    try:
                        return yaml.safe_dump(val, allow_unicode=True)
                    except Exception:
                        return str(val)
            if val is None:
                return ""
            return str(val)

        SOURCE_PRIORITY = [
            "file_path",
            "file_url",
            "json_document",
            "text_document",
        ]

        # Pick the highest-priority source provided
        document_source = None
        source_type = None
        for key in SOURCE_PRIORITY:
            if key in kwargs and kwargs[key] is not None:
                document_source = kwargs[key]
                source_type = key
                break

        if document_source is None:
            document_source = file_path
            source_type = "file_path"

        document_name = kwargs.get("document_name", None)
        document_path = None
        conversion_method = None
        ocr_method = None
        text = ""

        # File path or default
        if source_type == "file_path":
            if isinstance(document_source, Path):
                document_source = os.fspath(document_source)
            if not isinstance(document_source, str):
                raise ValueError("file_path must be a string or Path object.")

            if self.is_valid_file_path(document_source):
                ext = os.path.splitext(document_source)[-1].lower().lstrip(".")
                document_name = os.path.basename(document_source)
                document_path = os.path.relpath(document_source)

                if ext == "pdf":
                    if kwargs.get("scan_pdf_pages"):
                        # Vision-powered full-page description
                        model: Optional[BaseModel] = kwargs.get("model", self.model)
                        if model is None:
                            raise ValueError(
                                "scan_pdf_pages=True requires a vision-capable "
                                "`model` implementing BaseModel."
                            )
                        resolution = kwargs.get("resolution", 300)
                        vlm_parameters: Dict[str, Any] = kwargs.get(
                            "vlm_parameters", {}
                        )
                        page_placeholder = kwargs.get(
                            "page_placeholder", "<!-- page -->"
                        )

                        page_markdowns: List[str] = self.pdf_reader.describe_pages(
                            file_path=document_source,
                            model=model,
                            prompt=kwargs.get("prompt") or DEFAULT_EXTRACTION_PROMPT,
                            resolution=resolution,
                            **vlm_parameters,
                        )

                        # Join pages under clear headings
                        joined_pages = []
                        for _, md in enumerate(page_markdowns, start=1):
                            joined_pages.append(f"{page_placeholder}\n\n{md}")
                        text = "\n\n---\n\n".join(joined_pages)

                        conversion_method = "png"
                        ocr_method = model.model_name

                    else:
                        # Element-wised PDF extraction
                        pdf_reader = self.pdf_reader
                        model = kwargs.get("model", self.model)

                        image_placeholder = kwargs.get(
                            "image_placeholder", "<!-- image -->"
                        )
                        page_placeholder = kwargs.get(
                            "page_placeholder", "<!-- page -->"
                        )

                        if model is not None:
                            text = pdf_reader.read(
                                document_source,
                                model=model,
                                prompt=kwargs.get("prompt")
                                or DEFAULT_IMAGE_CAPTION_PROMPT,  # noqa: W503
                                show_base64_images=kwargs.get(
                                    "show_base64_images", False
                                ),
                                image_placeholder=image_placeholder,
                                page_placeholder=page_placeholder,
                            )
                            ocr_method = model.model_name
                        else:
                            text = pdf_reader.read(
                                document_source,
                                show_base64_images=kwargs.get(
                                    "show_base64_images", False
                                ),
                                image_placeholder=image_placeholder,
                                page_placeholder=page_placeholder,
                            )
                        conversion_method = "pdf"

                elif ext in (
                    "json",
                    "html",
                    "txt",
                    "xml",
                    "csv",
                    "tsv",
                    "md",
                    "markdown",
                ):
                    with open(document_source, "r", encoding="utf-8") as f:
                        text = f.read()
                    conversion_method = ext
                elif ext == "parquet":
                    df = pd.read_parquet(document_source)
                    text = df.to_csv(index=False)
                    conversion_method = "csv"
                elif ext in ("yaml", "yml"):
                    with open(document_source, "r", encoding="utf-8") as f:
                        yaml_text = f.read()
                    text = yaml.safe_load(yaml_text)
                    conversion_method = "json"
                elif ext in ("xlsx", "xls"):
                    text = str(
                        pd.read_excel(document_source, engine="openpyxl").to_csv()
                    )
                    conversion_method = ext
                elif ext in SUPPORTED_PROGRAMMING_LANGUAGES:
                    with open(document_source, "r", encoding="utf-8") as f:
                        text = f.read()
                    conversion_method = "txt"
                else:
                    raise ValueError(
                        f"Unsupported file extension: {ext}. Use another Reader component."
                    )

            # (2) URL
            elif self.is_url(document_source):
                ext = os.path.splitext(document_source)[-1].lower().lstrip(".")
                response = requests.get(document_source)
                response.raise_for_status()
                document_name = document_source.split("/")[-1] or "downloaded_file"
                document_path = document_source
                conversion_method = ext
                content_type = response.headers.get("Content-Type", "")

                if "application/json" in content_type or document_name.endswith(
                    ".json"
                ):
                    text = response.json()
                elif "text/html" in content_type or document_name.endswith(".html"):
                    parser = SimpleHTMLTextExtractor()
                    parser.feed(response.text)
                    text = parser.get_text()
                elif "text/yaml" in content_type or document_name.endswith(
                    (".yaml", ".yml")
                ):
                    text = yaml.safe_load(response.text)
                    conversion_method = "json"
                elif "text/csv" in content_type or document_name.endswith(".csv"):
                    text = response.text
                else:
                    text = response.text

            # (3) JSON/dict string
            else:
                try:
                    text = self.parse_json(document_source)
                    conversion_method = "json"
                except Exception:
                    try:
                        text = yaml.safe_load(document_source)
                        conversion_method = "json"
                    except Exception:
                        text = document_source
                        conversion_method = "txt"

        # --- 2. Explicit URL
        elif source_type == "file_url":
            ext = os.path.splitext(document_source)[-1].lower().lstrip(".")
            if not isinstance(document_source, str) or not self.is_url(document_source):
                raise ValueError("file_url must be a valid URL string.")
            response = requests.get(document_source)
            response.raise_for_status()
            document_name = document_source.split("/")[-1] or "downloaded_file"
            document_path = document_source
            conversion_method = ext
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type or document_name.endswith(".json"):
                text = response.json()
            elif "text/html" in content_type or document_name.endswith(".html"):
                parser = SimpleHTMLTextExtractor()
                parser.feed(response.text)
                text = parser.get_text()
            elif "text/yaml" in content_type or document_name.endswith(
                (".yaml", ".yml")
            ):
                text = yaml.safe_load(response.text)
                conversion_method = "json"
            elif "text/csv" in content_type or document_name.endswith(".csv"):
                text = response.text
            else:
                text = response.text

        # --- 3. Explicit JSON
        elif source_type == "json_document":
            document_name = kwargs.get("document_name", None)
            document_path = None
            text = self.parse_json(document_source)
            conversion_method = "json"

        # --- 4. Explicit text
        elif source_type == "text_document":
            document_name = kwargs.get("document_name", None)
            document_path = None
            try:
                parsed = self.parse_json(document_source)
                # Only treat as JSON if result is dict or list, not a string!
                if isinstance(parsed, (dict, list)):
                    text = parsed
                    conversion_method = "json"
                else:
                    raise ValueError  # Force fallback
            except Exception:
                try:
                    parsed = yaml.safe_load(document_source)
                    # Only treat as YAML if it returns a dict or list
                    if isinstance(parsed, (dict, list)):
                        text = parsed
                        conversion_method = "json"
                    else:
                        raise ValueError
                except Exception:
                    text = document_source
                    conversion_method = "txt"

        else:
            raise ValueError(f"Unrecognized document source: {source_type}")

        metadata = kwargs.get("metadata") or {}
        document_id = kwargs.get("document_id", str(uuid.uuid4()))
        document_path = document_path or ""

        return ReaderOutput(
            text=_ensure_str(text),
            document_name=document_name,
            document_path=document_path,
            document_id=document_id,
            conversion_method=conversion_method,
            reader_method="vanilla",
            ocr_method=ocr_method,
            metadata=metadata,
        )


class SimpleHTMLTextExtractor(HTMLParser):
    """Extract HTML Structures from a text"""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts).strip()
