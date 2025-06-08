import os
import uuid
from html.parser import HTMLParser
from typing import Any

import pandas as pd
import requests
import yaml

from ...schema import ReaderOutput
from ..base_reader import BaseReader


class SimpleHTMLTextExtractor(HTMLParser):
    """Extract HTML Structures from a text"""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts).strip()


class VanillaReader(BaseReader):
    """
    Read multiple file types using Python's built-in and standard libraries.
    Supported: .json, .html, .txt, .xml, .yaml/.yml, .csv, .tsv, .parquet
    """

    def read(self, file_path: Any = None, **kwargs) -> ReaderOutput:
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
            file_path (str, optional): Path to the input file.
            **kwargs:
                file_path (str, optional): Path to the input file (overrides positional argument).
                file_url (str, optional): URL to read the document from.
                json_document (dict or str, optional): Dictionary or JSON string containing document content.
                text_document (str, optional): Raw text or string content of the document.

        Returns:
            dict: Dictionary with the following keys:
                - text (str): The extracted or loaded text content.
                - document_name (str or None): The document's name (base filename, URL, or user-provided).
                - document_path (str or None): The path or URL of the document, if available.
                - document_id (str): Unique identifier for the document.
                - conversion_method (str): Method or strategy used to read/convert the document.
                - ocr_method (str or None): The OCR method applied (if any).
                - metadata (dict): Additional document-level metadata.

        Raises:
            ValueError: If the provided source is not valid or supported, or if file/URL/JSON detection fails.
            TypeError: If provided arguments are of unsupported types.

        Notes:
            - If reading from a file, supported formats include:
                .json, .html, .txt, .xml, .yaml/.yml, .csv, .tsv, .parquet.
            - For `.parquet` files, content is loaded via pandas and returned as CSV-formatted text.
            - For URLs, content type and extension are auto-detected to determine parsing strategy.
            - If a JSON or YAML string is provided, it will be parsed accordingly; otherwise,
                the input is treated as plain text.
            - If `document_id` is not provided, a UUID will be generated.
            - If `metadata` is not provided, an empty dictionary will be returned.

        Example:
            ```python
            from splitter_mr.readers import VanillaReader

            reader = VanillaReader()

            result = reader.read(file_path="data/sample.txt") # Read from file path
            result = reader.read(file_url="https://example.com/sample.txt") # Read from URL
            result = reader.read(json_document={"text": "Hello, world!"}) # Read from dict
            result = reader.read(text_document="Hello, world!") # Read from text

            print(result.text)
            ```
            ```bash
            Hello, world!
            ```
        """

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

        document_name = kwargs.get("document_name")
        document_path = None
        conversion_method = None

        # --- 1. File path or default
        if source_type == "file_path":
            if not isinstance(document_source, str):
                raise ValueError("file_path must be a string.")

            if self.is_valid_file_path(document_source):
                ext = os.path.splitext(document_source)[-1].lower().lstrip(".")
                document_name = os.path.basename(document_source)
                document_path = os.path.relpath(document_source)

                if ext in (
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

        metadata = kwargs.get("metadata", {})
        document_id = kwargs.get("document_id") or str(uuid.uuid4())
        ocr_method = kwargs.get("ocr_method")

        return ReaderOutput(
            text=text,
            document_name=document_name,
            document_path=document_path,
            document_id=document_id,
            conversion_method=conversion_method,
            reader_method="vanilla",
            ocr_method=ocr_method,
            metadata=metadata,
        )
