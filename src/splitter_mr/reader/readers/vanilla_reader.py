import os
from html.parser import HTMLParser
from typing import Any, Dict

import pandas as pd
import yaml

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

    def read(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Reads a file and returns its raw text content along with standardized metadata.

        Args:
            file_path (str): Path to the input file to be read.
            **kwargs:
                document_id (Optional[str]): Unique identifier for the document.
                conversion_method (Optional[str]): Name or description of the conversion method used.
                    Default is "vanilla".
                ocr_method (Optional[str]): OCR method applied (if any). Default is None.
                metadata (Optional[dict]): Additional document-level metadata as a dictionary.
                    Default is an empty dictionary.

        Returns:
            dict: Dictionary containing:
                - text (str): The raw text content of the file (for Parquet, content as CSV text).
                - document_name (str): The base name of the file.
                - document_path (str): The absolute path to the file.
                - document_id (Optional[str]): Unique identifier for the document.
                - conversion_method (str): The conversion method used ("vanilla").
                - ocr_method (Optional[str]): The OCR method applied (if any).
                - metadata (Optional[dict]): Additional metadata associated with the document.

        Notes:
            - For .json, .html, .txt, .xml, .yaml/.yml, .csv, .tsv: the raw file content is returned
                as a string.
            - For .parquet files, the content is loaded into a pandas DataFrame and returned as
                CSV-formatted text.
            - This method requires the pandas library to be installed for reading .parquet files.
            - For YAML support, ensure PyYAML is installed and imported.
            - If `document_id` is not provided, it will be set to None.
            - If `metadata` is not provided, an empty dictionary will be returned.

        Example:
            ```python
            reader = VanillaReader()
            result = reader.read("/path/to/document.html")
            print(result["text"])
            ```
        """
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        document_name = os.path.basename(file_path)
        document_path = os.path.abspath(file_path)
        metadata = kwargs.get("metadata", {})
        text = ""

        conversion_method = None
        if ext in ("json", "html", "txt", "xml", "csv", "tsv"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == "parquet":
            if pd is None:
                raise ImportError("Pandas must be installed to read parquet files.")
            df = pd.read_parquet(file_path)
            text = df.to_csv(index=False)
            conversion_method = "pandas"
        elif ext == "yaml" or ext == "yml":
            with open(file_path, "r", encoding="utf-8") as f:
                yaml_text = f.read()
            text = yaml.load(yaml_text, Loader=yaml.SafeLoader)
            conversion_method = "json"

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return {
            "text": text,
            "document_name": document_name,
            "document_path": document_path,
            "document_id": kwargs.get("document_id"),
            "conversion_method": conversion_method,
            "ocr_method": None,
            "metadata": metadata,
        }
