import os
import shutil
import uuid

from docling.document_converter import DocumentConverter

from ...schema.schemas import ReaderOutput
from ..base_reader import BaseReader


class DoclingReader(BaseReader):
    # TODO: Introduce a __init__ method, if needed
    def read(self, file_path: str, **kwargs) -> dict:
        """
        Reads and converts a document to Markdown format using the
        [Docling](https://github.com/docling-project/docling) library, supporting a wide range
        of file types including PDF, DOCX, HTML, and images.

        This method leverages Docling's advanced document parsing capabilities—including layout
        and table detection, code and formula extraction, and integrated OCR—to produce clean,
        markdown-formatted output for downstream processing. The output includes standardized
        metadata and can be easily integrated into generative AI or information retrieval pipelines.

        Args:
            file_path (str): Path to the input file to be read and converted.
            **kwargs:
                document_id (Optional[str]): Unique document identifier.
                    If not provided, a UUID will be generated.
                conversion_method (Optional[str]): Name or description of the
                    conversion method used. Default is None.
                ocr_method (Optional[str]): OCR method applied (if any).
                    Default is None.
                metadata (Optional[List[str]]): Additional metadata as a list of strings.
                    Default is an empty list.

        Returns:
            dict: Dictionary containing:
                - text (str): The Markdown-formatted text content of the file.
                - document_name (str): The base name of the file.
                - document_path (str): The absolute path to the file.
                - document_id (str): Unique identifier for the document.
                - conversion_method (Optional[str]): The conversion method used.
                - ocr_method (Optional[str]): The OCR method applied (if any).
                - metadata (Optionaal[dict]): Additional metadata associated with the document.

        Example:
            ```python
            from splitter_mr.readers import DoclingReader

            reader = DoclingReader()
            result = reader.read(file_path = "data/test_1.pdf")
            print(result["text"])
            ```
            ```bash
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """
        conversion_method = "markdown"
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")

        # In case that the extension is not compatible, convert to markdown directly
        if ext in ("txt", "json"):
            file_md = str(os.path.splitext(file_path)[0]) + ".md"
            shutil.copyfile(file_path, file_md)
            file_path = file_md
            if ext == "json":
                conversion_method = "json"

        # Read using Docling
        reader = DocumentConverter()
        markdown_text = reader.convert(file_path).document.export_to_markdown()

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id") or str(uuid.uuid4()),
            conversion_method=conversion_method,
            reader_method="docling",
            ocr_method=kwargs.get("ocr_method"),
            metadata=kwargs.get("metadata"),
        ).to_dict()
