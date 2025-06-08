import os
import uuid

from markitdown import MarkItDown

from ...schema import ReaderOutput
from ..base_reader import BaseReader


class MarkItDownReader(BaseReader):

    # TODO: Introduce a __init__ method, if needed

    def read(self, file_path: str, **kwargs) -> ReaderOutput:
        """
        Reads a file and converts its contents to Markdown using MarkItDown, returning
        structured metadata.

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

        Notes:
            - This method uses [MarkItDown](https://github.com/microsoft/markitdown) to convert
                a wide variety of file formats (e.g., PDF, DOCX, images, HTML, CSV) to Markdown.
            - If `document_id` is not provided, a UUID will be automatically assigned.
            - If `metadata` is not provided, an empty list will be used.
            - MarkItDown should be installed with all relevant optional dependencies for full
                file format support.

        Example:
            ```python
            from splitter_mr.readers import MarkItDownReader

            reader = MarkItDownReader()
            result = reader.read(file_path = "data/test_1.pdf")
            print(result.text)
            ```
            ```bash
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """

        # Read using Docling
        md = MarkItDown()
        markdown_text = md.convert(file_path).text_content
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        conversion_method = "json" if ext == "json" else "markdown"

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id") or str(uuid.uuid4()),
            conversion_method=conversion_method,
            reader_method="markitdown",
            ocr_method=kwargs.get("ocr_method"),
            metadata=kwargs.get("metadata"),
        )
