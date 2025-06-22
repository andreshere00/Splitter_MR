import os
import uuid
from typing import Any, Optional, Union

from markitdown import MarkItDown

from ...model import AzureOpenAIVisionModel, OpenAIVisionModel
from ...schema import ReaderOutput
from ..base_reader import BaseReader


class MarkItDownReader(BaseReader):
    """
    Read multiple file types using Microsoft's MarkItDown library, and convert
    the documents using markdown format.

    This reader supports both standard MarkItDown conversion and the use of Vision Language Models (VLMs)
    for LLM-based OCR when extracting text from images or scanned documents.

    Currently, only the following VLMs are supported:
        - OpenAIVisionModel
        - AzureOpenAIVisionModel

    If a compatible model is provided, MarkItDown will leverage the specified VLM for OCR, and the
    model's name will be recorded as the OCR method used.

    Notes:
        - This method uses [MarkItDown](https://github.com/microsoft/markitdown) to convert
            a wide variety of file formats (e.g., PDF, DOCX, images, HTML, CSV) to Markdown.
        - If `document_id` is not provided, a UUID will be automatically assigned.
        - If `metadata` is not provided, an empty list will be used.
        - MarkItDown should be installed with all relevant optional dependencies for full
            file format support.
    """

    def __init__(
        self, model: Optional[Union[AzureOpenAIVisionModel, OpenAIVisionModel]] = None
    ):
        self.model = model
        self.model_name = None

        if model is not None:
            if not isinstance(model, (OpenAIVisionModel, AzureOpenAIVisionModel)):
                raise ValueError(
                    "Incompatible client. Only AzureOpenAIVisionModel and OpenAIVisionModel are supported."
                )
            client = model.get_client()
            self.model_name = self.model.model_name
            self.md = MarkItDown(llm_client=client, llm_model=self.model_name)
        else:
            self.md = MarkItDown()

    def read(self, file_path: str, **kwargs: Any) -> ReaderOutput:
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
            ReaderOutput: Dataclass defining the output structure for all readers.

        Example:
            ```python
            from splitter_mr.reader import MarkItDownReader
            from splitter_mr.model import OpenAIVisionModel # Or AzureOpenAIVisionModel

            openai = OpenAIVisionModel() # make sure to have necessary environment variables on `.env`.

            reader = MarkItDownReader(model = openai)
            result = reader.read(file_path = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.pdf")
            print(result.text)
            ```
            ```python
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """
        # Read using MarkItDown
        markdown_text = self.md.convert(file_path).text_content
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
            ocr_method=self.model_name,
            metadata=kwargs.get("metadata"),
        )
