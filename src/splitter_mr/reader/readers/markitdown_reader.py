import io
import os
import uuid
from typing import Any, Optional, Union

import fitz
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
    """

    def __init__(
        self, model: Optional[Union[AzureOpenAIVisionModel, OpenAIVisionModel]] = None
    ):
        self.model = model
        self.model_name = model.model_name if self.model else None

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
            result = reader.read(file_path = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/lorem_ipsum.pdf")
            print(result.text)
            ```
            ```python
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """

        # Helper
        def pdf_pages_to_streams(pdf_path: str) -> list[io.BytesIO]:
            """Render each PDF page to PNG bytes wrapped in BytesIO."""
            doc = fitz.open(pdf_path)
            streams: list[io.BytesIO] = []
            for idx in range(len(doc)):
                pix = doc.load_page(idx).get_pixmap()
                buf = io.BytesIO(pix.tobytes("png"))
                # give MarkItDown a hint about the mime-type / extension
                buf.name = f"page_{idx + 1}.png"
                buf.seek(0)
                streams.append(buf)
            return streams

        # Initialise Mark-It-Down reader
        if self.model is not None:
            if not isinstance(self.model, (OpenAIVisionModel, AzureOpenAIVisionModel)):
                raise ValueError(
                    "Incompatible client. Only AzureOpenAIVisionModel or "
                    "OpenAIVisionModel are supported."
                )
            client = self.model.get_client()
            md = MarkItDown(llm_client=client, llm_model=self.model.model_name)
            ocr_method = self.model.model_name
        else:
            md = MarkItDown()
            ocr_method = None

        scan_pdf_pages: bool = kwargs.get("scan_pdf_pages", False)
        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        conversion_method: str | None = None

        #  Handle page-by-page OCR for PDFs
        if scan_pdf_pages:
            if ext != "pdf":
                raise ValueError("scan_pdf_pages=True requires a PDF file.")
            if self.model is None:
                raise ValueError("scan_pdf_pages=True requires a VisionModel.")

            prompt = kwargs.get("prompt") or (
                "Extract all elements detected in the page, in order. "
                "Return only the extracted content, in valid Markdown."
            )
            page_md: list[str] = []
            for idx, page_stream in enumerate(pdf_pages_to_streams(file_path), start=1):
                page_md.append(f"<!-- page {idx} -->\n")
                result = md.convert(page_stream, llm_prompt=prompt)
                page_md.append(result.text_content)

            markdown_text = "\n\n".join(page_md)
            conversion_method = "markdown"

        # Regular conversion path
        else:
            markdown_text = md.convert(file_path).text_content
            conversion_method = "json" if ext == "json" else "markdown"

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id") or str(uuid.uuid4()),
            conversion_method=conversion_method,
            reader_method="markitdown",
            ocr_method=ocr_method,
            metadata=kwargs.get("metadata"),
        )
