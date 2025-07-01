import io
import os
import uuid
from pathlib import Path
from typing import Any, Optional, Union

import fitz
from markitdown import MarkItDown

from ...model import AzureOpenAIVisionModel, OpenAIVisionModel
from ...schema import DEFAULT_EXTRACTION_PROMPT, ReaderOutput
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

    def read(self, file_path: Path | str = None, **kwargs: Any) -> ReaderOutput:
        """
        Reads a file and converts its contents to Markdown using MarkItDown.

        Features:
            - Standard file-to-Markdown conversion for most formats.
            - LLM-based OCR (if a Vision model is provided) for images and scanned PDFs.
            - Optional PDF page-wise OCR with fine-grained control and custom LLM prompt.

        Args:
            file_path (str): Path to the input file to be read and converted.
            **kwargs:
                - `document_id (Optional[str])`: Unique document identifier.
                    If not provided, a UUID will be generated.
                - `metadata (Dict[str, Any], optional)`: Additional metadata, given in dictionary format.
                    If not provided, no metadata is returned.
                - `prompt (Optional[str])`: Prompt for image captioning or VLM extraction.
                - `scan_pdf_pages (Optional[bool])`: If True (and model provided), extract PDFs page-by-page with VLM.

        Returns:
            ReaderOutput: Dataclass defining the output structure for all readers.

        Example:
            ```python
            from splitter_mr.model import OpenAIVisionModel
            from splitter_mr.reader import MarkItDownReader

            model = AzureOpenAIVisionModel()
            reader = MarkItDownReader(model=model)
            output = reader.read(file_path="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/lorem_ipsum.pdf")
            print(output.text)
            ```
            ```python
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec eget purus non est porta
            rutrum. Suspendisse euismod lectus laoreet sem pellentesque egestas et et sem.
            Pellentesque ex felis, cursus ege...
            ```
        """

        # Helpers

        def _pdf_pages_to_streams(pdf_path: str) -> list[io.BytesIO]:
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

        def _pdf_pages_to_markdown(
            file_path, prompt: str = DEFAULT_EXTRACTION_PROMPT
        ) -> str:
            """Convert scanned PDF pages into markdown format"""
            page_md: list[str] = []
            for idx, page_stream in enumerate(
                _pdf_pages_to_streams(file_path), start=1
            ):
                page_md.append(f"<!-- page {idx} -->")
                result = md.convert(page_stream, llm_prompt=prompt)
                page_md.append(result.text_content)
            markdown_text = "\n".join(page_md)
            return markdown_text

        # Initialise MarkItDown reader
        file_path = os.fspath(file_path)

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

        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")
        conversion_method: str = None

        prompt = kwargs.get("prompt") or DEFAULT_EXTRACTION_PROMPT
        #  Handle page-by-page OCR for PDFs
        if kwargs.get("scan_pdf_pages", False):
            if ext != "pdf" or self.model is None:
                raise ValueError(
                    "To scan PDF pages, a PDF file and a vision model are required."
                )

            markdown_text = _pdf_pages_to_markdown(file_path=file_path, prompt=prompt)
            conversion_method = "markdown"

        # Regular conversion path
        else:
            markdown_text = md.convert(file_path, llm_prompt=prompt).text_content
            conversion_method = "json" if ext == "json" else "markdown"

        # Return output
        return ReaderOutput(
            text=markdown_text,
            document_name=os.path.basename(file_path),
            document_path=file_path,
            document_id=kwargs.get("document_id", str(uuid.uuid4())),
            conversion_method=conversion_method,
            reader_method="markitdown",
            ocr_method=ocr_method,
            metadata=kwargs.get("metadata", {}),
        )
