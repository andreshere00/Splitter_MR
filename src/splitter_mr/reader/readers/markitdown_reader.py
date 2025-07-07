import io
import os
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

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

    def _pdf_pages_to_streams(self, pdf_path: str) -> List[io.BytesIO]:
        """
        Convert each PDF page to a PNG and wrap in a BytesIO stream.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[io.BytesIO]: List of PNG image streams for each page.
        """
        doc = fitz.open(pdf_path)
        streams = []
        for idx in range(len(doc)):
            pix = doc.load_page(idx).get_pixmap()
            buf = io.BytesIO(pix.tobytes("png"))
            buf.name = f"page_{idx + 1}.png"
            buf.seek(0)
            streams.append(buf)
        return streams

    def _pdf_pages_to_markdown(
        self, file_path: str, md: MarkItDown, prompt: str, page_placeholder: str
    ) -> str:
        """
        Convert each scanned PDF page to markdown using the provided MarkItDown instance.

        Args:
            file_path (str): Path to PDF.
            md (MarkItDown): The MarkItDown converter instance.
            prompt (str): The LLM prompt for OCR.
            page_placeholder (str): Page break placeholder for markdown.

        Returns:
            str: Markdown of the entire PDF (one page per placeholder).
        """
        page_md = []
        for idx, page_stream in enumerate(
            self._pdf_pages_to_streams(file_path), start=1
        ):
            page_md.append(page_placeholder.replace("{page}", str(idx)))
            result = md.convert(page_stream, llm_prompt=prompt)
            page_md.append(result.text_content)
        return "\n".join(page_md)

    def _get_markitdown(self) -> tuple:
        """
        Returns a MarkItDown instance and OCR method name depending on model presence.

        Returns:
            tuple[MarkItDown, Optional[str]]: MarkItDown instance, OCR method or None.

        Raises:
            ValueError: If provided model is not supported.
        """
        if self.model:
            if not isinstance(self.model, (OpenAIVisionModel, AzureOpenAIVisionModel)):
                raise ValueError(
                    "Incompatible client. Only AzureOpenAIVisionModel or OpenAIVisionModel are supported."
                )
            client = self.model.get_client()
            return (
                MarkItDown(llm_client=client, llm_model=self.model.model_name),
                self.model.model_name,
            )
        else:
            return MarkItDown(), None

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
                - `page_placeholder (str)`: Markdown placeholder string for pages (default: "<!-- page -->").

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

        # Initialize MarkItDown reader
        file_path = os.fspath(file_path)
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        prompt = kwargs.get("prompt", DEFAULT_EXTRACTION_PROMPT)
        page_placeholder = kwargs.get("page_placeholder", "<!-- page -->")
        conversion_method: str

        md, ocr_method = self._get_markitdown()

        if self.model is not None:
            markdown_text = self._pdf_pages_to_markdown(
                file_path=file_path,
                md=md,
                prompt=prompt,
                page_placeholder=page_placeholder,
            )
            conversion_method = "markdown"
        else:
            markdown_text = md.convert(file_path, llm_prompt=prompt).text_content
            conversion_method = "json" if ext == "json" else "markdown"

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
