import base64
from collections import defaultdict
from io import BytesIO
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from ...model import BaseModel
from ...schema import DEFAULT_IMAGE_CAPTION_PROMPT


class PDFPlumberReader:
    """
    Extracts structured content from PDF files using pdfplumber.

    This reader supports extracting and grouping text lines, identifying and extracting tables,
    detecting and extracting images (optionally with LLM-based annotation), and producing
    Markdown output with optional image rendering.

    Supported Output Types:
        - Text lines, grouped by visual alignment
        - Tables (as Markdown)
        - Images (as base64-encoded PNG, with optional annotation/caption)

    Example:
        ```python
        reader = PDFPlumberReader()
        markdown = reader.read("example.pdf", show_base64_images=True)
        print(markdown)
        ```
    """

    def group_by_lines(
        self, words: List[Dict[str, Any]], tolerance: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Groups OCR word dictionaries into text lines based on their vertical position.

        Args:
            words (List[Dict[str, Any]]): List of word dicts as returned by pdfplumber's `extract_words()`.
            tolerance (float): Tolerance in pixels for considering words as part of the same line.

        Returns:
            List[Dict[str, Any]]: List of line dictionaries, each containing line text and vertical coordinates.
        """
        lines = defaultdict(list)
        for word in words:
            top = round(word["top"] / tolerance) * tolerance
            lines[top].append(word)
        sorted_lines = []
        for top in sorted(lines):
            sorted_words = sorted(lines[top], key=lambda w: w["x0"])
            line_text = " ".join([w["text"] for w in sorted_words])
            sorted_lines.append(
                {
                    "type": "text",
                    "top": top,
                    "bottom": max(w["bottom"] for w in sorted_words),
                    "content": line_text,
                }
            )
        return sorted_lines

    def is_real_table(self, table: List[List[Any]]) -> bool:
        """
        Heuristically determines if a detected table is likely to be a meaningful table.

        Args:
            table (List[List[Any]]): 2D list representing table rows and columns.

        Returns:
            bool: True if the table passes basic heuristics (not mostly single-column or blank rows), else False.
        """
        if not table or len(table) < 2:
            return False
        col_counts = [len(row) for row in table if row]
        if col_counts.count(1) > len(col_counts) * 0.7:
            return False
        if max(col_counts) < 2:
            return False
        return True

    def extract_tables(
        self, page, page_num: int
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, float, float]]]:
        """
        Extracts valid tables from a PDF page.

        Args:
            page: pdfplumber page object.
            page_num (int): Page number.

        Returns:
            Tuple[
                List[Dict[str, Any]],                # List of table block dicts
                List[Tuple[float, float, float, float]] # Bounding boxes for extracted tables
            ]
        """
        tables = []
        table_bboxes = []
        for table in page.find_tables():
            bbox = table.bbox
            extracted = table.extract()
            cleaned = [
                [cell if cell is not None else "" for cell in row]
                for row in extracted
                if any(cell not in (None, "", " ") for cell in row)
            ]
            if self.is_real_table(cleaned):
                table_bboxes.append(bbox)
                tables.append(
                    {
                        "type": "table",
                        "top": bbox[1],
                        "bottom": bbox[3],
                        "content": cleaned,
                        "page": page_num,
                    }
                )
        return tables, table_bboxes

    def extract_images(
        self,
        page,
        page_num: int,
        prompt: Optional[str] = None,
        model: Optional[BaseModel] = None,
        image_placeholder: str = "<!-- image -->",
    ) -> List[Dict[str, Any]]:
        """
        Extracts images from a PDF page as base64-encoded PNG data, with optional annotation via a model.

        Args:
            page: pdfplumber page object.
            page_num (int): Page number.
            prompt (Optional[str]): Prompt for the annotation model.
            model (Optional[BaseModel]): Optional model to generate image captions/annotations.

        Returns:
            List[Dict[str, Any]]: List of image block dicts, each containing image URI and annotation if available.
        """
        images = []
        for idx, img in enumerate(page.images):
            try:
                bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                cropped = page.within_bbox(bbox).to_image(resolution=150)
                from io import BytesIO

                buf = BytesIO()
                cropped.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode()
                img_uri = f"data:image/png;base64,{img_b64}"

                image_description = image_placeholder
                if model:
                    annotation = model.extract_text(file=img_b64, prompt=prompt)
                    image_description += f"\n{annotation}"

                images.append(
                    {
                        "type": "image",
                        "top": img["top"],
                        "bottom": img["bottom"],
                        "content": img_uri,
                        "annotation": image_description,
                        "page": page_num,
                    }
                )
            except Exception as e:
                print(f"Error encoding image: {e}")
        return images

    def extract_text(
        self, page, page_num: int, table_bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts and groups lines of text from a PDF page, excluding those overlapping tables.

        Args:
            page: pdfplumber page object.
            page_num (int): Page number.
            table_bboxes (List[Tuple[float, float, float, float]]): List of table bounding boxes.

        Returns:
            List[Dict[str, Any]]: List of text line block dicts.
        """
        lines = self.group_by_lines(page.extract_words())
        texts = []
        for line in lines:
            mid_y = (line["top"] + line["bottom"]) / 2
            in_table = any(b[1] <= mid_y <= b[3] for b in table_bboxes)
            if not in_table:
                line["page"] = page_num
                texts.append(line)
        return texts

    def extract_page_blocks(
        self,
        page,
        page_num: int,
        prompt: Optional[str] = None,
        model: Optional[BaseModel] = None,
        image_placeholder: str = "<!-- image -->",
    ) -> List[Dict[str, Any]]:
        """
        Extracts all structural content blocks (tables, images, text) from a PDF page.

        Args:
            page: pdfplumber page object.
            page_num (int): Page number.
            prompt (Optional[str]): Prompt for image annotation.
            model (Optional[BaseModel], optional): Model for image annotation.

        Returns:
            List[Dict[str, Any]]: List of all content block dicts, sorted by vertical position.
        """
        tables, table_bboxes = self.extract_tables(page, page_num)
        images = self.extract_images(
            page,
            page_num=page_num,
            model=model,
            prompt=prompt,
            image_placeholder=image_placeholder,
        )
        texts = self.extract_text(
            page=page, page_num=page_num, table_bboxes=table_bboxes
        )
        blocks = tables + images + texts
        return sorted(blocks, key=lambda x: x["top"])

    def extract_pages_as_images(
        self,
        file_path: str,
        resolution: int = 300,
        image_format: str = "PNG",
        return_base64: bool = True,
    ) -> List[str]:
        """
        Render every page of *file_path* to an image.

        Args:
            file_path (str): PDF to rasterise.
            resolution (int): DPI for rendering (72-600).
            image_format (str): Pillow output format ("PNG", "JPEG", …).
            return_base64 (bool): If True (default) return base64 strings
                (without the `data:image/*;base64,` prefix); otherwise raw bytes.

        Returns:
            List[str] | List[bytes]: One element per page in order.
        """
        pages_data: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_img = page.to_image(resolution=resolution)
                buf = BytesIO()
                page_img.save(buf, format=image_format)
                data_bytes = buf.getvalue()
                pages_data.append(
                    base64.b64encode(data_bytes).decode()
                    if return_base64
                    else data_bytes
                )
        return pages_data

    def table_to_markdown(self, table: List[List[Any]]) -> str:
        """
        Converts a table (list of lists) to GitHub-flavored Markdown.

        Args:
            table (List[List[Any]]): Table as a 2D list.

        Returns:
            str: Markdown-formatted table string.
        """
        if not table or not isinstance(table, list) or not table[0]:
            return ""
        max_cols = max(len(row) for row in table)
        padded = [row + [""] * (max_cols - len(row)) for row in table]
        header = (
            "| "
            + " | ".join(  # noqa: W503
                str(cell).strip().replace("\n", " ") for cell in padded[0]
            )
            + " |"  # noqa: W503
        )
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        rows = [
            "| "
            + " | ".join(  # noqa: W503
                str(cell).strip().replace("\n", " ") for cell in row
            )  # noqa: W503
            + " |"  # noqa: W503
            for row in padded[1:]
        ]
        return "\n".join([header, separator] + rows)

    def blocks_to_markdown(
        self,
        all_blocks: List[Dict[str, Any]],
        show_base64_images: bool = True,
        image_placeholder: str = "<!-- image -->",
        page_placeholder: str = "<!-- page -->",
    ) -> str:
        """
        Converts a list of content blocks into Markdown, optionally embedding images and tables.

        Args:
            all_blocks (List[Dict[str, Any]]): All content blocks, possibly across multiple pages.
            show_base64_images (bool): Whether to render images inline. If False, images are omitted or replaced with an indicator.
            `image_placeholder (Optional[str])`: Placeholder string to use for omitted images in PDFs. Default is `"<!-- image -->"`.
            `page_placeholder (Optional[str])`: Placeholder string for PDF page breaks. Default is `"<!-- page -->"`.

        Returns:
            str: Markdown document representing the extracted content.
        """
        md_lines: List[str] = [""]
        all_blocks.sort(key=lambda x: (x["page"], x["top"]))
        for page, blocks in groupby(all_blocks, key=lambda x: x["page"]):
            md_lines += [page_placeholder + "\n"]
            last_type = None
            paragraph: List[str] = []
            for item in blocks:
                if item["type"] == "text":
                    if last_type not in (None, "text") and paragraph:
                        md_lines.append("\n".join(paragraph))
                        md_lines.append("")
                        paragraph = []
                    paragraph.append(item["content"])
                    last_type = "text"
                else:
                    if paragraph:
                        md_lines.append("\n".join(paragraph))
                        md_lines.append("")
                        paragraph = []
                    if item["type"] == "image":
                        if show_base64_images:
                            md_lines.append(
                                f'![Image page {item["page"]}]({item["content"]})\n'
                            )
                        elif item.get("annotation"):
                            md_lines.append(f'{item["annotation"]}\n')
                        else:
                            md_lines.append(f"\n{image_placeholder}\n")
                    elif item["type"] == "table":
                        md_lines.append(self.table_to_markdown(item["content"]))
                        md_lines.append("")
                    last_type = item["type"]
            if paragraph:
                md_lines.append("\n".join(paragraph))
                md_lines.append("")
        # Remove redundant blank lines
        clean_lines: List[str] = []
        for line in md_lines:
            if line != "" or (clean_lines and clean_lines[-1] != ""):
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def describe_pages(
        self,
        file_path: str,
        model: BaseModel,
        prompt: Optional[str] = None,
        resolution: Optional[int] = 300,
        **parameters,
    ) -> List[str]:
        """
        Uses a Vision-Language Model (VLM) to describe each full PDF page.

        Args:
            file_path (str): PDF to process.
            model (BaseModel): Any implementation of the provided BaseModel
                               interface (e.g. OpenAIVisionModel).
            prompt (Optional[str]): Instruction sent to the VLM.  Has the default
                          requested in the spec.
            resolution (Optional[str]): DPI to rasterise pages before sending.

        Returns:
            List[str]: Markdown descriptions, one per page, in order.
        """
        page_images_b64 = self.extract_pages_as_images(
            file_path=file_path, resolution=resolution, return_base64=True
        )

        descriptions: List[str] = []
        for i, img_b64 in enumerate(page_images_b64, start=1):
            try:
                description = model.extract_text(
                    file=img_b64, prompt=prompt, **parameters
                )
            except Exception as e:
                description = f"**Error on page {i}:** {e}"
            descriptions.append(description)

        return descriptions

    def read(
        self,
        file_path: str,
        prompt: Optional[str] = None,
        model: Optional[BaseModel] = None,
        show_base64_images: bool = False,
        image_placeholder: str = "<!-- image -->",
        page_placeholder: str = "<!-- page -->",
    ) -> str:
        """
        Reads a PDF file and returns extracted content as Markdown.

        Args:
            file_path (str): Path to the PDF file.
            model (Optional[BaseModel], optional): Optional model for image annotation.
            prompt (str, optional): Prompt for the image annotation model.
            show_base64_images (bool, optional): If True, images are included as base64 in Markdown. If False, they are omitted or replaced with a image_placeholder.
            `image_placeholder (Optional[str])`: Placeholder string to use for omitted images in PDFs. Default is `"<!-- image -->"`.
            `page_placeholder (Optional[str])`: Placeholder string for PDF page breaks. Default is `"<!-- page -->"`.

        Returns:
            str: Markdown-formatted string with structured content from the PDF.
        """
        all_blocks: List[Dict[str, Any]] = []
        prompt = prompt or DEFAULT_IMAGE_CAPTION_PROMPT
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                all_blocks.extend(
                    self.extract_page_blocks(
                        page=page,
                        page_num=i,
                        model=model,
                        prompt=prompt,
                        image_placeholder=image_placeholder,
                    )
                )
        markdown_test = self.blocks_to_markdown(
            all_blocks,
            show_base64_images=show_base64_images,
            image_placeholder=image_placeholder,
            page_placeholder=page_placeholder,
        )
        return markdown_test
