import re
from typing import List, Union

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class ParagraphSplitter(BaseSplitter):
    """
    ParagraphSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk contains a specified number of paragraphs, and overlap is defined
    by a number or percentage of words from the end of the previous chunk.

    Args:
        chunk_size (int): Maximum number of paragraphs per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping words between chunks.
        line_break (Union[str, List[str]]): Character(s) used to split text into paragraphs.
    """

    def __init__(
        self,
        chunk_size: int = 3,
        chunk_overlap: Union[int, float] = 0,
        line_break: Union[str, List[str]] = "\n",
    ):
        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.line_break = line_break if isinstance(line_break, list) else [line_break]

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits text in `reader_output['text']` into paragraph-based chunks, with optional word overlap.

        Args:
            reader_output (Dict[str, Any]): Dictionary containing at least a 'text' key (str)
                and optional document metadata (e.g., 'document_name', 'document_path').

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If 'text' is missing from `reader_output` or is not a string.

        Example:
            ```python
            from splitter_mr.splitter import ParagraphSplitter

            # This dictionary has been obtained as the output from a Reader object.
            reader_output = ReaderOutput(
                text: "Para 1.\\n\\nPara 2.\\n\\nPara 3.",
                document_name: "test.txt",
                document_path: "/tmp/test.txt"
            )
            splitter = ParagraphSplitter(chunk_size=2, chunk_overlap=1, line_break="\\n\\n")
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            ['Para 1.\\n\\nPara 2.', '2. Para 3.']
            ```
        """
        # Intialize variables
        text = reader_output.text
        line_breaks_pattern = "|".join(map(re.escape, self.line_break))
        paragraphs = [p for p in re.split(line_breaks_pattern, text) if p.strip()]
        num_paragraphs = len(paragraphs)

        # Determine overlap in words
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            max_para_words = max((len(p.split()) for p in paragraphs), default=0)
            overlap = int(max_para_words * self.chunk_overlap)
        else:
            overlap = int(self.chunk_overlap)

        # Split into chunks
        chunks = []
        start = 0
        while start < num_paragraphs:
            end = min(start + self.chunk_size, num_paragraphs)
            chunk_paragraphs = paragraphs[start:end]
            chunk_text = self.line_break[0].join(chunk_paragraphs)
            if overlap > 0 and chunks:
                prev_words = chunks[-1].split()
                overlap_words = (
                    prev_words[-overlap:] if overlap <= len(prev_words) else prev_words
                )
                chunk_text = (
                    self.line_break[0]
                    .join([" ".join(overlap_words), chunk_text])
                    .strip()
                )
            chunks.append(chunk_text)
            start += self.chunk_size

        # Generate chunk_id and append metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        # Return output
        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="paragraph_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "line_break": self.line_break,
            },
            metadata=metadata,
        )
        return output
