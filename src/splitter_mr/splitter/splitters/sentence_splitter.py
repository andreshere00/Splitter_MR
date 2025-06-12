import re
from typing import List, Union

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class SentenceSplitter(BaseSplitter):
    """
    SentenceSplitter splits a given text into overlapping or non-overlapping chunks,
    where each chunk contains a specified number of sentences, and overlap is defined
    by a number or percentage of words from the end of the previous chunk.

    Args:
        chunk_size (int): Maximum number of sentences per chunk.
        chunk_overlap (Union[int, float]): Number or percentage of overlapping words between chunks.
        separators (Union[str, List[str]]): Character(s) to split sentences.
    """

    def __init__(
        self,
        chunk_size: int = 5,
        chunk_overlap: Union[int, float] = 0,
        separators: Union[str, List[str]] = [".", "!", "?"],
    ):
        super().__init__(chunk_size)
        self.chunk_overlap = chunk_overlap
        self.sentence_separators = (
            separators if isinstance(separators, list) else [separators]
        )

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text from the `reader_output` dictionary into sentence-based chunks,
        allowing for overlap at the word level.

        Each chunk contains at most `chunk_size` sentences, where sentence boundaries are
        detected using the specified `sentence_separators` (e.g., '.', '!', '?').
        Overlap between consecutive chunks is specified by `chunk_overlap`, which can be an
        integer (number of words) or a float (fraction of the maximum words in a sentence).
        This is useful for downstream NLP tasks that require context preservation.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary containing at least a 'text' key (str) and optional document metadata,
                such as 'document_name', 'document_path', 'document_id', etc.

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If `chunk_overlap` is negative or greater than or equal to `chunk_size`.
            ValueError: If 'text' is missing in `reader_output`.

        Example:
            ```python
            from splitter_mr.splitter import SentenceSplitter

            # Example input: 7 sentences with varied punctuation
            # This dictionary has been obtained as an output from a Reader class.
            reader_output = ReaderOutput(
                text: "Hello world! How are you? I am fine. Testing sentence splitting. Short. End! And another?",
                document_name: "sample.txt",
                document_path: "/tmp/sample.txt",
                document_id: "123"
            )

            # Split into chunks of 3 sentences each, no overlap
            splitter = SentenceSplitter(chunk_size=3, chunk_overlap=0)
            result = splitter.split(reader_output)
            print(result.chunks)
            ```
            ```python
            ['Hello world! How are you? I am fine.',
            'Testing sentence splitting. Short. End!',
            'And another?', ...]
            ```
        """
        # Initialize variables
        text = reader_output.text
        chunk_size = self.chunk_size

        # Split text into sentences
        separators_pattern = "|".join([re.escape(d) for d in self.sentence_separators])
        sentences = re.split(f"({separators_pattern})", text)
        merged_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i].strip()
            punct = sentences[i + 1].strip() if i + 1 < len(sentences) else ""
            merged = (sent + punct).strip()
            if merged:
                merged_sentences.append(merged)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            merged_sentences.append(sentences[-1].strip())
        num_sentences = len(merged_sentences)

        # Determine overlap in words
        if isinstance(self.chunk_overlap, float) and 0 <= self.chunk_overlap < 1:
            max_sent_words = max((len(s.split()) for s in merged_sentences), default=0)
            overlap = int(max_sent_words * self.chunk_overlap)
        else:
            overlap = int(self.chunk_overlap)

        # Split into sentences
        chunks = []
        start = 0
        while start < num_sentences:
            end = min(start + chunk_size, num_sentences)
            chunk_sents = merged_sentences[start:end]
            chunk_text = " ".join(chunk_sents)
            if overlap > 0 and chunks:
                prev_words = chunks[-1].split()
                overlap_words = (
                    prev_words[-overlap:] if overlap <= len(prev_words) else prev_words
                )
                chunk_text = " ".join([" ".join(overlap_words), chunk_text]).strip()
            chunks.append(chunk_text)
            start += chunk_size

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
            split_method="sentence_splitter",
            split_params={
                "chunk_size": chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "sentence_separators": self.sentence_separators,
            },
            metadata=metadata,
        )
        return output
