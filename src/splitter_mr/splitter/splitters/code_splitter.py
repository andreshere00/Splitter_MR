from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


def get_langchain_language(lang_str: str) -> Language:
    """
    Map a string language name to Langchain Language enum.
    Raises ValueError if not found.
    """
    lookup = {lang.name.lower(): lang for lang in Language}
    key = lang_str.lower()
    if key not in lookup:
        raise ValueError(
            f"Unsupported language '{lang_str}'. Supported: {list(lookup.keys())}"
        )
    return lookup[key]


class CodeSplitter(BaseSplitter):
    """
    CodeSplitter recursively splits source code into programmatically meaningful chunks
    (functions, classes, methods, etc.) for the given programming language.

    Args:
        chunk_size (int): Maximum chunk size, in characters.
        language (str): Programming language (e.g., "python", "java", "kotlin", etc.)

    Notes:
        - Uses Langchain's RecursiveCharacterTextSplitter and its language-aware `from_language` method.
        - See Langchain docs: https://python.langchain.com/docs/how_to/code_splitter/
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        language: str = "python",
    ):
        super().__init__(chunk_size)
        self.language = language

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits code in `reader_output['text']` according to the syntax of the specified
        programming language, using function/class boundaries where possible.

        Args:
            reader_output (ReaderOutput): Object containing at least a 'text' field,
                plus optional document metadata.

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If language is not supported.

        Example:
            ```python
            from splitter_mr.splitter import CodeSplitter

            reader_output = ReaderOutput(
                text: "def foo():\\n    pass\\n\\nclass Bar:\\n    def baz(self):\\n        pass",
                document_name: "example.py",
                document_path: "/tmp/example.py"
            )
            splitter = CodeSplitter(chunk_size=50, language="python")
            output = splitter.split(reader_output)
            print(output.chunks)
            ```
            ```python
            ['def foo():\\n    pass\\n', 'class Bar:\\n    def baz(self):\\n        pass']
            ```
        """
        # Initialize variables
        text = reader_output.text
        chunk_size = self.chunk_size

        # Get Langchain language enum
        lang_enum = get_langchain_language(self.language)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=chunk_size, chunk_overlap=0
        )
        texts = splitter.create_documents([text])
        chunks = [doc.page_content for doc in texts]

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
            split_method="code_splitter",
            split_params={"chunk_size": chunk_size, "language": self.language},
            metadata=metadata,
        )
        return output
