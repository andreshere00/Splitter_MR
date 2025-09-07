import warnings
from pathlib import Path

import nltk
import spacy
import tiktoken
from langchain_text_splitters import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)

from ...schema import ReaderOutput, SplitterOutput
from ...schema.constants import (
    DEFAULT_TOKEN_LANGUAGE,
    DEFAULT_TOKENIZER,
    NLTK_DEFAULTS,
    SPACY_DEFAULTS,
    SUPPORTED_TOKENIZERS,
    TIKTOKEN_DEFAULTS,
)
from ..base_splitter import BaseSplitter


class TokenSplitter(BaseSplitter):
    """
    TokenSplitter splits a given text into chunks based on token counts
    derived from different tokenization models or libraries.

    This splitter supports tokenization via `tiktoken` (OpenAI tokenizer),
    `spacy` (spaCy tokenizer), and `nltk` (NLTK tokenizer). It allows splitting
    text into chunks of a maximum number of tokens (`chunk_size`), using the
    specified tokenizer model.

    Args:
        chunk_size (int): Maximum number of tokens per chunk.
        model_name (str): Specifies the tokenizer and model in the format `tokenizer/model`. Supported tokenizers are:

            - `tiktoken/cl100k_base` (OpenAI tokenizer via tiktoken)
            - `spacy/en_core_web_sm` (spaCy English model)
            - `nltk/punkt_tab` (NLTK Punkt tokenizer variant)

        language (str): Language code for NLTK tokenizer (default `"english"`).

    Notes:
        More info about the splitting methods by Tokens for Langchain:
        [Langchain Docs](https://python.langchain.com/docs/how_to/split_by_token/).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        model_name: str = DEFAULT_TOKENIZER,
        language: str = DEFAULT_TOKEN_LANGUAGE,
    ):
        super().__init__(chunk_size)
        # Use centralized defaults (already applied via signature) and keep on instance
        self.model_name = model_name or DEFAULT_TOKENIZER
        self.language = language or DEFAULT_TOKEN_LANGUAGE

    @staticmethod
    def list_nltk_punkt_languages():
        """Return a sorted list of available punkt models (languages) for NLTK."""
        models = set()
        for base in map(Path, nltk.data.path):
            punkt_dir = base / "tokenizers" / "punkt"
            if punkt_dir.exists():
                models.update(f.stem for f in punkt_dir.glob("*.pickle"))
        return sorted(models)

    def _parse_model(self) -> tuple[str, str]:
        """Parse `tokenizer/model` and validate the format."""
        if "/" not in self.model_name:
            raise ValueError(
                "model_name must be in the format 'tokenizer/model', "
                f"e.g. '{DEFAULT_TOKENIZER}'."
            )
        tokenizer, model = self.model_name.split("/", 1)
        return tokenizer, model

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input text from `reader_output` into token-based chunks using
        the specified tokenizer.

        Depending on `model_name`, the splitter chooses the appropriate tokenizer:

        - For `tiktoken`, uses `RecursiveCharacterTextSplitter` with tiktoken encoding.
            e.g.: `tiktoken/cl100k_base`.
        - For `spacy`, uses `SpacyTextSplitter` with the specified spaCy pipeline.
            e.g., `spacy/en_core_web_sm`.
        - For `nltk`, uses `NLTKTextSplitter` with the specified language tokenizer.
            e.g., `nltk/punkt_tab`.

        Automatically downloads spaCy and NLTK models if missing.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary containing at least a 'text' key (str) and optional document metadata,
                such as 'document_name', 'document_path', 'document_id', etc.

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            RuntimeError: If a spaCy model specified in `model_name` is not available.
            ValueError: If an unsupported tokenizer is specified in `model_name`.
        """
        text = reader_output.text
        tokenizer, model = self._parse_model()

        if tokenizer == "tiktoken":
            # Validate against installed tiktoken encodings; hint with our common defaults
            available_models = tiktoken.list_encoding_names()
            if model not in available_models:
                raise ValueError(
                    f"tiktoken encoding '{model}' is not available. "
                    f"Available encodings include (subset): {TIKTOKEN_DEFAULTS}. "
                    f"Full list from tiktoken: {available_models}"
                )
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=model,
                chunk_size=self.chunk_size,
                chunk_overlap=0,
            )

        elif tokenizer == "spacy":
            if not spacy.util.is_package(model):
                # Try to download; we surface our recommended list in the error if it fails
                try:
                    spacy.cli.download(model)
                except Exception as e:
                    raise RuntimeError(
                        f"spaCy model '{model}' is not available for download. "
                        f"Common models include: {SPACY_DEFAULTS}"
                    ) from e
            spacy.load(model)
            MAX_SAFE_LENGTH = 1_000_000
            if self.chunk_size > MAX_SAFE_LENGTH:
                warnings.warn(
                    "Too many characters: the v2.x parser and NER models require roughly "
                    "1GB of temporary memory per 100,000 characters in the input",
                    UserWarning,
                )
            splitter = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,
                max_length=MAX_SAFE_LENGTH,
                pipeline=model,
            )

        elif tokenizer == "nltk":
            # Ensure punkt language is present; download our specified default model if missing
            try:
                nltk.data.find(f"tokenizers/punkt/{self.language}.pickle")
            except LookupError:
                # Use constants instead of hard-coded 'punkt_tab'
                nltk.download(NLTK_DEFAULTS[0])
            splitter = NLTKTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,
                language=self.language,
            )

        else:
            raise ValueError(
                f"Unsupported tokenizer '{tokenizer}'. Supported tokenizers: {SUPPORTED_TOKENIZERS}"
            )

        chunks = splitter.split_text(text)
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="token_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "model_name": self.model_name,  # keeps centralized default visible
                "language": self.language,  # keeps centralized default visible
            },
            metadata=metadata,
        )
