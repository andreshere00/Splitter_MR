"""Explicit reader, splitter, vision-model, and embedding factories."""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr, ValidationError

from splitter_mr.reader.base_reader import BaseReader
from splitter_mr.reader.readers import EXTRA_BY_NAME, REGISTRY
from splitter_mr.schema.constants import DEFAULT_RECURSIVE_SEPARATORS
from splitter_mr.splitter import (
    CharacterSplitter,
    CodeSplitter,
    HeaderSplitter,
    HTMLTagSplitter,
    KeywordSplitter,
    PagedSplitter,
    ParagraphSplitter,
    RecursiveCharacterSplitter,
    RecursiveJSONSplitter,
    RowColumnSplitter,
    SemanticSplitter,
    SentenceSplitter,
    TokenSplitter,
    WordSplitter,
)
from splitter_mr.splitter.base_splitter import BaseSplitter

from .enums import (
    ComponentStatus,
    EmbeddingName,
    ReaderName,
    SourceType,
    VisionModelName,
)
from .exceptions import ServerComponentUnavailableError, ServerConfigurationError
from .schemas import (
    READER_SOURCE_TYPES,
    READER_VISION_MODELS,
    ComponentCatalogResponse,
    EmbeddingConfiguration,
    EmbeddingDescriptor,
    ReaderConfiguration,
    ReaderDescriptor,
    SplitterConfiguration,
    SplitterDescriptor,
    TextractReaderConfiguration,
    VisionModelConfiguration,
    VisionModelDescriptor,
)

READER_PURPOSES: dict[str, str] = {
    ReaderName.VANILLA.value: (
        "Core multi-format reader for text, JSON, local files, and URLs."
    ),
    ReaderName.MARKITDOWN.value: (
        "Markdown conversion for Office, PDF, and HTML via Microsoft MarkItDown."
    ),
    ReaderName.DOCLING.value: (
        "High-quality PDF and document conversion via IBM Docling."
    ),
    ReaderName.TEXTRACT.value: (
        "AWS Textract OCR for local PDFs, Office files, and images."
    ),
}

READER_LIMITATIONS: dict[str, str] = {
    ReaderName.VANILLA.value: (
        "Optional model JSON requires splitter-mr[multimodal]. Inline text and "
        "JSON file_path values are supported."
    ),
    ReaderName.MARKITDOWN.value: (
        "Requires the markitdown extra. File paths and URLs only. Compatible "
        "vision models must expose an OpenAI client."
    ),
    ReaderName.DOCLING.value: ("Requires the docling extra. File paths and URLs only."),
    ReaderName.TEXTRACT.value: (
        "Requires the textract extra and AWS credentials. Server-local files "
        "only. Vision models are not accepted."
    ),
}

READER_CONFIG_SCHEMA: dict[str, str] = {
    ReaderName.VANILLA.value: "VanillaReaderConfiguration",
    ReaderName.MARKITDOWN.value: "MarkItDownReaderConfiguration",
    ReaderName.DOCLING.value: "DoclingReaderConfiguration",
    ReaderName.TEXTRACT.value: "TextractReaderConfiguration",
}

VISION_MODEL_PURPOSES: dict[str, str] = {
    VisionModelName.OPENAI.value: "OpenAI Chat Completions vision models.",
    VisionModelName.AZURE_OPENAI.value: "Azure OpenAI vision deployments.",
    VisionModelName.ANTHROPIC.value: "Anthropic Claude vision models via the OpenAI SDK.",
    VisionModelName.OPENROUTER.value: "OpenRouter vision models via the OpenAI SDK.",
    VisionModelName.GROK.value: "xAI Grok vision models.",
    VisionModelName.GEMINI.value: "Google Gemini vision models.",
    VisionModelName.HUGGINGFACE.value: "Local or Hub Hugging Face vision-language models.",
}

VISION_MODEL_LIMITATIONS: dict[str, str] = {
    VisionModelName.GROK.value: (
        "MarkItDownReader rejects Grok because its client is openai.Client, "
        "not openai.OpenAI."
    ),
    VisionModelName.GEMINI.value: "Not compatible with MarkItDownReader.",
    VisionModelName.HUGGINGFACE.value: (
        "Not compatible with MarkItDownReader. Loads local weights and may "
        "use substantial memory."
    ),
}

VISION_MODEL_CONFIG_SCHEMA: dict[str, str] = {
    name.value: f"{name.value}Configuration" for name in VisionModelName
}

SPLITTER_FACTORIES: dict[str, type[BaseSplitter]] = {
    "CharacterSplitter": CharacterSplitter,
    "WordSplitter": WordSplitter,
    "SentenceSplitter": SentenceSplitter,
    "ParagraphSplitter": ParagraphSplitter,
    "RecursiveCharacterSplitter": RecursiveCharacterSplitter,
    "KeywordSplitter": KeywordSplitter,
    "TokenSplitter": TokenSplitter,
    "PagedSplitter": PagedSplitter,
    "RowColumnSplitter": RowColumnSplitter,
    "RecursiveJSONSplitter": RecursiveJSONSplitter,
    "HTMLTagSplitter": HTMLTagSplitter,
    "HeaderSplitter": HeaderSplitter,
    "CodeSplitter": CodeSplitter,
    "SemanticSplitter": SemanticSplitter,
}

SPLITTER_PURPOSES: dict[str, str] = {
    "CharacterSplitter": "Fixed-size character chunks with optional overlap.",
    "WordSplitter": "Fixed-size word chunks with optional overlap.",
    "SentenceSplitter": "Chunks by sentence count with optional word overlap.",
    "ParagraphSplitter": "Chunks by paragraph count with optional word overlap.",
    "RecursiveCharacterSplitter": (
        "Recursively split on a separator hierarchy until chunks fit."
    ),
    "KeywordSplitter": "Split around regular-expression keyword boundaries.",
    "TokenSplitter": "Token-count chunks using tiktoken, spaCy, or NLTK.",
    "PagedSplitter": "Group pages using the reader page placeholder.",
    "RowColumnSplitter": "Split tabular data by rows, columns, or size.",
    "RecursiveJSONSplitter": "Recursively split JSON while preserving structure.",
    "HTMLTagSplitter": "Split HTML by tag, with optional Markdown conversion.",
    "HeaderSplitter": "Split Markdown or HTML by heading levels.",
    "CodeSplitter": "Language-aware source-code chunks.",
    "SemanticSplitter": (
        "Semantic similarity splitting using a live BaseEmbedding backend."
    ),
}

SEMANTIC_SPLITTER_NAME = "SemanticSplitter"
VISION_MODEL_EXTRA = "multimodal"
EMBEDDING_EXTRA = "multimodal"

EMBEDDING_PURPOSES: dict[str, str] = {
    EmbeddingName.OPENAI.value: "OpenAI text-embedding models.",
    EmbeddingName.AZURE_OPENAI.value: "Azure OpenAI embedding deployments.",
    EmbeddingName.OPENROUTER.value: "OpenRouter embedding models via the OpenAI SDK.",
    EmbeddingName.GEMINI.value: "Google Gemini embedding models.",
    EmbeddingName.HUGGINGFACE.value: (
        "Local or Hub Sentence-Transformers embedding models."
    ),
    EmbeddingName.ANTHROPIC.value: "Voyage AI embeddings following Anthropic guidance.",
}

EMBEDDING_LIMITATIONS: dict[str, str] = {
    EmbeddingName.HUGGINGFACE.value: (
        "Loads local weights and may use substantial memory. Torch is required."
    ),
}

EMBEDDING_CONFIG_SCHEMA: dict[str, str] = {
    name.value: f"{name.value}Configuration" for name in EmbeddingName
}


def _is_reader_available(name: str) -> bool:
    """Return whether a reader class can be imported in this environment.

    Args:
        name: Public reader class name.

    Returns:
        ``True`` when the class imports successfully.
    """
    try:
        from splitter_mr.reader import readers as reader_pkg

        getattr(reader_pkg, name)
    except ModuleNotFoundError:
        return False
    return True


def _is_vision_model_available(name: str) -> bool:
    """Return whether a vision-model class can be imported.

    Args:
        name: Public vision-model class name.

    Returns:
        ``True`` when the class imports successfully.
    """
    try:
        from splitter_mr.model import models as model_pkg

        getattr(model_pkg, name)
    except ModuleNotFoundError:
        return False
    return True


def _is_embedding_available(name: str) -> bool:
    """Return whether an embedding class can be imported.

    Args:
        name: Public embedding class name.

    Returns:
        ``True`` when the class imports successfully.
    """
    try:
        from splitter_mr.embedding import embeddings as embedding_pkg

        getattr(embedding_pkg, name)
    except ModuleNotFoundError:
        return False
    return True


def create_vision_model(config: VisionModelConfiguration) -> Any:
    """Instantiate a vision model from a validated JSON configuration.

    Args:
        config: Discriminated vision-model configuration.

    Returns:
        A ``BaseVisionModel`` implementation.

    Raises:
        ServerComponentUnavailableError: If the multimodal extra is missing.
        ServerConfigurationError: If constructor validation fails.
    """
    name = config.model
    try:
        from splitter_mr.model import models as model_pkg

        model_cls = getattr(model_pkg, name)
    except ModuleNotFoundError as error:
        raise ServerComponentUnavailableError(
            f"{name} requires the '{VISION_MODEL_EXTRA}' extra. "
            f"Install with: pip install 'splitter-mr[{VISION_MODEL_EXTRA}]'"
        ) from error

    kwargs = _secret_constructor_kwargs(config, exclude={"model"})
    try:
        return model_cls(**kwargs)
    except ValueError as error:
        raise ServerConfigurationError(str(error)) from error


def create_embedding(config: EmbeddingConfiguration) -> Any:
    """Instantiate an embedding backend from a validated JSON configuration.

    Args:
        config: Discriminated embedding configuration.

    Returns:
        A ``BaseEmbedding`` implementation.

    Raises:
        ServerComponentUnavailableError: If the multimodal extra is missing.
        ServerConfigurationError: If constructor validation fails.
    """
    name = config.embedding
    try:
        from splitter_mr.embedding import embeddings as embedding_pkg

        embedding_cls = getattr(embedding_pkg, name)
    except ModuleNotFoundError as error:
        raise ServerComponentUnavailableError(
            f"{name} requires the '{EMBEDDING_EXTRA}' extra. "
            f"Install with: pip install 'splitter-mr[{EMBEDDING_EXTRA}]'"
        ) from error

    kwargs = _secret_constructor_kwargs(config, exclude={"embedding"})
    try:
        return embedding_cls(**kwargs)
    except ValueError as error:
        raise ServerConfigurationError(str(error)) from error


def create_reader(
    config: ReaderConfiguration,
    vision_model: Any | None = None,
) -> BaseReader:
    """Instantiate a reader from a validated configuration.

    Args:
        config: Discriminated reader configuration.
        vision_model: Optional constructed ``BaseVisionModel`` instance.

    Returns:
        A ``BaseReader`` implementation.

    Raises:
        ServerComponentUnavailableError: If the extra is missing or the name is
            unknown.
        ServerConfigurationError: If the reader rejects the vision model.
    """
    name = config.reader
    if name not in REGISTRY:
        raise ServerComponentUnavailableError(f"Unknown reader: {name}")
    try:
        from splitter_mr.reader import readers as reader_pkg

        reader_cls = getattr(reader_pkg, name)
    except ModuleNotFoundError as error:
        extra = EXTRA_BY_NAME.get(name)
        hint = (
            f"{name} requires the '{extra}' extra. "
            f"Install with: pip install 'splitter-mr[{extra}]'"
            if extra
            else str(error)
        )
        raise ServerComponentUnavailableError(hint) from error

    constructor_kwargs = _reader_constructor_kwargs(config, vision_model)
    try:
        return reader_cls(**constructor_kwargs)
    except Exception as error:
        from splitter_mr.schema.exceptions import ReaderConfigException

        if isinstance(error, ReaderConfigException):
            raise ServerConfigurationError(str(error)) from error
        raise


def create_splitter(
    config: SplitterConfiguration,
    extra_kwargs: dict[str, Any] | None = None,
    embedding: Any | None = None,
) -> BaseSplitter:
    """Instantiate a splitter from a validated configuration.

    Args:
        config: Discriminated splitter configuration.
        extra_kwargs: Additional constructor arguments merged into ``config``.
            Values in ``extra_kwargs`` override the same keys on ``config``.
        embedding: Constructed ``BaseEmbedding`` instance. Required for
            ``SemanticSplitter``.

    Returns:
        A ``BaseSplitter`` implementation.

    Raises:
        ServerComponentUnavailableError: If the splitter name is not in the
            allowlist.
        ServerConfigurationError: If constructor arguments are invalid or
            embedding usage does not match the selected splitter.
    """
    name = config.splitter
    splitter_cls = SPLITTER_FACTORIES.get(name)
    if splitter_cls is None:
        raise ServerComponentUnavailableError(f"Unsupported splitter: {name}.")
    if name == SEMANTIC_SPLITTER_NAME and embedding is None:
        raise ServerConfigurationError(
            "SemanticSplitter requires a top-level embedding configuration."
        )
    if name != SEMANTIC_SPLITTER_NAME and embedding is not None:
        raise ServerConfigurationError(
            "embedding is only valid when splitter is SemanticSplitter."
        )

    merged = _merge_splitter_kwargs(config, extra_kwargs)
    kwargs = merged.model_dump(exclude={"splitter"}, exclude_none=True)
    separators = kwargs.get("separators")
    if separators == list(DEFAULT_RECURSIVE_SEPARATORS):
        kwargs["separators"] = DEFAULT_RECURSIVE_SEPARATORS
    if embedding is not None:
        kwargs["embedding"] = embedding
    if name == "KeywordSplitter" and not kwargs.get("patterns"):
        raise ServerConfigurationError(
            "KeywordSplitter requires patterns on the splitter object or in kwargs."
        )
    try:
        return splitter_cls(**kwargs)
    except TypeError as error:
        raise ServerConfigurationError(str(error)) from error
    except Exception as error:
        from splitter_mr.schema.exceptions import SplitterConfigException

        if isinstance(error, SplitterConfigException):
            raise ServerConfigurationError(str(error)) from error
        raise


def reader_read_kwargs(config: ReaderConfiguration) -> dict[str, Any]:
    """Return constructor-excluded kwargs that belong on ``reader.read``.

    Reader JSON configs now hold constructor-only fields, so this is empty
    unless a future constructor/read split is added.

    Args:
        config: Discriminated reader configuration.

    Returns:
        Keyword arguments forwarded to ``read``, excluding constructor fields.
    """
    excluded = {"reader"}
    if isinstance(config, TextractReaderConfiguration):
        excluded.update({"region_name", "profile_name"})
    return config.model_dump(exclude=excluded, exclude_none=True)


def list_components() -> ComponentCatalogResponse:
    """Build the reader, splitter, vision-model, and embedding catalog.

    Returns:
        Typed component catalog including ``SemanticSplitter``.
    """
    readers = [
        ReaderDescriptor(
            name=ReaderName(name),
            purpose=READER_PURPOSES[name],
            extra=EXTRA_BY_NAME.get(name),
            status=(
                ComponentStatus.AVAILABLE
                if _is_reader_available(name)
                else ComponentStatus.MISSING_EXTRA
            ),
            available=_is_reader_available(name),
            supported_source_types=[
                SourceType(item) for item in sorted(READER_SOURCE_TYPES[name])
            ],
            compatible_vision_models=[
                VisionModelName(item) for item in sorted(READER_VISION_MODELS[name])
            ],
            limitation=READER_LIMITATIONS.get(name),
            configuration_schema=READER_CONFIG_SCHEMA[name],
        )
        for name in REGISTRY
    ]
    splitters = [
        SplitterDescriptor(
            name=name,
            purpose=SPLITTER_PURPOSES[name],
            extra=EMBEDDING_EXTRA if name == SEMANTIC_SPLITTER_NAME else None,
            status=ComponentStatus.AVAILABLE,
            available=True,
            supported=True,
            limitation=(
                "Requires a top-level embedding object (splitter-mr[multimodal])."
                if name == SEMANTIC_SPLITTER_NAME
                else None
            ),
            configuration_schema=f"{name}Configuration",
        )
        for name in SPLITTER_FACTORIES
    ]
    vision_models = [
        VisionModelDescriptor(
            name=model_name,
            purpose=VISION_MODEL_PURPOSES[model_name.value],
            extra=VISION_MODEL_EXTRA,
            status=(
                ComponentStatus.AVAILABLE
                if _is_vision_model_available(model_name.value)
                else ComponentStatus.MISSING_EXTRA
            ),
            available=_is_vision_model_available(model_name.value),
            compatible_readers=[
                ReaderName(reader)
                for reader, models in READER_VISION_MODELS.items()
                if model_name.value in models
            ],
            limitation=VISION_MODEL_LIMITATIONS.get(model_name.value),
            configuration_schema=VISION_MODEL_CONFIG_SCHEMA[model_name.value],
        )
        for model_name in VisionModelName
    ]
    embeddings = [
        EmbeddingDescriptor(
            name=embedding_name,
            purpose=EMBEDDING_PURPOSES[embedding_name.value],
            extra=EMBEDDING_EXTRA,
            status=(
                ComponentStatus.AVAILABLE
                if _is_embedding_available(embedding_name.value)
                else ComponentStatus.MISSING_EXTRA
            ),
            available=_is_embedding_available(embedding_name.value),
            limitation=EMBEDDING_LIMITATIONS.get(embedding_name.value),
            configuration_schema=EMBEDDING_CONFIG_SCHEMA[embedding_name.value],
        )
        for embedding_name in EmbeddingName
    ]
    return ComponentCatalogResponse(
        readers=readers,
        splitters=splitters,
        vision_models=vision_models,
        embeddings=embeddings,
    )


def _merge_splitter_kwargs(
    config: SplitterConfiguration,
    extra_kwargs: dict[str, Any] | None,
) -> SplitterConfiguration:
    """Merge extra constructor kwargs into a splitter configuration.

    Args:
        config: Discriminated splitter configuration.
        extra_kwargs: Additional constructor arguments.

    Returns:
        Re-validated configuration after the merge.

    Raises:
        ServerConfigurationError: If merged fields fail schema validation.
    """
    if not extra_kwargs:
        return config
    payload = config.model_dump()
    payload.update(extra_kwargs)
    try:
        return type(config).model_validate(payload)
    except ValidationError as error:
        raise ServerConfigurationError(str(error)) from error


def _secret_constructor_kwargs(
    config: VisionModelConfiguration | EmbeddingConfiguration,
    exclude: set[str],
) -> dict[str, Any]:
    """Return constructor kwargs with secrets unwrapped.

    Args:
        config: Discriminated vision-model or embedding configuration.
        exclude: Field names omitted from the constructor call.

    Returns:
        Keyword arguments passed to the provider constructor.
    """
    kwargs: dict[str, Any] = {}
    for name, value in config.model_dump(exclude=exclude, exclude_none=True).items():
        field = getattr(config, name)
        if isinstance(field, SecretStr):
            kwargs[name] = field.get_secret_value()
        else:
            kwargs[name] = value
    return kwargs


def _reader_constructor_kwargs(
    config: ReaderConfiguration,
    vision_model: Any | None,
) -> dict[str, Any]:
    """Return constructor kwargs for the selected reader.

    Args:
        config: Discriminated reader configuration.
        vision_model: Optional constructed vision model.

    Returns:
        Keyword arguments passed to the reader constructor.
    """
    kwargs: dict[str, Any] = {}
    if isinstance(config, TextractReaderConfiguration):
        kwargs.update(
            config.model_dump(
                include={"region_name", "profile_name"},
                exclude_none=True,
            )
        )
    if vision_model is not None:
        kwargs["model"] = vision_model
    return kwargs
