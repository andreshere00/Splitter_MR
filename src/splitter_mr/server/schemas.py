"""Typed request, response, and discovery models for the SplitterMR server."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from splitter_mr.schema.constants import (
    ALLOWED_HEADERS_LITERAL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    DEFAULT_AZURE_OPENAI_VISION_DEPLOYMENT,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_VISION_MODEL,
    DEFAULT_GROK_VISION_MODEL,
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_KEYWORD_DELIMITER_POS,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_PARAGRAPH_SEPARATORS,
    DEFAULT_RECURSIVE_SEPARATORS,
    DEFAULT_SENTENCE_SEPARATORS,
    DEFAULT_TOKEN_LANGUAGE,
    DEFAULT_TOKENIZER,
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
    SUPPORTED_KEYWORD_DELIMITERS_LITERAL,
    BreakpointThresholdType,
)
from splitter_mr.schema.models import ReaderOutput

from .enums import (
    ComponentStatus,
    EmbeddingName,
    HealthStatus,
    ReaderName,
    SourceType,
    VisionModelName,
)

HeaderName = ALLOWED_HEADERS_LITERAL
KeywordDelimiter = SUPPORTED_KEYWORD_DELIMITERS_LITERAL

RESERVED_READ_KWARGS: frozenset[str] = frozenset({"file_path", "model"})
RESERVED_SPLIT_KWARGS: frozenset[str] = frozenset({"splitter", "embedding"})

READER_SOURCE_TYPES: dict[str, frozenset[str]] = {
    ReaderName.VANILLA.value: frozenset(
        {
            SourceType.TEXT.value,
            SourceType.JSON.value,
            SourceType.FILE.value,
            SourceType.URL.value,
        }
    ),
    ReaderName.MARKITDOWN.value: frozenset(
        {SourceType.FILE.value, SourceType.URL.value}
    ),
    ReaderName.DOCLING.value: frozenset({SourceType.FILE.value, SourceType.URL.value}),
    ReaderName.TEXTRACT.value: frozenset({SourceType.FILE.value}),
}

MARKITDOWN_COMPATIBLE_VISION_MODELS: frozenset[str] = frozenset(
    {
        VisionModelName.OPENAI.value,
        VisionModelName.AZURE_OPENAI.value,
        VisionModelName.ANTHROPIC.value,
        VisionModelName.OPENROUTER.value,
    }
)

READER_VISION_MODELS: dict[str, frozenset[str]] = {
    ReaderName.VANILLA.value: frozenset(item.value for item in VisionModelName),
    ReaderName.MARKITDOWN.value: MARKITDOWN_COMPATIBLE_VISION_MODELS,
    ReaderName.DOCLING.value: frozenset(item.value for item in VisionModelName),
    ReaderName.TEXTRACT.value: frozenset(),
}

DEFAULT_AZURE_OPENAI_API_VERSION: str = "2025-04-14-preview"


class SchemaBase(BaseModel):
    """Base model that forbids unknown fields so Swagger stays exact."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _secret_api_key_field(env_name: str) -> Any:
    """Build an optional write-only API key field.

    Args:
        env_name: Environment variable used when the field is omitted.

    Returns:
        A Pydantic ``Field`` configured for optional secret input.
    """
    return Field(
        default=None,
        description=(
            f"Optional API key. Omit to use {env_name}. Inline keys may be "
            "captured by proxies, MCP clients, or traces."
        ),
        json_schema_extra={"writeOnly": True},
    )


class OpenAIVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``OpenAIVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"model": "OpenAIVisionModel", "model_name": DEFAULT_OPENAI_MODEL}
            ]
        },
    )

    model: Literal["OpenAIVisionModel"] = Field(
        default="OpenAIVisionModel",
        description="Selects OpenAIVisionModel. Requires splitter-mr[multimodal].",
        examples=["OpenAIVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("OPENAI_API_KEY")
    model_name: str = Field(
        default=DEFAULT_OPENAI_MODEL,
        min_length=1,
        description="Vision-capable OpenAI model name.",
        examples=[DEFAULT_OPENAI_MODEL],
    )


class AzureOpenAIVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``AzureOpenAIVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "AzureOpenAIVisionModel",
                    "azure_deployment": DEFAULT_AZURE_OPENAI_VISION_DEPLOYMENT,
                }
            ]
        },
    )

    model: Literal["AzureOpenAIVisionModel"] = Field(
        default="AzureOpenAIVisionModel",
        description="Selects AzureOpenAIVisionModel. Requires splitter-mr[multimodal].",
        examples=["AzureOpenAIVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("AZURE_OPENAI_API_KEY")
    azure_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint. Omit to use AZURE_OPENAI_ENDPOINT.",
        examples=["https://example.openai.azure.com"],
    )
    azure_deployment: str | None = Field(
        default=None,
        description=(
            "Azure deployment name. Omit to use AZURE_OPENAI_DEPLOYMENT or "
            f"{DEFAULT_AZURE_OPENAI_VISION_DEPLOYMENT}."
        ),
        examples=[DEFAULT_AZURE_OPENAI_VISION_DEPLOYMENT],
    )
    api_version: str | None = Field(
        default=None,
        description=(
            "Azure OpenAI API version. Omit to use AZURE_OPENAI_API_VERSION or "
            f"{DEFAULT_AZURE_OPENAI_API_VERSION}."
        ),
        examples=[DEFAULT_AZURE_OPENAI_API_VERSION],
    )


class AnthropicVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``AnthropicVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"model": "AnthropicVisionModel", "model_name": DEFAULT_ANTHROPIC_MODEL}
            ]
        },
    )

    model: Literal["AnthropicVisionModel"] = Field(
        default="AnthropicVisionModel",
        description="Selects AnthropicVisionModel. Requires splitter-mr[multimodal].",
        examples=["AnthropicVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("ANTHROPIC_API_KEY")
    model_name: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        min_length=1,
        description="Vision-capable Claude model name.",
        examples=[DEFAULT_ANTHROPIC_MODEL],
    )


class OpenRouterVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``OpenRouterVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "OpenRouterVisionModel",
                    "model_name": DEFAULT_OPENROUTER_MODEL,
                }
            ]
        },
    )

    model: Literal["OpenRouterVisionModel"] = Field(
        default="OpenRouterVisionModel",
        description="Selects OpenRouterVisionModel. Requires splitter-mr[multimodal].",
        examples=["OpenRouterVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("OPENROUTER_API_KEY")
    model_name: str = Field(
        default=DEFAULT_OPENROUTER_MODEL,
        min_length=1,
        description="OpenRouter vision model slug.",
        examples=[DEFAULT_OPENROUTER_MODEL],
    )
    site_url: str | None = Field(
        default=None,
        description="Optional HTTP-Referer. Omit to use OPENROUTER_SITE_URL.",
        examples=["https://example.com"],
    )
    app_name: str | None = Field(
        default=None,
        description="Optional X-OpenRouter-Title. Omit to use OPENROUTER_APP_NAME.",
        examples=["SplitterMR"],
    )


class GrokVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``GrokVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"model": "GrokVisionModel", "model_name": DEFAULT_GROK_VISION_MODEL}
            ]
        },
    )

    model: Literal["GrokVisionModel"] = Field(
        default="GrokVisionModel",
        description="Selects GrokVisionModel. Requires splitter-mr[multimodal].",
        examples=["GrokVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("XAI_API_KEY")
    model_name: str = Field(
        default=DEFAULT_GROK_VISION_MODEL,
        min_length=1,
        description="xAI Grok vision model name.",
        examples=[DEFAULT_GROK_VISION_MODEL],
    )


class GeminiVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``GeminiVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "GeminiVisionModel",
                    "model_name": DEFAULT_GEMINI_VISION_MODEL,
                }
            ]
        },
    )

    model: Literal["GeminiVisionModel"] = Field(
        default="GeminiVisionModel",
        description="Selects GeminiVisionModel. Requires splitter-mr[multimodal].",
        examples=["GeminiVisionModel"],
    )
    api_key: SecretStr | None = _secret_api_key_field("GEMINI_API_KEY")
    model_name: str = Field(
        default=DEFAULT_GEMINI_VISION_MODEL,
        min_length=1,
        description="Vision-capable Gemini model name.",
        examples=[DEFAULT_GEMINI_VISION_MODEL],
    )


class HuggingFaceVisionModelConfiguration(SchemaBase):
    """Constructor configuration for ``HuggingFaceVisionModel``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "HuggingFaceVisionModel",
                    "model_name": DEFAULT_HUGGINGFACE_MODEL,
                }
            ]
        },
    )

    model: Literal["HuggingFaceVisionModel"] = Field(
        default="HuggingFaceVisionModel",
        description="Selects HuggingFaceVisionModel. Requires splitter-mr[multimodal].",
        examples=["HuggingFaceVisionModel"],
    )
    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        min_length=1,
        description="Hugging Face Hub model id or local path.",
        examples=[DEFAULT_HUGGINGFACE_MODEL],
    )


VisionModelConfiguration = Annotated[
    Union[
        OpenAIVisionModelConfiguration,
        AzureOpenAIVisionModelConfiguration,
        AnthropicVisionModelConfiguration,
        OpenRouterVisionModelConfiguration,
        GrokVisionModelConfiguration,
        GeminiVisionModelConfiguration,
        HuggingFaceVisionModelConfiguration,
    ],
    Field(discriminator="model"),
]


class OpenAIEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``OpenAIEmbedding``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "OpenAIEmbedding",
                    "model_name": DEFAULT_OPENAI_EMBEDDING_MODEL,
                }
            ]
        },
    )

    embedding: Literal["OpenAIEmbedding"] = Field(
        default="OpenAIEmbedding",
        description="Selects OpenAIEmbedding. Requires splitter-mr[multimodal].",
        examples=["OpenAIEmbedding"],
    )
    api_key: SecretStr | None = _secret_api_key_field("OPENAI_API_KEY")
    model_name: str = Field(
        default=DEFAULT_OPENAI_EMBEDDING_MODEL,
        min_length=1,
        description="OpenAI embedding model name.",
        examples=[DEFAULT_OPENAI_EMBEDDING_MODEL],
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Optional tiktoken encoding override.",
        examples=["cl100k_base"],
    )


class AzureOpenAIEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``AzureOpenAIEmbedding``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "AzureOpenAIEmbedding",
                    "azure_deployment": DEFAULT_AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                }
            ]
        },
    )

    embedding: Literal["AzureOpenAIEmbedding"] = Field(
        default="AzureOpenAIEmbedding",
        description="Selects AzureOpenAIEmbedding. Requires splitter-mr[multimodal].",
        examples=["AzureOpenAIEmbedding"],
    )
    api_key: SecretStr | None = _secret_api_key_field("AZURE_OPENAI_API_KEY")
    model_name: str | None = Field(
        default=None,
        description="Optional model name used when azure_deployment is omitted.",
        examples=[DEFAULT_AZURE_OPENAI_EMBEDDING_DEPLOYMENT],
    )
    azure_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint. Omit to use AZURE_OPENAI_ENDPOINT.",
        examples=["https://example.openai.azure.com"],
    )
    azure_deployment: str | None = Field(
        default=None,
        description=(
            "Azure embedding deployment. Omit to use AZURE_OPENAI_DEPLOYMENT or "
            f"{DEFAULT_AZURE_OPENAI_EMBEDDING_DEPLOYMENT}."
        ),
        examples=[DEFAULT_AZURE_OPENAI_EMBEDDING_DEPLOYMENT],
    )
    api_version: str | None = Field(
        default=None,
        description=(
            "Azure OpenAI API version. Omit to use AZURE_OPENAI_API_VERSION or "
            f"{DEFAULT_AZURE_OPENAI_API_VERSION}."
        ),
        examples=[DEFAULT_AZURE_OPENAI_API_VERSION],
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Optional tiktoken encoding override.",
        examples=["cl100k_base"],
    )


class OpenRouterEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``OpenRouterEmbedding``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "OpenRouterEmbedding",
                    "model_name": DEFAULT_OPENROUTER_EMBEDDING_MODEL,
                }
            ]
        },
    )

    embedding: Literal["OpenRouterEmbedding"] = Field(
        default="OpenRouterEmbedding",
        description="Selects OpenRouterEmbedding. Requires splitter-mr[multimodal].",
        examples=["OpenRouterEmbedding"],
    )
    api_key: SecretStr | None = _secret_api_key_field("OPENROUTER_API_KEY")
    model_name: str = Field(
        default=DEFAULT_OPENROUTER_EMBEDDING_MODEL,
        min_length=1,
        description="OpenRouter embedding model slug.",
        examples=[DEFAULT_OPENROUTER_EMBEDDING_MODEL],
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Optional tiktoken encoding override.",
        examples=["cl100k_base"],
    )
    site_url: str | None = Field(
        default=None,
        description="Optional HTTP-Referer. Omit to use OPENROUTER_SITE_URL.",
        examples=["https://example.com"],
    )
    app_name: str | None = Field(
        default=None,
        description="Optional X-OpenRouter-Title. Omit to use OPENROUTER_APP_NAME.",
        examples=["SplitterMR"],
    )


class GeminiEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``GeminiEmbedding``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "GeminiEmbedding",
                    "model_name": DEFAULT_GEMINI_EMBEDDING_MODEL,
                }
            ]
        },
    )

    embedding: Literal["GeminiEmbedding"] = Field(
        default="GeminiEmbedding",
        description="Selects GeminiEmbedding. Requires splitter-mr[multimodal].",
        examples=["GeminiEmbedding"],
    )
    api_key: SecretStr | None = _secret_api_key_field("GEMINI_API_KEY")
    model_name: str = Field(
        default=DEFAULT_GEMINI_EMBEDDING_MODEL,
        min_length=1,
        description="Gemini embedding model name.",
        examples=[DEFAULT_GEMINI_EMBEDDING_MODEL],
    )


class HuggingFaceEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``HuggingFaceEmbedding``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "HuggingFaceEmbedding",
                    "model_name": DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
                    "device": "cpu",
                }
            ]
        },
    )

    embedding: Literal["HuggingFaceEmbedding"] = Field(
        default="HuggingFaceEmbedding",
        description="Selects HuggingFaceEmbedding. Requires splitter-mr[multimodal].",
        examples=["HuggingFaceEmbedding"],
    )
    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
        min_length=1,
        description="Sentence-Transformers model id or local path.",
        examples=[DEFAULT_HUGGINGFACE_EMBEDDING_MODEL],
    )
    device: str | None = Field(
        default="cpu",
        description="Torch device spec such as cpu, cuda, or mps.",
        examples=["cpu"],
    )
    normalize: bool = Field(
        default=True,
        description="If true, return L2-normalized embeddings.",
        examples=[True],
    )
    enforce_max_length: bool = Field(
        default=False,
        description="If true, reject inputs longer than the model sequence length.",
        examples=[False],
    )


class AnthropicEmbeddingConfiguration(SchemaBase):
    """Constructor configuration for ``AnthropicEmbedding`` (Voyage AI)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "embedding": "AnthropicEmbedding",
                    "model_name": DEFAULT_VOYAGE_EMBEDDING_MODEL,
                }
            ]
        },
    )

    embedding: Literal["AnthropicEmbedding"] = Field(
        default="AnthropicEmbedding",
        description=(
            "Selects AnthropicEmbedding (Voyage AI). Requires splitter-mr[multimodal]."
        ),
        examples=["AnthropicEmbedding"],
    )
    api_key: SecretStr | None = _secret_api_key_field("VOYAGE_API_KEY")
    model_name: str = Field(
        default=DEFAULT_VOYAGE_EMBEDDING_MODEL,
        min_length=1,
        description="Voyage embedding model name.",
        examples=[DEFAULT_VOYAGE_EMBEDDING_MODEL],
    )
    default_input_type: str | None = Field(
        default="document",
        description="Default Voyage input_type: document or query.",
        examples=["document"],
    )


EmbeddingConfiguration = Annotated[
    Union[
        OpenAIEmbeddingConfiguration,
        AzureOpenAIEmbeddingConfiguration,
        OpenRouterEmbeddingConfiguration,
        GeminiEmbeddingConfiguration,
        HuggingFaceEmbeddingConfiguration,
        AnthropicEmbeddingConfiguration,
    ],
    Field(discriminator="embedding"),
]


class VanillaReaderConfiguration(SchemaBase):
    """Constructor configuration for ``VanillaReader``.

    Read-time options such as ``html_to_markdown`` belong in request ``kwargs``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"reader": "VanillaReader"}]},
    )

    reader: Literal["VanillaReader"] = Field(
        default="VanillaReader",
        description="Selects VanillaReader, the core multi-format reader.",
        examples=["VanillaReader"],
    )


class MarkItDownReaderConfiguration(SchemaBase):
    """Constructor configuration for ``MarkItDownReader``.

    Requires the ``markitdown`` extra. Compatible vision models must expose an
    OpenAI client: OpenAI, Azure OpenAI, Anthropic, or OpenRouter.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"reader": "MarkItDownReader"}]},
    )

    reader: Literal["MarkItDownReader"] = Field(
        default="MarkItDownReader",
        description="Selects MarkItDownReader. Requires splitter-mr[markitdown].",
        examples=["MarkItDownReader"],
    )


class DoclingReaderConfiguration(SchemaBase):
    """Constructor configuration for ``DoclingReader``.

    Requires the ``docling`` extra. Read-time options belong in request ``kwargs``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"reader": "DoclingReader"}]},
    )

    reader: Literal["DoclingReader"] = Field(
        default="DoclingReader",
        description="Selects DoclingReader. Requires splitter-mr[docling].",
        examples=["DoclingReader"],
    )


class TextractReaderConfiguration(SchemaBase):
    """Constructor configuration for ``TextractReader``.

    Requires the ``textract`` extra. Vision models are not supported.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"reader": "TextractReader", "region_name": "us-east-1"}]
        },
    )

    reader: Literal["TextractReader"] = Field(
        default="TextractReader",
        description="Selects TextractReader. Requires splitter-mr[textract].",
        examples=["TextractReader"],
    )
    region_name: str | None = Field(
        default=None,
        description="AWS region override for the Textract client.",
        examples=["us-east-1"],
    )
    profile_name: str | None = Field(
        default=None,
        description="AWS shared-credentials profile name.",
        examples=["default"],
    )


ReaderConfiguration = Annotated[
    Union[
        VanillaReaderConfiguration,
        MarkItDownReaderConfiguration,
        DoclingReaderConfiguration,
        TextractReaderConfiguration,
    ],
    Field(discriminator="reader"),
]


def _overlap_ge_zero(chunk_overlap: int | float) -> int | float:
    """Reject negative overlap values.

    Args:
        chunk_overlap: Overlap as an integer count or a fraction.

    Returns:
        The validated overlap value.

    Raises:
        ValueError: If the overlap is negative or a fraction is not in ``[0, 1)``.
    """
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, (int, float)):
        raise ValueError("chunk_overlap must be an int or float")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if isinstance(chunk_overlap, float) and not (0.0 <= chunk_overlap < 1.0):
        raise ValueError("float chunk_overlap must be in [0.0, 1.0)")
    return chunk_overlap


class CharacterSplitterConfiguration(SchemaBase):
    """Fixed-size character chunks with optional overlap."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "CharacterSplitter",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                }
            ]
        },
    )

    splitter: Literal["CharacterSplitter"] = Field(
        default="CharacterSplitter",
        description="Selects CharacterSplitter.",
        examples=["CharacterSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of characters per chunk.",
        examples=[1000],
    )
    chunk_overlap: int | float = Field(
        default=0,
        description=(
            "Overlap between consecutive chunks. An integer is a character count "
            "smaller than chunk_size. A float is a fraction in [0.0, 1.0)."
        ),
        examples=[0],
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> CharacterSplitterConfiguration:
        """Validate overlap against chunk size.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If overlap is invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if isinstance(self.chunk_overlap, int):
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(
                    "integer chunk_overlap must be smaller than chunk_size"
                )
        return self


class WordSplitterConfiguration(SchemaBase):
    """Word-count chunks with optional overlap."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"splitter": "WordSplitter", "chunk_size": 20, "chunk_overlap": 2}
            ]
        },
    )

    splitter: Literal["WordSplitter"] = Field(
        default="WordSplitter",
        description="Selects WordSplitter.",
        examples=["WordSplitter"],
    )
    chunk_size: int = Field(
        default=5,
        ge=1,
        description="Maximum number of words per chunk.",
        examples=[5],
    )
    chunk_overlap: int | float = Field(
        default=0,
        description=(
            "Overlap in words (int) or as a fraction of chunk_size (float in [0, 1))."
        ),
        examples=[0],
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> WordSplitterConfiguration:
        """Validate overlap against chunk size.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If overlap is invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if isinstance(self.chunk_overlap, int):
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(
                    "integer chunk_overlap must be smaller than chunk_size"
                )
        return self


class SentenceSplitterConfiguration(SchemaBase):
    """Sentence-count chunks with optional word overlap."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "SentenceSplitter",
                    "chunk_size": 5,
                    "chunk_overlap": 0,
                }
            ]
        },
    )

    splitter: Literal["SentenceSplitter"] = Field(
        default="SentenceSplitter",
        description="Selects SentenceSplitter.",
        examples=["SentenceSplitter"],
    )
    chunk_size: int = Field(
        default=5,
        ge=1,
        description="Maximum number of sentences per chunk.",
        examples=[5],
    )
    chunk_overlap: int | float = Field(
        default=0,
        description="Overlap in words (int) or as a fraction (float in [0, 1)).",
        examples=[0],
    )
    separators: str | list[str] = Field(
        default=DEFAULT_SENTENCE_SEPARATORS,
        description=(
            "Sentence-boundary regex or a list of separator strings joined into a "
            "regex pattern."
        ),
        examples=[DEFAULT_SENTENCE_SEPARATORS],
    )

    @model_validator(mode="after")
    def validate_fields(self) -> SentenceSplitterConfiguration:
        """Validate overlap and separators.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If overlap or separators are invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if isinstance(self.separators, str):
            if not self.separators:
                raise ValueError("separators must be a non-empty string")
        elif not self.separators or any(not item for item in self.separators):
            raise ValueError("separators must contain at least one non-empty string")
        return self


class ParagraphSplitterConfiguration(SchemaBase):
    """Paragraph-count chunks with optional word overlap."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "ParagraphSplitter",
                    "chunk_size": 3,
                    "line_break": "\n",
                }
            ]
        },
    )

    splitter: Literal["ParagraphSplitter"] = Field(
        default="ParagraphSplitter",
        description="Selects ParagraphSplitter.",
        examples=["ParagraphSplitter"],
    )
    chunk_size: int = Field(
        default=3,
        ge=1,
        description="Maximum number of paragraphs per chunk.",
        examples=[3],
    )
    chunk_overlap: int | float = Field(
        default=0,
        description="Overlap in words (int) or as a fraction (float in [0, 1)).",
        examples=[0],
    )
    line_break: str | list[str] = Field(
        default=DEFAULT_PARAGRAPH_SEPARATORS,
        description="Delimiter or delimiters used to detect paragraphs.",
        examples=["\n"],
    )

    @model_validator(mode="after")
    def validate_fields(self) -> ParagraphSplitterConfiguration:
        """Validate overlap and paragraph delimiters.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If overlap or line_break are invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if isinstance(self.line_break, str):
            if not self.line_break:
                raise ValueError("line_break must be a non-empty string")
        elif not self.line_break or any(not item for item in self.line_break):
            raise ValueError("line_break must contain at least one non-empty string")
        return self


class RecursiveCharacterSplitterConfiguration(SchemaBase):
    """Recursive separator hierarchy until chunks fit ``chunk_size``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "RecursiveCharacterSplitter",
                    "chunk_size": 1000,
                    "chunk_overlap": 0.1,
                }
            ]
        },
    )

    splitter: Literal["RecursiveCharacterSplitter"] = Field(
        default="RecursiveCharacterSplitter",
        description="Selects RecursiveCharacterSplitter.",
        examples=["RecursiveCharacterSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Approximate maximum characters per chunk.",
        examples=[1000],
    )
    chunk_overlap: int | float = Field(
        default=0.1,
        description=(
            "Overlap as a character count (int) or a fraction of chunk_size "
            "(float in [0, 1))."
        ),
        examples=[0.1],
    )
    separators: str | list[str] = Field(
        default=list(DEFAULT_RECURSIVE_SEPARATORS),
        description="Separator hierarchy tried from coarsest to finest.",
        examples=[["\n\n", "\n", " ", ""]],
    )

    @model_validator(mode="after")
    def validate_fields(self) -> RecursiveCharacterSplitterConfiguration:
        """Validate overlap and separators.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If overlap or separators are invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if isinstance(self.chunk_overlap, int):
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError(
                    "integer chunk_overlap must be smaller than chunk_size"
                )
        if isinstance(self.separators, str):
            if self.separators == "":
                raise ValueError("separators string must be non-empty")
        elif not isinstance(self.separators, list):
            raise ValueError("separators must be a string or a list of strings")
        return self


class KeywordSplitterConfiguration(SchemaBase):
    """Split around one or more regular-expression keyword boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "KeywordSplitter",
                    "patterns": ["CHAPTER\\s+\\d+"],
                    "include_delimiters": "before",
                    "chunk_size": 100000,
                }
            ]
        },
    )

    splitter: Literal["KeywordSplitter"] = Field(
        default="KeywordSplitter",
        description="Selects KeywordSplitter.",
        examples=["KeywordSplitter"],
    )
    patterns: list[str] | dict[str, str] | None = Field(
        default=None,
        description=(
            "Regex patterns as a list, or a mapping of name to pattern used in "
            "metadata counts. Required on this object or in kwargs."
        ),
        examples=[["CHAPTER\\s+\\d+"]],
    )
    flags: int = Field(
        default=0,
        ge=0,
        description=(
            "Python ``re`` flags combined with bitwise OR. Example: IGNORECASE is 2."
        ),
        examples=[0],
    )
    include_delimiters: KeywordDelimiter = Field(
        default=DEFAULT_KEYWORD_DELIMITER_POS,
        description="Where to attach the matched delimiter: none, before, after, or both.",
        examples=["before"],
    )
    chunk_size: int = Field(
        default=100000,
        ge=1,
        description="Soft maximum characters per chunk after keyword splitting.",
        examples=[100000],
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> KeywordSplitterConfiguration:
        """Reject empty pattern collections.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If no patterns are provided.
        """
        if self.patterns is None:
            return self
        if not self.patterns:
            raise ValueError("patterns must contain at least one regex")
        return self


class TokenSplitterConfiguration(SchemaBase):
    """Token-count chunks using tiktoken, spaCy, or NLTK."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "TokenSplitter",
                    "chunk_size": 256,
                    "model_name": DEFAULT_TOKENIZER,
                    "language": "english",
                }
            ]
        },
    )

    splitter: Literal["TokenSplitter"] = Field(
        default="TokenSplitter",
        description="Selects TokenSplitter.",
        examples=["TokenSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens per chunk.",
        examples=[1000],
    )
    model_name: str = Field(
        default=DEFAULT_TOKENIZER,
        min_length=1,
        description="Tokenizer spec in the form tokenizer/model, e.g. tiktoken/cl100k_base.",
        examples=[DEFAULT_TOKENIZER],
    )
    language: str = Field(
        default=DEFAULT_TOKEN_LANGUAGE,
        min_length=1,
        description="Language code used by the NLTK tokenizer backend.",
        examples=["english"],
    )


class PagedSplitterConfiguration(SchemaBase):
    """Group pages using the reader page placeholder."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"splitter": "PagedSplitter", "chunk_size": 1, "chunk_overlap": 0}
            ]
        },
    )

    splitter: Literal["PagedSplitter"] = Field(
        default="PagedSplitter",
        description="Selects PagedSplitter. chunk_size is pages per chunk.",
        examples=["PagedSplitter"],
    )
    chunk_size: int = Field(
        default=1,
        ge=1,
        description="Number of pages per chunk.",
        examples=[1],
    )
    chunk_overlap: int = Field(
        default=0,
        ge=0,
        description="Number of overlapping pages between consecutive chunks.",
        examples=[0],
    )


class RowColumnSplitterConfiguration(SchemaBase):
    """Split tabular text by rows, columns, or character budget."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "RowColumnSplitter",
                    "num_rows": 20,
                    "num_cols": 0,
                    "chunk_overlap": 2,
                }
            ]
        },
    )

    splitter: Literal["RowColumnSplitter"] = Field(
        default="RowColumnSplitter",
        description="Selects RowColumnSplitter. num_rows and num_cols are exclusive.",
        examples=["RowColumnSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum characters per chunk when splitting by size.",
        examples=[1000],
    )
    num_rows: int = Field(
        default=0,
        ge=0,
        description="Rows per chunk. Mutually exclusive with num_cols. 0 disables.",
        examples=[20],
    )
    num_cols: int = Field(
        default=0,
        ge=0,
        description="Columns per chunk. Mutually exclusive with num_rows. 0 disables.",
        examples=[0],
    )
    chunk_overlap: int | float = Field(
        default=0,
        description="Overlap in rows/columns (int) or as a fraction (float in [0, 1)).",
        examples=[0],
    )

    @model_validator(mode="after")
    def validate_fields(self) -> RowColumnSplitterConfiguration:
        """Validate mutually exclusive row/column mode and overlap.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If both row and column modes are set or overlap is invalid.
        """
        _overlap_ge_zero(self.chunk_overlap)
        if self.num_rows and self.num_cols:
            raise ValueError("num_rows and num_cols are mutually exclusive")
        return self


class RecursiveJSONSplitterConfiguration(SchemaBase):
    """Recursively split JSON while preserving object structure."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "RecursiveJSONSplitter",
                    "chunk_size": 1000,
                    "min_chunk_size": 200,
                }
            ]
        },
    )

    splitter: Literal["RecursiveJSONSplitter"] = Field(
        default="RecursiveJSONSplitter",
        description="Selects RecursiveJSONSplitter.",
        examples=["RecursiveJSONSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum characters per JSON chunk.",
        examples=[1000],
    )
    min_chunk_size: int = Field(
        default=200,
        ge=1,
        description="Minimum characters per JSON chunk.",
        examples=[200],
    )


class HTMLTagSplitterConfiguration(SchemaBase):
    """Split HTML by a tag, with optional batching and Markdown conversion."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "HTMLTagSplitter",
                    "chunk_size": 1,
                    "tag": "div",
                    "batch": True,
                    "to_markdown": True,
                }
            ]
        },
    )

    splitter: Literal["HTMLTagSplitter"] = Field(
        default="HTMLTagSplitter",
        description="Selects HTMLTagSplitter.",
        examples=["HTMLTagSplitter"],
    )
    chunk_size: int = Field(
        default=1,
        ge=0,
        description=(
            "Maximum characters per batched chunk. 0 or 1 groups matching "
            "elements according to the splitter's batching rules."
        ),
        examples=[1],
    )
    tag: str | None = Field(
        default=None,
        description="HTML tag to split on. If omitted, the tag is auto-detected.",
        examples=["div"],
    )
    batch: bool = Field(
        default=True,
        description="If true, group elements up to chunk_size; otherwise one chunk each.",
        examples=[True],
    )
    to_markdown: bool = Field(
        default=True,
        description="If true, convert each emitted HTML chunk to Markdown.",
        examples=[True],
    )

    @model_validator(mode="after")
    def validate_tag(self) -> HTMLTagSplitterConfiguration:
        """Reject blank tag strings.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If tag is an empty string.
        """
        if self.tag is not None and not self.tag.strip():
            raise ValueError("tag must be a non-empty string when provided")
        return self


class HeaderSplitterConfiguration(SchemaBase):
    """Split Markdown or HTML by heading levels."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "HeaderSplitter",
                    "chunk_size": 1000,
                    "headers_to_split_on": ["Header 1", "Header 2"],
                    "group_header_with_content": True,
                }
            ]
        },
    )

    splitter: Literal["HeaderSplitter"] = Field(
        default="HeaderSplitter",
        description="Selects HeaderSplitter.",
        examples=["HeaderSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Size hint used only by the fallback recursive splitter.",
        examples=[1000],
    )
    headers_to_split_on: list[HeaderName] | None = Field(
        default=None,
        description=(
            "Semantic header names such as 'Header 1'. Defaults to all allowed headers."
        ),
        examples=[["Header 1", "Header 2"]],
    )
    group_header_with_content: bool = Field(
        default=True,
        description="If true, keep headings with the following content.",
        examples=[True],
    )


class CodeSplitterConfiguration(SchemaBase):
    """Language-aware source-code chunks."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"splitter": "CodeSplitter", "chunk_size": 1000, "language": "python"}
            ]
        },
    )

    splitter: Literal["CodeSplitter"] = Field(
        default="CodeSplitter",
        description="Selects CodeSplitter.",
        examples=["CodeSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum characters per code chunk.",
        examples=[1000],
    )
    language: str = Field(
        default="python",
        min_length=1,
        description=(
            "Programming language name understood by LangChain Language, "
            "for example python, java, js, go, or rust."
        ),
        examples=["python"],
    )


class SemanticSplitterConfiguration(SchemaBase):
    """Semantic similarity chunks. Requires a top-level ``embedding`` object."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "splitter": "SemanticSplitter",
                    "chunk_size": 1000,
                    "buffer_size": 1,
                    "breakpoint_threshold_type": "percentile",
                }
            ]
        },
    )

    splitter: Literal["SemanticSplitter"] = Field(
        default="SemanticSplitter",
        description=(
            "Selects SemanticSplitter. Pass embedding as a top-level request field."
        ),
        examples=["SemanticSplitter"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        description="Minimum characters per emitted chunk after merging short spans.",
        examples=[1000],
    )
    buffer_size: int = Field(
        default=1,
        ge=0,
        description="Neighboring sentences included on each side of the window.",
        examples=[1],
    )
    breakpoint_threshold_type: BreakpointThresholdType = Field(
        default="percentile",
        description=(
            "Breakpoint strategy: percentile, standard_deviation, interquartile, "
            "or gradient."
        ),
        examples=["percentile"],
    )
    breakpoint_threshold_amount: float | None = Field(
        default=None,
        description=(
            "Threshold strength for the chosen strategy. Omit to use the library "
            "default for that strategy."
        ),
        examples=[95.0],
    )
    number_of_chunks: int | None = Field(
        default=None,
        ge=1,
        description="Optional target number of chunks.",
        examples=[8],
    )


SplitterConfiguration = Annotated[
    Union[
        CharacterSplitterConfiguration,
        WordSplitterConfiguration,
        SentenceSplitterConfiguration,
        ParagraphSplitterConfiguration,
        RecursiveCharacterSplitterConfiguration,
        KeywordSplitterConfiguration,
        TokenSplitterConfiguration,
        PagedSplitterConfiguration,
        RowColumnSplitterConfiguration,
        RecursiveJSONSplitterConfiguration,
        HTMLTagSplitterConfiguration,
        HeaderSplitterConfiguration,
        CodeSplitterConfiguration,
        SemanticSplitterConfiguration,
    ],
    Field(discriminator="splitter"),
]


def _file_path_kind(file_path: str | dict[str, Any] | list[Any]) -> str | None:
    """Return a known source kind when it can be inferred without I/O.

    Args:
        file_path: Request ``file_path`` value.

    Returns:
        ``json`` for objects and arrays, otherwise ``None``.
    """
    if isinstance(file_path, (dict, list)):
        return SourceType.JSON.value
    return None


def _validate_reader_file_path(
    file_path: str | dict[str, Any] | list[Any],
    reader: ReaderConfiguration,
) -> None:
    """Reject reader and file_path combinations that cannot be handled.

    Args:
        file_path: Request ``file_path`` value.
        reader: Selected reader configuration.

    Raises:
        ValueError: If the reader does not accept this kind of input.
    """
    kind = _file_path_kind(file_path)
    if kind is None:
        return
    allowed = READER_SOURCE_TYPES[reader.reader]
    if kind not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"{reader.reader} does not support JSON file_path values. "
            f"Allowed kinds: {allowed_list}."
        )


def _validate_reader_model(
    reader: ReaderConfiguration,
    model: VisionModelConfiguration | None,
) -> None:
    """Reject unsupported reader and vision-model combinations.

    Args:
        reader: Selected reader configuration.
        model: Optional vision-model configuration.

    Raises:
        ValueError: If the reader cannot use the selected model.
    """
    if model is None:
        return
    allowed = READER_VISION_MODELS[reader.reader]
    if not allowed:
        raise ValueError(f"{reader.reader} does not accept a vision model.")
    if model.model not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"{reader.reader} does not support model={model.model!r}. "
            f"Allowed: {allowed_list}."
        )


def _validate_read_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject reserved keys that would collide with the method signature.

    Args:
        kwargs: Read-time keyword arguments.

    Raises:
        ValueError: If ``file_path`` or ``model`` is present.
    """
    reserved = RESERVED_READ_KWARGS.intersection(kwargs)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"kwargs must not include reserved keys: {names}. "
            "Pass file_path and model as top-level request fields."
        )


def _validate_split_kwargs(kwargs: dict[str, Any], field_name: str = "kwargs") -> None:
    """Reject reserved keys that belong on the split request, not in kwargs.

    Args:
        kwargs: Splitter constructor keyword arguments.
        field_name: Request field name used in the error message.

    Raises:
        ValueError: If ``splitter`` or ``embedding`` is present.
    """
    reserved = RESERVED_SPLIT_KWARGS.intersection(kwargs)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"{field_name} must not include reserved keys: {names}. "
            "Pass splitter and embedding as top-level request fields."
        )


def _validate_splitter_embedding(
    splitter: SplitterConfiguration,
    embedding: EmbeddingConfiguration | None,
) -> None:
    """Require embedding only for ``SemanticSplitter``.

    Args:
        splitter: Selected splitter configuration.
        embedding: Optional embedding constructor configuration.

    Raises:
        ValueError: If embedding is missing for SemanticSplitter or set otherwise.
    """
    is_semantic = splitter.splitter == "SemanticSplitter"
    if is_semantic and embedding is None:
        raise ValueError(
            "SemanticSplitter requires a top-level embedding configuration."
        )
    if not is_semantic and embedding is not None:
        raise ValueError("embedding is only valid when splitter is SemanticSplitter.")


class _ReadRequestMixin(SchemaBase):
    """Shared ``BaseReader.read`` fields for REST and MCP requests."""

    file_path: str | dict[str, Any] | list[Any] = Field(
        ...,
        description=(
            "Same argument as BaseReader.read. A server-local file path requires "
            "SPLITTER_MR_ALLOWED_ROOT. An http(s) URL requires "
            "SPLITTER_MR_ALLOW_URLS=true. Other strings are treated as raw text. "
            "JSON objects and arrays are accepted by VanillaReader."
        ),
        examples=["Lorem ipsum dolor sit amet."],
    )
    reader: ReaderConfiguration = Field(
        default_factory=VanillaReaderConfiguration,
        description=(
            "Reader implementation and constructor-only options. Read-time "
            "options belong in kwargs. Optional readers need their extras."
        ),
    )
    model: VisionModelConfiguration | None = Field(
        default=None,
        description=(
            "Optional BaseVisionModel constructor configuration. Credentials "
            "may be omitted to use provider environment variables. Requires "
            "splitter-mr[multimodal]. TextractReader does not accept a model."
        ),
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments forwarded to reader.read. Common "
            "keys include document_name, document_id, metadata, prompt, "
            "vlm_parameters, html_to_markdown, scan_pdf_pages, resolution, "
            "page_placeholder, and split_by_pages. Do not set file_path or model."
        ),
        examples=[{"document_name": "lorem.txt"}],
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(
        cls, value: str | dict[str, Any] | list[Any]
    ) -> str | dict[str, Any] | list[Any]:
        """Reject blank string inputs.

        Args:
            value: Incoming file_path value.

        Returns:
            The validated value.

        Raises:
            ValueError: If a string is empty or whitespace.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("file_path must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_read_contract(self) -> _ReadRequestMixin:
        """Validate kwargs, reader compatibility, and VLM-gated flags.

        Returns:
            The validated request.

        Raises:
            ValueError: If the combination is unsupported.
        """
        _validate_read_kwargs(self.kwargs)
        _validate_reader_file_path(self.file_path, self.reader)
        _validate_reader_model(self.reader, self.model)
        if self.kwargs.get("scan_pdf_pages") and self.model is None:
            raise ValueError("scan_pdf_pages requires a model configuration.")
        return self


class ReadDocumentRequest(_ReadRequestMixin):
    """Read one input with a selected reader and return ``ReaderOutput``.

    Attributes:
        file_path: Path, URL, raw string, or JSON value.
        reader: Discriminated reader constructor configuration.
        model: Optional vision-model constructor configuration.
        kwargs: Additional ``read`` keyword arguments.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "file_path": "Lorem ipsum dolor sit amet.",
                    "reader": {"reader": "VanillaReader"},
                    "kwargs": {"document_name": "lorem.txt"},
                }
            ]
        },
    )


class SplitDocumentRequest(SchemaBase):
    """Split a previously produced ``ReaderOutput``.

    Attributes:
        reader_output: Complete reader result, including metadata and IDs.
        splitter: Discriminated splitter configuration.
        embedding: Optional embedding constructor configuration.
        kwargs: Additional splitter constructor arguments.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "reader_output": {
                        "text": "Lorem ipsum dolor sit amet.",
                        "document_name": "lorem.txt",
                        "document_path": "",
                        "document_id": "732b9530-3e41-4a1a-a4ea-1d9d6fe815d3",
                        "conversion_method": "txt",
                        "reader_method": "vanilla",
                        "ocr_method": None,
                        "page_placeholder": None,
                        "metadata": {},
                    },
                    "splitter": {"splitter": "CharacterSplitter"},
                    "kwargs": {"chunk_size": 50, "chunk_overlap": 10},
                },
                {
                    "reader_output": {
                        "text": "CHAPTER 1 Once upon a time.",
                        "document_name": "story.txt",
                        "document_path": "",
                        "document_id": "732b9530-3e41-4a1a-a4ea-1d9d6fe815d3",
                        "conversion_method": "txt",
                        "reader_method": "vanilla",
                        "metadata": {},
                    },
                    "splitter": {
                        "splitter": "KeywordSplitter",
                        "patterns": ["CHAPTER\\s+\\d+"],
                    },
                    "kwargs": {
                        "include_delimiters": "before",
                        "chunk_size": 100000,
                    },
                },
                {
                    "reader_output": {
                        "text": "Lorem ipsum dolor sit amet.",
                        "document_name": "lorem.txt",
                        "document_path": "",
                        "document_id": "732b9530-3e41-4a1a-a4ea-1d9d6fe815d3",
                        "conversion_method": "txt",
                        "reader_method": "vanilla",
                        "metadata": {},
                    },
                    "splitter": {"splitter": "SemanticSplitter"},
                    "embedding": {
                        "embedding": "OpenAIEmbedding",
                        "model_name": DEFAULT_OPENAI_EMBEDDING_MODEL,
                    },
                    "kwargs": {
                        "chunk_size": 1000,
                        "buffer_size": 1,
                        "breakpoint_threshold_type": "percentile",
                    },
                },
            ]
        },
    )

    reader_output: ReaderOutput = Field(
        ...,
        description=(
            "Full ReaderOutput from a previous read_document call. Pass the object "
            "as-is so document IDs and metadata propagate into SplitterOutput."
        ),
    )
    splitter: SplitterConfiguration = Field(
        ...,
        description=(
            "Splitter implementation. Constructor parameters may live on this "
            "object or in kwargs. SemanticSplitter also requires embedding."
        ),
    )
    embedding: EmbeddingConfiguration | None = Field(
        default=None,
        description=(
            "Optional BaseEmbedding constructor configuration. Required for "
            "SemanticSplitter. Requires splitter-mr[multimodal]. Omit api_key "
            "to use the provider environment variable."
        ),
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional splitter constructor arguments merged into the selected "
            "splitter. Use this for fields beyond chunk_size and chunk_overlap, "
            "such as patterns, separators, include_delimiters, flags, "
            "headers_to_split_on, language, tag, num_rows, num_cols, "
            "min_chunk_size, buffer_size, and breakpoint_threshold_type. Do not "
            "set splitter or embedding here. Values in kwargs override the same "
            "keys on the splitter object."
        ),
        examples=[{"patterns": ["CHAPTER\\s+\\d+"], "include_delimiters": "before"}],
    )

    @model_validator(mode="after")
    def validate_split_contract(self) -> SplitDocumentRequest:
        """Validate kwargs and embedding compatibility.

        Returns:
            The validated request.

        Raises:
            ValueError: If reserved keys are set or embedding usage is invalid.
        """
        _validate_split_kwargs(self.kwargs)
        _validate_splitter_embedding(self.splitter, self.embedding)
        return self


class ReadAndSplitRequest(_ReadRequestMixin):
    """Read a source and split it in one call, returning ``SplitterOutput``.

    Attributes:
        file_path: Path, URL, raw string, or JSON value.
        reader: Discriminated reader constructor configuration.
        model: Optional vision-model constructor configuration.
        kwargs: Additional ``read`` keyword arguments.
        splitter: Discriminated splitter configuration.
        embedding: Optional embedding constructor configuration.
        splitter_kwargs: Additional splitter constructor arguments.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "file_path": "Lorem ipsum dolor sit amet.",
                    "reader": {"reader": "VanillaReader"},
                    "kwargs": {"document_name": "lorem.txt"},
                    "splitter": {"splitter": "RecursiveCharacterSplitter"},
                    "splitter_kwargs": {"chunk_size": 100, "chunk_overlap": 0.1},
                },
                {
                    "file_path": "/data/docs/manual.pdf",
                    "reader": {"reader": "VanillaReader"},
                    "model": {"model": "OpenAIVisionModel", "model_name": "gpt-4.1"},
                    "kwargs": {"prompt": "Extract the visible text."},
                    "splitter": {"splitter": "CharacterSplitter"},
                    "splitter_kwargs": {"chunk_size": 500, "chunk_overlap": 50},
                },
                {
                    "file_path": "CHAPTER 1 Once upon a time.",
                    "reader": {"reader": "VanillaReader"},
                    "kwargs": {"document_name": "story.txt"},
                    "splitter": {
                        "splitter": "KeywordSplitter",
                        "patterns": ["CHAPTER\\s+\\d+"],
                    },
                    "splitter_kwargs": {
                        "include_delimiters": "before",
                        "chunk_size": 100000,
                    },
                },
                {
                    "file_path": "Lorem ipsum dolor sit amet.",
                    "reader": {"reader": "VanillaReader"},
                    "kwargs": {"document_name": "lorem.txt"},
                    "splitter": {"splitter": "SemanticSplitter"},
                    "embedding": {
                        "embedding": "OpenAIEmbedding",
                        "model_name": DEFAULT_OPENAI_EMBEDDING_MODEL,
                    },
                    "splitter_kwargs": {"chunk_size": 1000, "buffer_size": 1},
                },
            ]
        },
    )

    splitter: SplitterConfiguration = Field(
        default_factory=RecursiveCharacterSplitterConfiguration,
        description=(
            "Splitter applied to the produced ReaderOutput. Defaults to "
            "RecursiveCharacterSplitter. Constructor parameters may live on "
            "this object or in splitter_kwargs."
        ),
    )
    embedding: EmbeddingConfiguration | None = Field(
        default=None,
        description=(
            "Optional BaseEmbedding constructor configuration. Required for "
            "SemanticSplitter. Requires splitter-mr[multimodal]. Omit api_key "
            "to use the provider environment variable."
        ),
    )
    splitter_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional splitter constructor arguments, equivalent to kwargs on "
            "POST /split. Use this for fields beyond chunk_size and "
            "chunk_overlap. Do not set splitter or embedding here. Values "
            "override the same keys on the splitter object. Read-time options "
            "belong in kwargs."
        ),
        examples=[{"chunk_size": 100, "separators": ["\\n\\n", "\\n", " ", ""]}],
    )

    @model_validator(mode="after")
    def validate_split_contract(self) -> ReadAndSplitRequest:
        """Validate splitter kwargs and embedding compatibility.

        Returns:
            The validated request.

        Raises:
            ValueError: If reserved keys are set or embedding usage is invalid.
        """
        _validate_split_kwargs(self.splitter_kwargs, field_name="splitter_kwargs")
        _validate_splitter_embedding(self.splitter, self.embedding)
        return self


class ValidationErrorDetail(SchemaBase):
    """Single Pydantic or request-validation issue."""

    loc: list[str | int] = Field(
        default_factory=list,
        description="Location of the invalid field.",
        examples=[["body", "kwargs", "document_name"]],
    )
    msg: str = Field(
        ...,
        description="Human-readable validation message.",
        examples=["String should have at least 1 character"],
    )
    type: str = Field(
        ...,
        description="Pydantic error type identifier.",
        examples=["string_too_short"],
    )


class ApiErrorResponse(SchemaBase):
    """Stable error payload returned by REST handlers and MCP tool errors."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "code": "access_denied",
                    "message": (
                        "File sources are disabled until SPLITTER_MR_ALLOWED_ROOT is set."
                    ),
                    "details": [],
                }
            ]
        },
    )

    code: str = Field(
        ...,
        description="Stable machine-readable error code.",
        examples=["access_denied"],
    )
    message: str = Field(
        ...,
        description="Human-readable error description. No credentials or source text.",
        examples=["File sources are disabled until SPLITTER_MR_ALLOWED_ROOT is set."],
    )
    details: list[ValidationErrorDetail] = Field(
        default_factory=list,
        description="Optional field-level validation details.",
    )


class HealthResponse(SchemaBase):
    """Liveness payload for container probes. Omits secrets and settings."""

    service: str = Field(
        ...,
        description="Service name.",
        examples=["splitter-mr-mcp"],
    )
    status: HealthStatus = Field(
        ...,
        description="Liveness status.",
        examples=["ok"],
    )
    version: str = Field(
        ...,
        description="Installed splitter-mr package version.",
        examples=["1.3.0"],
    )
    docs_url: str = Field(
        ...,
        description="Swagger UI path.",
        examples=["/docs"],
    )
    redoc_url: str = Field(
        ...,
        description="ReDoc path.",
        examples=["/redoc"],
    )
    openapi_url: str = Field(
        ...,
        description="OpenAPI schema path.",
        examples=["/openapi.json"],
    )
    mcp_path: str = Field(
        ...,
        description="Mounted Streamable HTTP MCP path.",
        examples=["/mcp"],
    )
    api_prefix: str = Field(
        ...,
        description="Versioned REST prefix.",
        examples=["/api/v1"],
    )


class ReaderDescriptor(SchemaBase):
    """Catalog entry describing one reader implementation."""

    name: ReaderName = Field(..., description="Public reader class name.")
    purpose: str = Field(..., description="When to choose this reader.")
    extra: str | None = Field(
        default=None,
        description="Optional extra required to import this reader.",
    )
    status: ComponentStatus = Field(..., description="Current availability.")
    available: bool = Field(..., description="Whether the reader can be instantiated.")
    supported_source_types: list[SourceType] = Field(
        ...,
        description="file_path kinds accepted by this reader through the server.",
    )
    compatible_vision_models: list[VisionModelName] = Field(
        default_factory=list,
        description="Vision models that can be attached to this reader.",
    )
    limitation: str | None = Field(
        default=None,
        description="Server-specific limitation.",
    )
    configuration_schema: str = Field(
        ...,
        description="OpenAPI component schema name for this reader's configuration.",
    )


class SplitterDescriptor(SchemaBase):
    """Catalog entry describing one splitter implementation."""

    name: str = Field(..., description="Public splitter class name.")
    purpose: str = Field(..., description="When to choose this splitter.")
    extra: str | None = Field(
        default=None,
        description="Optional extra required to import this splitter.",
    )
    status: ComponentStatus = Field(..., description="Current availability.")
    available: bool = Field(
        ..., description="Whether the splitter can be instantiated."
    )
    supported: bool = Field(
        ...,
        description="False when the server cannot expose this splitter in v1.",
    )
    limitation: str | None = Field(
        default=None,
        description="Server-specific limitation.",
    )
    configuration_schema: str | None = Field(
        default=None,
        description="OpenAPI component schema name, if the splitter is supported.",
    )


class VisionModelDescriptor(SchemaBase):
    """Catalog entry describing one vision-model implementation."""

    name: VisionModelName = Field(..., description="Public vision-model class name.")
    purpose: str = Field(..., description="When to choose this vision model.")
    extra: str = Field(
        default="multimodal",
        description="Optional extra required to import this vision model.",
    )
    status: ComponentStatus = Field(..., description="Current availability.")
    available: bool = Field(..., description="Whether the model can be instantiated.")
    compatible_readers: list[ReaderName] = Field(
        ...,
        description="Readers that accept this vision model over JSON.",
    )
    limitation: str | None = Field(
        default=None,
        description="Server-specific limitation.",
    )
    configuration_schema: str = Field(
        ...,
        description="OpenAPI component schema name for this model's configuration.",
    )


class EmbeddingDescriptor(SchemaBase):
    """Catalog entry describing one embedding implementation."""

    name: EmbeddingName = Field(..., description="Public embedding class name.")
    purpose: str = Field(..., description="When to choose this embedder.")
    extra: str = Field(
        default="multimodal",
        description="Optional extra required to import this embedder.",
    )
    status: ComponentStatus = Field(..., description="Current availability.")
    available: bool = Field(
        ..., description="Whether the embedder can be instantiated."
    )
    limitation: str | None = Field(
        default=None,
        description="Server-specific limitation.",
    )
    configuration_schema: str = Field(
        ...,
        description="OpenAPI component schema name for this embedder's configuration.",
    )


class ComponentCatalogResponse(SchemaBase):
    """Discovery payload for readers, splitters, vision models, and embeddings."""

    readers: list[ReaderDescriptor] = Field(
        ...,
        description="Readers that can be selected through the reader discriminator.",
    )
    splitters: list[SplitterDescriptor] = Field(
        ...,
        description=(
            "Splitters that can be selected through the splitter discriminator."
        ),
    )
    vision_models: list[VisionModelDescriptor] = Field(
        ...,
        description="Vision models that can be selected through the model discriminator.",
    )
    embeddings: list[EmbeddingDescriptor] = Field(
        ...,
        description=(
            "Embedding backends that can be selected through the embedding "
            "discriminator. Required when splitter is SemanticSplitter."
        ),
    )
