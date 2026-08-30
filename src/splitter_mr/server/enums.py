"""Enumerations used by the SplitterMR MCP and REST contracts."""

from enum import Enum


class SourceType(str, Enum):
    """Supported ``file_path`` kinds inferred by the read pipeline."""

    TEXT = "text"
    JSON = "json"
    FILE = "file"
    URL = "url"


class ReaderName(str, Enum):
    """Public reader class names that the server can instantiate."""

    VANILLA = "VanillaReader"
    MARKITDOWN = "MarkItDownReader"
    DOCLING = "DoclingReader"
    TEXTRACT = "TextractReader"


class VisionModelName(str, Enum):
    """Public vision-model class names that the server can instantiate."""

    OPENAI = "OpenAIVisionModel"
    AZURE_OPENAI = "AzureOpenAIVisionModel"
    ANTHROPIC = "AnthropicVisionModel"
    OPENROUTER = "OpenRouterVisionModel"
    GROK = "GrokVisionModel"
    GEMINI = "GeminiVisionModel"
    HUGGINGFACE = "HuggingFaceVisionModel"


class EmbeddingName(str, Enum):
    """Public embedding class names that the server can instantiate."""

    OPENAI = "OpenAIEmbedding"
    AZURE_OPENAI = "AzureOpenAIEmbedding"
    OPENROUTER = "OpenRouterEmbedding"
    GEMINI = "GeminiEmbedding"
    HUGGINGFACE = "HuggingFaceEmbedding"
    ANTHROPIC = "AnthropicEmbedding"


class SplitterName(str, Enum):
    """Public splitter class names that the server can instantiate."""

    CHARACTER = "CharacterSplitter"
    WORD = "WordSplitter"
    SENTENCE = "SentenceSplitter"
    PARAGRAPH = "ParagraphSplitter"
    RECURSIVE_CHARACTER = "RecursiveCharacterSplitter"
    KEYWORD = "KeywordSplitter"
    TOKEN = "TokenSplitter"
    PAGED = "PagedSplitter"
    ROW_COLUMN = "RowColumnSplitter"
    RECURSIVE_JSON = "RecursiveJSONSplitter"
    HTML_TAG = "HTMLTagSplitter"
    HEADER = "HeaderSplitter"
    CODE = "CodeSplitter"
    SEMANTIC = "SemanticSplitter"


class ComponentStatus(str, Enum):
    """Availability of a catalogued reader or splitter."""

    AVAILABLE = "available"
    MISSING_EXTRA = "missing_extra"
    UNSUPPORTED = "unsupported"


class HealthStatus(str, Enum):
    """Liveness status reported by the health endpoint."""

    OK = "ok"
