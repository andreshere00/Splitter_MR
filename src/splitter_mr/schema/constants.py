from types import MappingProxyType
from typing import Dict, Final, FrozenSet, Literal, Mapping, Tuple, get_args

# ---------- #
# Utilities  #
# ---------- #


def _litset(lit: object) -> FrozenSet[str]:
    """Turn a Literal[...] into a frozenset of its values."""
    return frozenset(get_args(lit))


def _littuple(lit: object) -> Tuple[str, ...]:
    """Turn a Literal[...] into an immutable tuple of its values."""
    return tuple(get_args(lit))


# ------- #
# Readers #
# ------- #

DEFAULT_PAGE_PLACEHOLDER: Final[str] = "<!-- page -->"
DEFAULT_IMAGE_PLACEHOLDER: Final[str] = "<!-- image -->"

# ---- MarkitDownReader ---- #

MARKITDOWN_SUPPORTED_MODELS_LITERAL = Literal[
    "AzureOpenAIVisionModel",
    "OpenAIVisionModel",
    "AnthropicVisionModel",
    "GrokVisionModel",
]
MARKITDOWN_SUPPORTED_MODELS: Final[FrozenSet[str]] = _litset(
    MARKITDOWN_SUPPORTED_MODELS_LITERAL
)

# ---- DoclingReader ---- #

SUPPORTED_DOCLING_FILE_EXTENSIONS_LITERAL = Literal[
    "md",
    "markdown",
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "html",
    "htm",
    "odt",
    "rtf",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif",
    "tiff",
]
SUPPORTED_DOCLING_FILE_EXTENSIONS: Final[FrozenSet[str]] = _litset(
    SUPPORTED_DOCLING_FILE_EXTENSIONS_LITERAL
)

# ---- VanillaReader ---- #

SUPPORTED_VANILLA_IMAGE_EXTENSIONS_LITERAL = Literal[
    "png", "jpg", "jpeg", "webp", "gif"
]
SUPPORTED_VANILLA_IMAGE_EXTENSIONS: Final[FrozenSet[str]] = _litset(
    SUPPORTED_VANILLA_IMAGE_EXTENSIONS_LITERAL
)

VANILLA_TXT_FILES_EXTENSIONS_LITERAL = Literal[
    "json", "txt", "xml", "csv", "tsv", "md", "markdown"
]
VANILLA_TXT_FILES_EXTENSIONS: Final[FrozenSet[str]] = _litset(
    VANILLA_TXT_FILES_EXTENSIONS_LITERAL
)

# ------------- #
# Vision Models #
# ------------- #

DEFAULT_IMAGE_EXTENSION: Final[str] = "png"

# ---- OpenAI & AzureOpenAI Vision Model ---- #

DEFAULT_OPENAI_MODEL: Final[str] = "gpt-5"

OPENAI_MIME = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
SUPPORTED_OPENAI_MIME_TYPES: Final[FrozenSet[str]] = _litset(OPENAI_MIME)

OPENAI_MIME_BY_EXTENSION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
)
# Validate mapping values against the Literal-derived set
assert set(OPENAI_MIME_BY_EXTENSION.values()).issubset(SUPPORTED_OPENAI_MIME_TYPES), (
    "OPENAI_MIME_BY_EXTENSION has values not in OPENAI_MIME Literal"
)
# ---- Anthropic Vision Model ---- #

DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-sonnet-4"
DEFAULT_ANTHROPIC_ENTRYPOINT: Final[str] = "https://api.anthropic.com/v1/"

# ---- OpenRouter Vision Model ---- #

DEFAULT_OPENROUTER_ENTRYPOINT: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL: Final[str] = "openai/gpt-4o"
DEFAULT_OPENROUTER_EMBEDDING_MODEL: Final[str] = "openai/text-embedding-3-large"

# ---- Grok Vision Model ---- #

DEFAULT_GROK_VISION_MODEL: Final[str] = "grok-4"
DEFAULT_GROK_ENDPOINT: Final[str] = "https://api.x.ai/v1"

GROK_MIME = Literal["image/png", "image/jpeg"]
SUPPORTED_GROK_MIME_TYPES: Final[FrozenSet[str]] = _litset(GROK_MIME)

GROK_MIME_BY_EXTENSION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
    }
)
assert set(GROK_MIME_BY_EXTENSION.values()).issubset(SUPPORTED_GROK_MIME_TYPES), (
    "GROK_MIME_BY_EXTENSION has values not in GROK_MIME Literal"
)

# ---- Gemini Vision Model ---- #

DEFAULT_GEMINI_VISION_MODEL: Final[str] = "gemini-2.5-flash"

# ---- HuggingFace Vision Model ---- #

DEFAULT_HUGGINGFACE_MODEL: Final[str] = "ds4sd/SmolDocling-256M-preview"

DEFAULT_IMAGE_TOKENS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "llava": "<image>",
        "llava-phi": "<image>",
        "llava-mistral": "<image>",
        "qwen": "<|image|>",
        "qwen2": "<|image|>",
        "idefics": "<image>",
        "blip": "<image>",
        "mini-gemini": "<image>",
        "kosmos": "<image>",
        "cogvlm": "<image>",
        "shi": "<image>",
        "idefics2": "<image>",
        "pix2struct": "<image>",
    }
)

# ---------------- #
# Embedding Models #
# ---------------- #

# ---- OpenAI ---- #

OPENAI_EMBEDDING_MAX_TOKENS: Final[int] = 8192
OPENAI_EMBEDDING_MODEL_FALLBACK: Final[str] = "cl100k_base"

# --------- #
# Splitters #
# --------- #

# ---- Sentence / Paragraph Splitter ---- #

DEFAULT_SENTENCE_SEPARATORS: Final[str] = r'(?:\.\.\.|…|[.!?])(?:["”’\'\)\]\}»]*)\s*'
DEFAULT_PARAGRAPH_SEPARATORS: Final[str] = "\n"

# ---- Recursive Splitter ---- #

DEFAULT_RECURSIVE_SEPARATORS: Final[Tuple[str, ...]] = (
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
)

# ---- Header Splitter ----- #

ALLOWED_HEADERS_LITERAL = Literal[
    "Header 1", "Header 2", "Header 3", "Header 4", "Header 5", "Header 6", "Header 7"
]

ALLOWED_HEADERS: Final[Tuple[str, ...]] = _littuple(ALLOWED_HEADERS_LITERAL)

# ---- HTML Tag Splitter ---- #

TABLE_CHILDREN_LITERAL = Literal["tr", "thead", "tbody", "th", "td"]
TABLE_CHILDREN: Final[Tuple[str, ...]] = _littuple(TABLE_CHILDREN_LITERAL)

# ---- Token splitter ---- #

DEFAULT_TOKENIZER: Final[str] = "tiktoken/cl100k_base"
DEFAULT_TOKEN_LANGUAGE: Final[str] = "english"
SUPPORTED_TOKENIZERS: Final[Tuple[str, ...]] = ("tiktoken", "spacy", "nltk")

TIKTOKEN_DEFAULTS: Final[Tuple[str, ...]] = (
    "cl100k_base",  # GPT-4o, GPT-4-turbo, GPT-3.5-turbo
    "p50k_base",  # Codex series
    "r50k_base",  # GPT-3
)

SPACY_DEFAULTS: Final[Tuple[str, ...]] = (
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
)

DEFAULT_NLTK: Final[Tuple[str, ...]] = ("punkt_tab",)

# ---- Code Splitter ---- #

# Optional: turn this large set into a Literal for the same benefits.
SUPPORTED_PROGRAMMING_LANGUAGES_LIT = Literal[
    "lua",
    "java",
    "ts",
    "tsx",
    "ps1",
    "psm1",
    "psd1",
    "ps1xml",
    "php",
    "php3",
    "php4",
    "php5",
    "phps",
    "phtml",
    "rs",
    "cs",
    "csx",
    "cob",
    "cbl",
    "hs",
    "scala",
    "swift",
    "tex",
    "rb",
    "erb",
    "kt",
    "kts",
    "go",
    "html",
    "htm",
    "rst",
    "ex",
    "exs",
    "md",
    "markdown",
    "proto",
    "sol",
    "c",
    "h",
    "cpp",
    "cc",
    "cxx",
    "c++",
    "hpp",
    "hh",
    "hxx",
    "js",
    "mjs",
    "py",
    "pyw",
    "pyc",
    "pyo",
    "pl",
    "pm",
]

SUPPORTED_PROGRAMMING_LANGUAGES: Final[FrozenSet[str]] = _litset(
    SUPPORTED_PROGRAMMING_LANGUAGES_LIT
)

# ---- Keyword Splitter ---- #

DEFAULT_KEYWORD_DELIMITER_POS: Final[str] = "before"

SUPPORTED_KEYWORD_DELIMITERS_LITERAL = Literal["none", "before", "after", "both"]
SUPPORTED_KEYWORD_DELIMITERS: Final[Tuple[str, ...]] = _littuple(
    SUPPORTED_KEYWORD_DELIMITERS_LITERAL
)

# ---- Semantic Splitter ---- #

BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]

DEFAULT_BREAKPOINTS: Final[Dict[BreakpointThresholdType, float]] = {
    "percentile": 95.0,
    "standard_deviation": 3.0,
    "interquartile": 1.5,
    "gradient": 95.0,
}
