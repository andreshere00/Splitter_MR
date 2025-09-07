from typing import Dict, List, Literal, Set

# ------- #
# Readers #
# ------- #

# ---- MarkitDown constants ---- #

MARKITDOWN_SUPPORTED_MODELS: Set[str] = {
    "AzureOpenAIVisionModel",
    "OpenAIVisionModel",
    "AnthropicVisionModel",
    "GrokVisionModel",
}

# ---- Docling constants ---- #

SUPPORTED_DOCLING_FILE_EXTENSIONS: Set[str] = {
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
}

SUPPORTED_VANILLA_IMAGE_EXTENSIONS: Set[str] = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
}
# TODO: Review if these image extensions make sense or it depends on the Vision Model

# ------ #
# Models #
# ------ #

# ---- OpenAI and AzureOpenAI constants ---- #

SUPPORTED_OPENAI_MIME_TYPES: Set[str] = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

OPENAI_MIME_BY_EXTENSION: Dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

# ---- Grok Vision model constants ---- #

SUPPORTED_GROK_MIME_TYPES: Set[str] = {
    "image/png",
    "image/jpeg",
}

GROK_MIME_BY_EXTENSION: Dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

# ---- HuggingFace Vision Model constants ---- #

DEFAULT_IMAGE_TOKENS: Dict[str, str] = {
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

# ---------- #
# Embeddings #
# ---------- #

# ---- OpenAI embedding models constants ---- #

OPENAI_EMBEDDING_MAX_TOKENS: int = 8192

# TODO: Define max tokens by model

OPENAI_EMBEDDING_MODEL_FALLBACK: str = "cl100k_base"

# --------- #
# Splitters #
# --------- #

# ---- Sentence Splitter constants ---- #

DEFAULT_SENTENCE_SEPARATORS: str = r'(?:\.\.\.|…|[.!?])(?:["”’\'\)\]\}»]*)\s*'

# ---- Paragraph Splitter constants ---- #

DEFAULT_PARAGRAPH_SEPARATORS: str = "\n"

# ---- Recursive Splitter constants ---- #

DEFAULT_RECURSIVE_SEPARATORS: List[str] = [
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
]

# ---- Token Splitter constants ---- #

# -> Default settings for TokenSplitter

DEFAULT_TOKENIZER: str = "tiktoken/cl100k_base"
DEFAULT_TOKEN_LANGUAGE: str = "english"
SUPPORTED_TOKENIZERS: List[str] = ["tiktoken", "spacy", "nltk"]

# -> Known/default models per tokenizer

TIKTOKEN_DEFAULTS: List[str] = [
    "cl100k_base",  # GPT-4o, GPT-4-turbo, GPT-3.5-turbo
    "p50k_base",  # Codex series
    "r50k_base",  # GPT-3
]

SPACY_DEFAULTS: List[str] = [
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
]

NLTK_DEFAULTS: List[str] = [
    "punkt_tab",
]

# ---- Code Splitter constants ---- #

SUPPORTED_PROGRAMMING_LANGUAGES: Set[str] = {
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
}

# ---- Semantic Splitter constants ---- #

BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]

DEFAULT_BREAKPOINTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95.0,
    "standard_deviation": 3.0,
    "interquartile": 1.5,
    "gradient": 95.0,
}
