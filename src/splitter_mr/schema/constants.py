from typing import Dict, Set

# ------- #
# Readers #
# ------- #

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

# ---- Sentence Splitters constants ---- #

DEFAULT_SENTENCE_SEPARATOR: str = r'(?:\.\.\.|…|[.!?])(?:["”’\'\)\]\}»]*)\s*'

# ---- Code Splitters constants ---- #

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
