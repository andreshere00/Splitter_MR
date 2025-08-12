from typing import Dict, Set

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

SUPPORTED_VANILLA_IMAGE_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "webp", "gif"}

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

OPENAI_EMBEDDING_MAX_TOKENS: int = 8192

OPENAI_EMBEDDING_MODEL_FALLBACK: str = "cl100k_base"

DEFAULT_SENTENCE_SEPARATOR: str = r'(?:\.\.\.|…|[.!?])(?:["”’\'\)\]\}»]*)\s*'
