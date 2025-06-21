import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReaderOutput:
    """
    Dataclass defining the output structure for all readers.
    """

    text: Optional[str] = ""
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    conversion_method: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())

    def to_dict(self):
        return asdict(self)


@dataclass
class SplitterOutput:
    """
    Dataclass defining the output structure for all splitters.
    """

    chunks: List[str]
    chunk_id: List[str]
    document_name: Optional[str] = None
    document_path: str = ""
    document_id: Optional[str] = None
    conversion_method: Optional[str] = None
    reader_method: Optional[str] = None
    ocr_method: Optional[str] = None
    split_method: str = ""
    split_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


LANGUAGES: str = {
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
