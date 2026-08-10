"""Smoke-test OpenRouter vision + embedding against low-cost model slugs."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_DOCUMENT = DATA_DIR / "lorem_ipsum.txt"
DEFAULT_IMAGE = DATA_DIR / "chameleon.jpg"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Cheap OpenRouter slugs (override via env if needed)
DEFAULT_VISION_MODEL = "openai/gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"


def _parse_env_value(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value


def _load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _parse_env_value(value)
        if value.upper() == "XXXX" or not value:
            continue
        if key.startswith("OPENROUTER_"):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def _model_from_env(var: str, default: str) -> str:
    raw = os.getenv(var, "").strip()
    if not raw or raw.upper() == "XXXX":
        return default
    return _parse_env_value(raw)


def _load_document_chunks(path: Path, max_chunks: int = 2) -> tuple[str, list[str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Document is empty: {path}")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    if len(paragraphs) < max_chunks:
        sentences = [s.strip() for s in paragraphs[0].split(". ") if s.strip()]
        if len(sentences) >= max_chunks:
            paragraphs = sentences[:max_chunks]
        else:
            mid = max(1, len(paragraphs[0]) // 2)
            paragraphs = [paragraphs[0][:mid].strip(), paragraphs[0][mid:].strip()]
    chunks = paragraphs[:max_chunks]
    return chunks[0], chunks


def _load_image_b64(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    ext = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("utf-8"), ext


def main() -> int:
    _load_dotenv(ROOT / ".env")
    if (
        not os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY", "").upper() == "XXXX"
    ):
        print("ERROR: OPENROUTER_API_KEY not set (check .env).", file=sys.stderr)
        return 1

    if not DEFAULT_DOCUMENT.is_file():
        print(f"ERROR: missing document {DEFAULT_DOCUMENT}", file=sys.stderr)
        return 1
    if not DEFAULT_IMAGE.is_file():
        print(f"ERROR: missing image {DEFAULT_IMAGE}", file=sys.stderr)
        return 1

    vision_model = _model_from_env("OPENROUTER_MODEL", DEFAULT_VISION_MODEL)
    embedding_model = _model_from_env(
        "OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )

    from splitter_mr.embedding import OpenRouterEmbedding
    from splitter_mr.model import OpenRouterVisionModel

    first_paragraph, doc_chunks = _load_document_chunks(DEFAULT_DOCUMENT)

    print("=== OpenRouterEmbedding ===")
    print(f"model: {embedding_model}")
    print(f"document: {DEFAULT_DOCUMENT.relative_to(ROOT)}")
    embedder = OpenRouterEmbedding(model_name=embedding_model)
    vector = embedder.embed_text(first_paragraph)
    print(f"embed_text: dim={len(vector)}, sample={vector[:3]}")
    print(f"embed_text preview: {first_paragraph[:80]!r}...")

    batch = embedder.embed_documents(doc_chunks)
    print(
        f"embed_documents: count={len(batch)}, dims={[len(v) for v in batch]}, "
        f"chunks from same file"
    )

    print("\n=== OpenRouterVisionModel ===")
    print(f"model: {vision_model}")
    print(f"image: {DEFAULT_IMAGE.relative_to(ROOT)}")
    vision = OpenRouterVisionModel(model_name=vision_model)
    image_b64, file_ext = _load_image_b64(DEFAULT_IMAGE)
    caption = vision.analyze_content(
        file=image_b64,
        prompt="Describe this image in one or two sentences.",
        file_ext=file_ext,
    )
    print(f"analyze_content: {caption!r}")

    print("\nOK: OpenRouter embedding and vision calls succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
