---
name: splitter-mr
description: >-
  Reads documents and splits them into LLM-ready chunks using SplitterMR
  readers, splitters, vision models, and embeddings. Use when working with
  SplitterMR, document chunking, ReaderOutput, SplitterOutput, semantic
  splitting, multimodal PDF reading, or RAG ingestion pipelines.
---

# SplitterMR Usage Guide

## Quick start

SplitterMR turns files into structured, LLM-ready text chunks through a modular
pipeline:

```text
Reader -> (optional VisionModel) -> Splitter -> (optional Embedding)
```

Public contracts:

- **`ReaderOutput`** from any reader's `read(...)` method.
- **`SplitterOutput`** from any splitter's `split(ReaderOutput)` method.

Access fields with dot notation. Do not return ad-hoc dictionaries.

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import CharacterSplitter

reader_output = VanillaReader().read("document.txt")
splitter_output = CharacterSplitter(chunk_size=500).split(reader_output)

print(len(splitter_output.chunks))
print(splitter_output.chunk_id[0])
```

## Installation

Python 3.11+ is required.

```bash
pip install splitter-mr
```

Optional extras:

| Extra | Use when |
| ----- | -------- |
| `markitdown` | Rich document parsing (HTML, DOCX, etc.) |
| `docling` | High-quality PDF/document to Markdown conversion |
| `textract` | AWS Textract OCR for scanned PDFs, Office files, images |
| `multimodal` | Vision models, HuggingFace embeddings, Gemini, Voyage |
| `all` | Full install (heavy) |

```bash
pip install "splitter-mr[markitdown,docling,multimodal]"
```

## Component selection

### Readers

| Reader | Choose when |
| ------ | ----------- |
| `VanillaReader` | Preserve original structure; simple text/PDF/Office parsing |
| `DoclingReader` | Documents with tables or rich visual layout |
| `MarkItDownReader` | Fast Markdown conversion for many formats |
| `TextractReader` | Managed cloud OCR via AWS Textract |

Docs: [Reader API](docs/api_reference/reader.md)

### Splitters

| Splitter | Choose when |
| -------- | ----------- |
| `CharacterSplitter`, `WordSplitter`, `SentenceSplitter` | Fixed-size chunks with overlap |
| `RecursiveCharacterSplitter` | Preserve semantic units while respecting size limits |
| `KeywordSplitter` | Split on regex patterns (chapters, sections, labels) |
| `SemanticSplitter` | Group text by meaning using embeddings |
| `HeaderSplitter`, `HTMLTagSplitter`, `CodeSplitter` | Structured Markdown, HTML, or source code |

Docs: [Splitter API](docs/api_reference/splitter.md)

### Vision models

Pass a vision model to a reader as `model=...` when PDFs or images need VLM
extraction.

Docs: [Vision models](docs/api_reference/model.md)

### Embeddings

Required for `SemanticSplitter` and external vector workflows.

Docs: [Embedding models](docs/api_reference/embedding.md)

## Workflow 1: Read and split by fixed size

Adapted from [fixed_splitter example](docs/examples/text/fixed_splitter.md).

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import (
    CharacterSplitter,
    WordSplitter,
    SentenceSplitter,
    ParagraphSplitter,
)

file_path = (
    "https://raw.githubusercontent.com/andreshere00/Splitter_MR/"
    "refs/heads/main/data/quijote_example.txt"
)

reader_output = VanillaReader().read(file_path)

char_output = CharacterSplitter(chunk_size=100).split(reader_output)
word_output = WordSplitter(chunk_size=20).split(reader_output)
sentence_output = SentenceSplitter(chunk_size=3).split(reader_output)
paragraph_output = ParagraphSplitter(chunk_size=1).split(reader_output)

for idx, chunk in enumerate(char_output.chunks):
    print(f"Chunk {idx + 1}: {chunk[:80]}...")
```

Key points:

- `read(...)` accepts local paths, URLs, or raw strings.
- Splitters preserve document metadata from `ReaderOutput`.
- Tune `chunk_size` and `chunk_overlap` per downstream LLM limits.

## Workflow 2: Semantic splitting with embeddings

Adapted from [semantic_splitter example](docs/examples/text/semantic_splitter.md).

```python
from dotenv import load_dotenv

from splitter_mr.embedding import OpenRouterEmbedding
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import SemanticSplitter

load_dotenv()

file_path = (
    "https://raw.githubusercontent.com/andreshere00/Splitter_MR/"
    "refs/heads/main/data/pinocchio_example.md"
)

reader_output = VanillaReader().read(file_path=file_path)
embedding = OpenRouterEmbedding()  # OPENROUTER_API_KEY

splitter = SemanticSplitter(
    embedding=embedding,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80.0,
    chunk_size=1000,
)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print(f"\n--- Chunk {idx} ---\n{chunk[:300]}...")
```

Environment:

```txt
OPENROUTER_API_KEY=<your-api-key>
# optional: OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-large
```

Install `pip install "splitter-mr[multimodal]"` when using local HuggingFace
embeddings instead of cloud providers.

## Workflow 3: Multimodal PDF reading with a vision model

Adapted from [pdf_vanilla example](docs/examples/pdf/pdf_vanilla.md).

```python
from splitter_mr.model import OpenRouterVisionModel
from splitter_mr.reader import VanillaReader

model = OpenRouterVisionModel()  # OPENROUTER_API_KEY
reader = VanillaReader(model=model)

file_path = "data/sample_pdf.pdf"
reader_output = reader.read(file_path=file_path)

print(reader_output.text)
print(reader_output.reader_method)
print(reader_output.conversion_method)
```

Environment:

```txt
OPENROUTER_API_KEY=<your-api-key>
# optional: OPENROUTER_MODEL=openai/gpt-5.6-luna
```

Swap providers by changing only the model constructor:

```python
from splitter_mr.model import OpenAIVisionModel, GeminiVisionModel

# OpenAI cloud
model = OpenAIVisionModel()

# Google Gemini (requires multimodal extra)
model = GeminiVisionModel()
```

Install multimodal extras when the provider needs optional SDK dependencies:

```bash
pip install "splitter-mr[multimodal]"
```

## Workflow 4: RAG ingestion with SplitterMR + Qdrant

Adapted from [rag_simple example](docs/examples/use_cases/rag_simple.md).

### Step 1: Read and split

```python
import os
import re

from dotenv import load_dotenv, find_dotenv
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import KeywordSplitter

load_dotenv(dotenv_path=find_dotenv())

file_path = "https://www.gutenberg.org/cache/epub/16865/pg16865.txt"

reader_output = VanillaReader().read(file_path=file_path)

splitter = KeywordSplitter(
    patterns=[r"CHAPTER\s+[IVXLCDM]+"],
    include_delimiters="after",
    flags=re.IGNORECASE,
    chunk_size=100000,
)
splitter_output = splitter.split(reader_output)

texts = splitter_output.chunks
chunk_ids = splitter_output.chunk_id
```

### Step 2: Embed and upsert to Qdrant

```python
from itertools import batched

from openai import OpenAI
from qdrant_client import QdrantClient, models

from splitter_mr.schema import (
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_ENTRYPOINT,
)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
embedding_model = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", DEFAULT_OPENROUTER_EMBEDDING_MODEL
)
base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_ENTRYPOINT)
collection_name = "pinocchio_demo_chunks"
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

client = OpenAI(api_key=openrouter_api_key, base_url=base_url)
qdrant = QdrantClient(url=qdrant_url)

probe = client.embeddings.create(model=embedding_model, input=["dim-probe"])
embedding_dim = len(probe.data[0].embedding)

if collection_name in [c.name for c in qdrant.get_collections().collections]:
    qdrant.delete_collection(collection_name)

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.DOT),
)

base_payload = {
    "source": splitter_output.document_name,
    "document_path": splitter_output.document_path,
    "document_id": splitter_output.document_id,
    "conversion_method": splitter_output.conversion_method,
    "reader_method": splitter_output.reader_method,
    "ocr_method": splitter_output.ocr_method,
    "split_method": splitter_output.split_method,
}

points = []
for i, chunk_text in enumerate(texts):
    payload = {
        **base_payload,
        "chunk_id": chunk_ids[i],
        "chunk_index": i,
        "text": chunk_text,
    }
    points.append((chunk_ids[i], chunk_text, payload))

for pack in batched(points, 64):
    ids = [pid for pid, _, _ in pack]
    inputs = [txt for _, txt, _ in pack]
    payloads = [pl for _, _, pl in pack]

    emb = client.embeddings.create(model=embedding_model, input=inputs)
    vectors = [d.embedding for d in emb.data]

    qdrant.upsert(
        collection_name=collection_name,
        points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        wait=True,
    )
```

### Step 3: Retrieve and generate

```python
from splitter_mr.schema import DEFAULT_OPENROUTER_MODEL

generative_model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
system_prompt = (
    "Answer the user's question concisely but precisely using ONLY the "
    "provided context. Cite sources as [chunk_id] next to claims."
)


def retrieve(query: str, k: int = 5) -> list[dict]:
    query_vec = (
        client.embeddings.create(model=embedding_model, input=[query])
        .data[0]
        .embedding
    )
    res = qdrant.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=k,
        with_payload=True,
    )
    hits = []
    for point in res.points:
        payload = point.payload or {}
        hits.append(
            {
                "score": point.score,
                "chunk_id": payload.get("chunk_id", point.id),
                "text": payload.get("text", ""),
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
            }
        )
    return hits


def answer_with_rag(query: str, k: int = 5) -> dict:
    hits = retrieve(query, k=k)
    context_blocks = []
    for hit in hits:
        header = (
            f"[chunk_id: {hit['chunk_id']}] source: {hit['source']} "
            f"(idx {hit['chunk_index']})"
        )
        context_blocks.append(f"{header}\n{hit['text'][:2000]}")

    response = client.chat.completions.create(
        model=generative_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n" + "\n\n".join(
                    context_blocks
                ),
            },
        ],
        temperature=0.2,
    )
    return {"query": query, "hits": hits, "answer": response.choices[0].message.content}


print(answer_with_rag("Who is the author of Pinocchio?")["answer"])
```

External requirements:

- Docker or a running Qdrant instance (`QDRANT_URL`).
- `OPENROUTER_API_KEY` for embeddings and generation.
- Optional: `pip install qdrant-client openai python-dotenv`.

## Common patterns

### Preserve metadata across the pipeline

```python
splitter_output.document_name == reader_output.document_name
splitter_output.document_id == reader_output.document_id
splitter_output.reader_method == reader_output.reader_method
```

### Choose a splitter from document structure

- Plain prose: `RecursiveCharacterSplitter` or `SemanticSplitter`.
- Chaptered books: `KeywordSplitter` with chapter regex.
- Markdown/HTML docs: `HeaderSplitter`.
- JSON payloads: `RecursiveJSONSplitter`.
- Source code: `CodeSplitter(language="python")`.

### Optional dependencies

Keep imports lazy. Install extras only when a workflow needs them:

```bash
pip install "splitter-mr[textract]"      # TextractReader
pip install "splitter-mr[docling]"       # DoclingReader
pip install "splitter-mr[multimodal]"    # Gemini, HuggingFace, Anthropic/Voyage
```

## Additional resources

- Architecture and API overview: [docs/api_reference/api_reference.md](docs/api_reference/api_reference.md)
- Runnable examples index: [docs/examples/examples.md](docs/examples/examples.md)
- Developer contracts and testing rules: [AGENTS.md](AGENTS.md)
- Published docs: https://andreshere00.github.io/Splitter_MR/
