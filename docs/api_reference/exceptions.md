# Warnings and Exceptions

SplitterMR exposes a small, explicit exception hierarchy so you can reliably handle errors coming from **Readers** (document ingestion/conversion) and **Splitters** (chunking/segmentation).

Exceptions live in:

```python
from splitter_mr.schema.exceptions import *
```

## Why a custom exception hierarchy?

- **Stable contracts**: you can catch `ReaderException` or `SplitterException` to handle “library-level” failures without depending on implementation details.
- **More precise handling**: configuration errors vs conversion/runtime errors are separated.
- **Wrapped backends**: some readers wrap upstream library errors into SplitterMR-specific exceptions.

!!! tip

    **Recommended practice**

    - Catch specific exceptions when you can (e.g., `ReaderConfigException`).
    - Fall back to the base class for a broad handler (e.g., `ReaderException` / `SplitterException`).
    - Avoid catching Exception unless you are at an application boundary.

## Reader exceptions

Readers raise `ReaderException` (or subclasses) when the library cannot read/convert/validate inputs.

Hierarchy

```sh
`ReaderException` (`Exception`)
├── `ReaderOutputException`
├── `HtmlConversionError`
├── `ReaderConfigException` (also `ValueError`)
├── `VanillaReaderException` (also `RuntimeError`)
├── `MarkItDownReaderException` (also `RuntimeError`)
├── `DoclingReaderException` (also `RuntimeError`)
└── `TextractReaderException` (also `RuntimeError`)
```

### General

#### ReaderException

::: src.splitter_mr.schema.exceptions.ReaderException
    handler: python
    options:
      members_order: source

### I/O and Configuration

#### ReaderOutputException

::: src.splitter_mr.schema.exceptions.ReaderOutputException
    handler: python
    options:
      members_order: source

#### ReaderConfigException

::: src.splitter_mr.schema.exceptions.ReaderConfigException
    handler: python
    options:
      members_order: source

**Typical cases**:

- Unsupported file extension or mode.
- Mutually incompatible flags (e.g., page-splitting options on formats that don’t support it).
- Invalid values (negative sizes, unknown enum-like strings, etc.).

### Readers

#### VanillaReaderException

::: src.splitter_mr.schema.exceptions.VanillaReaderException
    handler: python
    options:
      members_order: source

!!! note
    
    Wraps exceptions coming from vanilla_reader.exceptions.VanillaReaderError.

**Typical cases:**

- A subprocess/tool invocation fails (if used internally)
- Conversion/parse errors for JSON/XML/YAML/CSV/Parquet, etc.
- Filesystem/temporary directory issues during conversion

#### MarkItDownReaderException

::: src.splitter_mr.schema.exceptions.MarkItDownReaderException
    handler: python
    options:
      members_order: source

!!! note

    Wraps exceptions coming from markitdown.exceptions.MarkItDownError.

**Typical cases**:

- Backend conversion fails for a supported document type
- External dependency misconfiguration (where applicable)

#### DoclingReaderException

::: src.splitter_mr.schema.exceptions.DoclingReaderException
    handler: python
    options:
      members_order: source

!!! note

    Wraps exceptions coming from docling.exceptions.BaseError.

**Typical cases**:

- Docling pipeline errors while parsing PDFs or documents
- Model/runtime errors in the Docling stack

#### TextractReaderException

::: src.splitter_mr.schema.exceptions.TextractReaderException
    handler: python
    options:
      members_order: source

**Typical cases**:

- AWS Textract API failures (throttling, unsupported document, size limits)
- Local conversion failures (LibreOffice, rasterization, image normalization)

## Splitter exceptions

Splitters raise `SplitterException` (or subclasses) when the library cannot construct chunks or validate splitter configuration/output.

Hierarchy

```sh
`SplitterException` (`Exception`)
├── `InvalidChunkException` (also `ValueError`)
├── `SplitterConfigException` (also `ValueError`)
│   ├── `InvalidHeaderNameError`
│   └── `HeaderLevelOutOfRangeError`
└── `SplitterOutputException` (also `TypeError`)
```

### General

#### SplitterException

::: src.splitter_mr.schema.exceptions.SplitterException
    handler: python
    options:
      members_order: source

### I/O and Configuration

#### InvalidChunkException

::: src.splitter_mr.schema.exceptions.InvalidChunkException
    handler: python
    options:
      members_order: source

**Typical cases**:

- Chunk boundaries cannot be computed
- Empty/invalid intermediate structures prevent chunk creation

#### SplitterConfigException

::: src.splitter_mr.schema.exceptions.InvalidChunkException
    handler: python
    options:
      members_order: source

**Typical cases:**

- Missing required parameters
- Invalid ranges (e.g., chunk sizes)
- Unsupported strategy options

#### SplitterOutputException

::: src.splitter_mr.schema.exceptions.SplitterOutputException
    handler: python
    options:
      members_order: source

**Typical cases:**

- Output validation fails
- Inconsistent internal fields (e.g., missing chunks, wrong metadata types)

### Splitters-specific exceptions

#### HeaderSplitter

**`NormalizationError`**

::: src.splitter_mr.schema.exceptions.NormalizationError
    handler: python
    options:
      members_order: source

**`HeaderLevelOutOfRangeError`**

::: src.splitter_mr.schema.exceptions.HeaderLevelOutOfRangeError
    handler: python
    options:
      members_order: source

**`InvalidHeaderNameError`**

::: src.splitter_mr.schema.exceptions.InvalidHeaderNameError
    handler: python
    options:
      members_order: source

#### HtmlTagSplitter

**`HtmlConversionError`**

::: src.splitter_mr.schema.exceptions.HtmlConversionError
    handler: python
    options:
      members_order: source

## Reference table

| **Area**     | **Exception** | **Type** | **Description** |
|----------|----------|------|---------|
| Reader   | `ReaderException` | `Exception` | Base reader error |
| Reader   | `ReaderOutputException` | `ReaderException` | Invalid `ReaderOutput` structure |
| Reader   | `HtmlConversionError` | `ReaderException` | HTML → Markdown conversion failed |
| Reader   | `ReaderConfigException` | `ValueError` | Invalid reader configuration |
| Reader   | `VanillaReaderException` | `RuntimeError` | Vanilla conversion failed (wrapped) |
| Reader   | `MarkItDownReaderException` | `RuntimeError` | MarkItDown conversion failed (wrapped) |
| Reader   | `DoclingReaderException` | `RuntimeError` | Docling conversion failed (wrapped) |
| Reader   | `TextractReaderException` | `RuntimeError` | Textract conversion failed (wrapped) |
| Splitter | `SplitterException` | `Exception` | Base splitter error |
| Splitter | `InvalidChunkException` | `ValueError` | Chunks cannot be constructed |
| Splitter | `SplitterConfigException` | `ValueError` | Invalid splitter configuration |
| Splitter | `SplitterOutputException` | `TypeError` | Invalid `SplitterOutput` |
| Header Splitter | `InvalidHeaderNameError` | `SplitterConfigException` | Bad `"Header N"` format |
| Header Splitter | `HeaderLevelOutOfRangeError` | `SplitterConfigException` | Header level not in `1..6` |
| Header Splitter   | `NormalizationError` | `ReaderException` | Setext → ATX normalization failed |
| HTML Tag Splitter   | `InvalidHtmlTagError` | `ReaderException` | Invalid/missing HTML tag |
