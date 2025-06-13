# Splitter

## Introduction

The **Splitter** component implements the main functionality of this library. This component is designed to deliver classes (inherited from `BaseSplitter`) which supports to split a markdown text or a string following many different strategies. 

### Splitter strategies description

| Splitting Technique       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Character Splitter**    | Splits text into chunks based on a specified number of characters. Supports overlapping by character count or percentage. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars: int or %). <br> **Compatible with:** Text.                                                                                                                                                                                                                               |
| **Word Splitter**         | Splits text into chunks based on a specified number of words. Supports overlapping by word count or percentage. <br> **Parameters:** `chunk_size` (max words per chunk), `chunk_overlap` (overlapping words: int or %). <br> **Compatible with:** Text.                                                                                                                                                                                                                                         |
| **Sentence Splitter**     | Splits text into chunks by a specified number of sentences. Allows overlap defined by a number or percentage of words from the end of the previous chunk. Customizable sentence separators (e.g., `.`, `!`, `?`). <br> **Parameters:** `chunk_size` (max sentences per chunk), `chunk_overlap` (overlapping words: int or %), `sentence_separators` (list of characters). <br> **Compatible with:** Text.                                                                                       |
| **Paragraph Splitter**    | Splits text into chunks based on a specified number of paragraphs. Allows overlapping by word count or percentage, and customizable line breaks. <br> **Parameters:** `chunk_size` (max paragraphs per chunk), `chunk_overlap` (overlapping words: int or %), `line_break` (delimiter(s) for paragraphs). <br> **Compatible with:** Text.                                                                                                                                                       |
| **Recursive Splitter**    | Recursively splits text based on a hierarchy of separators (e.g., paragraph, sentence, word, character) until chunks reach a target size. Tries to preserve semantic units as long as possible. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars), `separators` (list of characters to split on, e.g., `["\n\n", "\n", " ", ""]`). <br> **Compatible with:** Text.                                                                                   |
| **Token Splitter**        | Splits text into chunks based on the number of tokens, using various tokenization models (e.g., tiktoken, spaCy, NLTK). Useful for ensuring chunks are compatible with LLM context limits. <br> **Parameters:** `chunk_size` (max tokens per chunk), `model_name` (tokenizer/model, e.g., `"tiktoken/cl100k_base"`, `"spacy/en_core_web_sm"`, `"nltk/punkt"`), `language` (for NLTK). <br> **Compatible with:** Text.                                                                           |
| **Paged Splitter**        | **WORK IN PROGRESS**. Splits text by pages for documents that have page structure. Each chunk contains a specified number of pages, with optional word overlap. <br> **Parameters:** `num_pages` (pages per chunk), `chunk_overlap` (overlapping words). <br> **Compatible with:** Word, PDF, Excel, PowerPoint.                                                                                                                                                                                |
| **Row/Column Splitter**   | For tabular formats, splits data by a set number of rows or columns per chunk, with possible overlap. Row-based and column-based splitting are mutually exclusive. <br> **Parameters:** `num_rows`, `num_cols` (rows/columns per chunk), `overlap` (overlapping rows or columns). <br> **Compatible with:** Tabular formats (csv, tsv, parquet, flat json).                                                                                                                                     |
| **Schema Based Splitter** | **WORK IN PROGRESS**. Splits hierarchical documents (XML, HTML) based on element tags or keys, preserving the schema/structure. Splitting can be done on a specified or inferred parent key/tag. <br> **Parameters:** `chunk_size` (approx. max chars per chunk), `key` (optional parent key or tag). <br> **Compatible with:** XML, HTML.                                                                                                                                                      |
| **JSON Splitter**         | Recursively splits JSON documents into smaller sub-structures that preserve the original JSON schema. <br> **Parameters:** `max_chunk_size` (max chars per chunk), `min_chunk_size` (min chars per chunk). <br> **Compatible with:** JSON.                                                                                                                                                                                                                                                      |
| **Semantic Splitter**     | **WORK IN PROGRESS**. Splits text into chunks based on semantic similarity, using an embedding model and a max tokens parameter. Useful for meaningful semantic groupings. <br> **Parameters:** `embedding_model` (model for embeddings), `max_tokens` (max tokens per chunk). <br> **Compatible with:** Text.                                                                                                                                                                                  |
| **HTMLTagSplitter**       | Splits HTML content based on a specified tag, or automatically detects the most frequent and shallowest tag if not specified. Each chunk is a complete HTML fragment for that tag. <br> **Parameters:** `chunk_size` (max chars per chunk), `tag` (HTML tag to split on, optional). <br> **Compatible with:** HTML.                                                                                                                                                                             |
| **HeaderSplitter**        | Splits Markdown or HTML documents into chunks using header levels (e.g., `#`, `##`, or `<h1>`, `<h2>`). Uses configurable headers for chunking. <br> **Parameters:** `headers_to_split_on` (list of headers and semantic names), `chunk_size` (unused, for compatibility). <br> **Compatible with:** Markdown, HTML.                                                                                                                                                                            |
| **Code Splitter**         | Splits source code files into programmatically meaningful chunks (functions, classes, methods, etc.), aware of the syntax of the specified programming language (e.g., Python, Java, Kotlin). Uses language-aware logic to avoid splitting inside code blocks. <br> **Parameters:** `chunk_size` (max chars per chunk), `language` (programming language as string, e.g., `"python"`, `"java"`). <br> **Compatible with:** Source code files (Python, Java, Kotlin, C++, JavaScript, Go, etc.). |


!!! warning
    **Schema Based Splitter**, **PagedSplitter** amd **Semantic Splitter** are **not fully implemented yet**. Stay aware to updates.

### Output format

::: splitter_mr.schema.schemas.SplitterOutput
    handler: python
    options:
      members_order: source

## Splitters

### BaseSplitter

::: splitter_mr.splitter.base_splitter
    handler: python
    options:
      members_order: source

### CharacterSplitter

::: splitter_mr.splitter.splitters.character_splitter
    handler: python
    options:
      members_order: source

### WordSplitter

::: splitter_mr.splitter.splitters.word_splitter
    handler: python
    options:
      members_order: source

### SentenceSplitter

::: splitter_mr.splitter.splitters.sentence_splitter
    handler: python
    options:
      members_order: source

### ParagraphSplitter

::: splitter_mr.splitter.splitters.paragraph_splitter
    handler: python
    options:
      members_order: source

### RecursiveSplitter

::: splitter_mr.splitter.splitters.recursive_splitter
    handler: python
    options:
      members_order: source

### HeaderSplitter

::: splitter_mr.splitter.splitters.header_splitter
    handler: python
    options:
      members_order: source

### JSONRecursiveSplitter

::: splitter_mr.splitter.splitters.json_splitter
    handler: python
    options:
      members_order: source

### HTMLTagSplitter

::: splitter_mr.splitter.splitters.html_tag_splitter
    handler: python
    options:
      members_order: source

### RowColumnSplitter

::: splitter_mr.splitter.splitters.row_column_splitter
    handler: python
    options:
      members_order: source

### CodeSplitter

::: splitter_mr.splitter.splitters.code_splitter
    handler: python
    options:
      members_order: source

### TokenSplitter

::: splitter_mr.splitter.splitters.token_splitter
    handler: python
    options:
      members_order: source

### PagedSplitter

Splits text by pages for documents that have page structure. Each chunk contains a specified number of pages, with optional word overlap.

> Coming soon!

### SchemaBasedSplitter

Splits hierarchical documents (XML, HTML) based on element tags or keys, preserving the schema/structure. Splitting can be done on a specified or inferred parent key/tag.

> Coming soon!

### SemanticSplitter

Splits text into chunks based on semantic similarity, using an embedding model and a max tokens parameter. Useful for meaningful semantic groupings.

> Coming soon!
