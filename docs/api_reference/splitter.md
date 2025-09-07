# **Splitter**

## Introduction

The **Splitter** component implements the main functionality of this library. This component is designed to deliver classes (inherited from [**`BaseSplitter`**](#basesplitter)) which supports to split a markdown text or a string following many different strategies. 

### Splitter strategies description

| Splitting Technique | Description |
| ------------------------- | -----------------------------|
| [**Character Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#charactersplitter)    | Splits text into chunks based on a specified number of characters. Supports overlapping by character count or percentage. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars: int or %). <br> **Compatible with:** Text. |
| [**Word Splitter** ](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#wordsplitter)        | Splits text into chunks based on a specified number of words. Supports overlapping by word count or percentage. <br> **Parameters:** `chunk_size` (max words per chunk), `chunk_overlap` (overlapping words: int or %). <br> **Compatible with:** Text. |
| [**Sentence Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#sentencesplitter)     | Splits text into chunks by a specified number of sentences. Allows overlap defined by a number or percentage of words from the end of the previous chunk. Customizable sentence separators (e.g., `.`, `!`, `?`). <br> **Parameters:** `chunk_size` (max sentences per chunk), `chunk_overlap` (overlapping words: int or %), `sentence_separators` (list of characters). <br> **Compatible with:** Text. |
| [**Paragraph Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#paragraphsplitter)    | Splits text into chunks based on a specified number of paragraphs. Allows overlapping by word count or percentage, and customizable line breaks. <br> **Parameters:** `chunk_size` (max paragraphs per chunk), `chunk_overlap` (overlapping words: int or %), `line_break` (delimiter(s) for paragraphs). <br> **Compatible with:** Text. |
| [**Recursive Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#recursivesplitter)    | Recursively splits text based on a hierarchy of separators (e.g., paragraph, sentence, word, character) until chunks reach a target size. Tries to preserve semantic units as long as possible. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars), `separators` (list of characters to split on, e.g., `["\n\n", "\n", " ", ""]`). <br> **Compatible with:** Text.                                                                                   |
| [**Token Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#tokensplitter)        | Splits text into chunks based on the number of tokens, using various tokenization models (e.g., tiktoken, spaCy, NLTK). Useful for ensuring chunks are compatible with LLM context limits. <br> **Parameters:** `chunk_size` (max tokens per chunk), `model_name` (tokenizer/model, e.g., `"tiktoken/cl100k_base"`, `"spacy/en_core_web_sm"`, `"nltk/punkt"`), `language` (for NLTK). <br> **Compatible with:** Text. |
| [**Paged Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#pagedsplitter)        | Splits text by pages for documents that have page structure. Each chunk contains a specified number of pages, with optional word overlap. <br> **Parameters:** `num_pages` (pages per chunk), `chunk_overlap` (overlapping words). <br> **Compatible with:** Word, PDF, Excel, PowerPoint. |
| [**Row/Column Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#rowcolumnsplitter)   | For tabular formats, splits data by a set number of rows or columns per chunk, with possible overlap. Row-based and column-based splitting are mutually exclusive. <br> **Parameters:** `num_rows`, `num_cols` (rows/columns per chunk), `overlap` (overlapping rows or columns). <br> **Compatible with:** Tabular formats (csv, tsv, parquet, flat json). |
| [**JSON Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#recursivejsonsplitter)         | Recursively splits JSON documents into smaller sub-structures that preserve the original JSON schema. <br> **Parameters:** `max_chunk_size` (max chars per chunk), `min_chunk_size` (min chars per chunk). <br> **Compatible with:** JSON. |
| [**Semantic Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#semanticsplitter)     | Splits text into chunks based on semantic similarity, using an embedding model and a max tokens parameter. Useful for meaningful semantic groupings. <br> **Parameters:** `embedding_model` (model for embeddings), `max_tokens` (max tokens per chunk). <br> **Compatible with:** Text. |
| [**HTML Tag Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#htmltagsplitter)       | Splits HTML content based on a specified tag, or automatically detects the most frequent and shallowest tag if not specified. Each chunk is a complete HTML fragment for that tag. <br> **Parameters:** `chunk_size` (max chars per chunk), `tag` (HTML tag to split on, optional). <br> **Compatible with:** HTML. |
| [**Header Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#headersplitter)       | Splits Markdown or HTML documents into chunks using header levels (e.g., `#`, `##`, or `<h1>`, `<h2>`). Uses configurable headers for chunking. <br> **Parameters:** `headers_to_split_on` (list of headers and semantic names), `chunk_size` (unused, for compatibility). <br> **Compatible with:** Markdown, HTML. |
| [**Code Splitter**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#codesplitter)         | Splits source code files into programmatically meaningful chunks (functions, classes, methods, etc.), aware of the syntax of the specified programming language (e.g., Python, Java, Kotlin). Uses language-aware logic to avoid splitting inside code blocks. <br> **Parameters:** `chunk_size` (max chars per chunk), `language` (programming language as string, e.g., `"python"`, `"java"`). <br> **Compatible with:** Source code files (Python, Java, Kotlin, C++, JavaScript, Go, etc.). |

### Output format

::: splitter_mr.schema.models.SplitterOutput
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

### RecursiveCharacterSplitter

::: splitter_mr.splitter.splitters.recursive_splitter
    handler: python
    options:
      members_order: source

### HeaderSplitter

::: splitter_mr.splitter.splitters.header_splitter
    handler: python
    options:
      members_order: source

### RecursiveJSONSplitter

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

::: splitter_mr.splitter.splitters.paged_splitter
    handler: python
    options:
      members_order: source

### SemanticSplitter

Splits text into chunks based on semantic similarity, using an embedding model and a max tokens parameter. Useful for meaningful semantic groupings.

::: splitter_mr.splitter.splitters.semantic_splitter
    handler: python
    options:
      members_order: source
