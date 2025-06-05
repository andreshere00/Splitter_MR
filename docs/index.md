# Splitter MR

<img src="./assets/splitter_mr_logo.svg" alt="SplitterMR logo" width="80%"/>

## Description

**SplitterMR** is a library for chunking data into convenient text blocks compatible with your LLM applications.

## Features

### Different input formats

Splitter MR can read data from multiples sources and files. For reading the files, it uses a component name `BaseReader`. This object allows you to read the files as a formatted string or converted into a markdown document. Currently, there are supported three readers: `VanillaReader`, and `MarkItDownReader` and `DoclingReader`. 

| **Reader**         | **Unstructured files & PDFs**    | **MS Office suite files**         | **Tabular data**        | **Files with hierarchical schema**      | **Image files**                  | **Markdown conversion** |
|--------------------|----------------------------------|-----------------------------------|-------------------------|----------------------------------------|----------------------------------|----------------------------------|
| **Vanilla Reader**      | `txt`, `csv`                     | –                                 | `csv`, `tsv`, `parquet`| `json`, `yaml`, `html`, `xml`          || No |----------------------------------| –                                |
| **MarkItDown Reader**   | `txt`, `md`, `pdf`               | `docx`, `xlsx`, `pptx`            | `csv`, `tsv`                  | `json`, `html`, `xml`                  | `jpg`, `png`, `pneg`             | Yes |
| **Docling Reader**      | `txt`, `md`, `pdf`                     | `docx`, `xlsx`, `pptx`            | –                 | `html`, `xhtml`                        | `png`, `jpeg`, `tiff`, `bmp`, `webp` | Yes |

### Serveral splitting methods

Splitter_MR allows you to split the files on many different ways depending on your needs. 

Main splitting methods include:

| Splitting Technique       | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Character Splitter**    | Splits text into chunks based on a specified number of characters. Supports overlapping by character count or percentage. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars: int or %). <br> **Compatible with:** Text.                                                                                                                                             |
| **Word Splitter**         | Splits text into chunks based on a specified number of words. Supports overlapping by word count or percentage. <br> **Parameters:** `chunk_size` (max words per chunk), `chunk_overlap` (overlapping words: int or %). <br> **Compatible with:** Text.                                                                                                                                                       |
| **Sentence Splitter**     | Splits text into chunks by a specified number of sentences. Allows overlap defined by a number or percentage of words from the end of the previous chunk. Customizable sentence separators (e.g., `.`, `!`, `?`). <br> **Parameters:** `chunk_size` (max sentences per chunk), `chunk_overlap` (overlapping words: int or %), `sentence_separators` (list of characters). <br> **Compatible with:** Text.     |
| **Paragraph Splitter**    | Splits text into chunks based on a specified number of paragraphs. Allows overlapping by word count or percentage, and customizable line breaks. <br> **Parameters:** `chunk_size` (max paragraphs per chunk), `chunk_overlap` (overlapping words: int or %), `line_break` (delimiter(s) for paragraphs). <br> **Compatible with:** Text.                                                                     |
| **Recursive Splitter**    | Recursively splits text based on a hierarchy of separators (e.g., paragraph, sentence, word, character) until chunks reach a target size. Tries to preserve semantic units as long as possible. <br> **Parameters:** `chunk_size` (max chars per chunk), `chunk_overlap` (overlapping chars), `separators` (list of characters to split on, e.g., `["\n\n", "\n", " ", ""]`). <br> **Compatible with:** Text. |
| **Paged Splitter**        | Splits text by pages for documents that have page structure. Each chunk contains a specified number of pages, with optional word overlap. <br> **Parameters:** `num_pages` (pages per chunk), `chunk_overlap` (overlapping words). <br> **Compatible with:** Word, PDF, Excel, PowerPoint.                                                                                                                    |
| **Row/Column Splitter**   | For tabular formats, splits data by a set number of rows or columns per chunk, with possible overlap. Row-based and column-based splitting are mutually exclusive. <br> **Parameters:** `num_rows`, `num_cols` (rows/columns per chunk), `overlap` (overlapping rows or columns). <br> **Compatible with:** Tabular formats (csv, tsv, parquet, flat json).                                                   |
| **Schema Based Splitter** | Splits hierarchical documents (XML, HTML) based on element tags or keys, preserving the schema/structure. Splitting can be done on a specified or inferred parent key/tag. <br> **Parameters:** `chunk_size` (approx. max chars per chunk), `key` (optional parent key or tag). <br> **Compatible with:** XML, HTML.                                                                                          |
| **JSON Splitter**         | Recursively splits JSON documents into smaller sub-structures that preserve the original JSON schema. <br> **Parameters:** `max_chunk_size` (max chars per chunk), `min_chunk_size` (min chars per chunk). <br> **Compatible with:** JSON.                                                                                                                                                                    |
| **Semantic Splitter**     | Splits text into chunks based on semantic similarity, using an embedding model and a max tokens parameter. Useful for meaningful semantic groupings. <br> **Parameters:** `embedding_model` (model for embeddings), `max_tokens` (max tokens per chunk). <br> **Compatible with:** Text.                                                                                                                      |
| **HTMLTagSplitter**       | Splits HTML content based on a specified tag, or automatically detects the most frequent and shallowest tag if not specified. Each chunk is a complete HTML fragment for that tag. <br> **Parameters:** `chunk_size` (max chars per chunk), `tag` (HTML tag to split on, optional). <br> **Compatible with:** HTML.                                                                                           |
| **HeaderSplitter**        | Splits Markdown or HTML documents into chunks using header levels (e.g., `#`, `##`, or `<h1>`, `<h2>`). Uses configurable headers for chunking. <br> **Parameters:** `headers_to_split_on` (list of headers and semantic names), `chunk_size` (unused, for compatibility). <br> **Compatible with:** Markdown, HTML.                                                                                          |

### Output Format

The output is a dictionary (or `JSON`) with the following structure:

```python
{
  'chunks': List[str],  # The extracted chunks from the text
  'chunk_id': List[str],  # The identifier for the chunks (given by default with uuid)
  'document_name': Optional[str],  # The base name of the file.
  'document_path': str,  # The path to the document
  'document_id': Optional[str],  # The identifier for that document
  'conversion_method': Optional[str],  # The method used to convert the file (i.e., markdown, json, etc.)
  'ocr_method': Optional[str],  # The OCR method or VLM used to analyze images (TBD)
  'split_method': str,  # The splitting strategy used for chunking the document
  'split_params': Optional[Dict[str, Any]],  # The specific splitter parameters
  'metadata': Optional[List[str]]  # The adjunted metadata introduced by the user (TBD)
}
```

## Architecture

![SplitterMR architecture diagram](./assets/architecture_splitter_mr.svg)

**SplitterMR** is designed around a modular pipeline that processes files from raw data all the way to chunked, LLM-ready text.

- **Reading**
    - A **`BaseReader`** implementation reads the (optionally converted) file.
    - Supported readers (e.g., **`VanillaReader`**, **`MarkItDownReader`**, **`DoclingReader`**) produce a ReaderOutput dictionary containing:
        - **Text** content (often Markdown).
        - Document **metadata** (name, path, ID, etc.).
        - **Conversion** method.
- **Splitting**
    - A **`BaseSplitter`** implementation takes the **`ReaderOutput`** and divides the text into meaningful chunks for LLM or other downstream use.
    - Splitter classes (e.g., **`CharacterSplitter`**, **`SentenceSplitter`**, **`RecursiveSplitter`**, etc.) allow flexible chunking strategies with optional overlap and rich configuration.

**And that's it!** Your data is now prepared to be used in several LLM applications.

## How to install

Currently, the package can be installed executing the following instruction:

```python
pip install -e .
```

We strongly recommend to install it using a python package management tool such as `uv`:

```python
uv add .
```

In the next days, the package will be published to be publicly accessible using the by default [pypi.org](https://pypi.org/simple) package directory. 

> Work-in-progress...

## How to use

### Read files

Firstly, you need to instantiate an object from a BaseReader class, for example, `DoclingReader`.

```python
from splitter_mr.reader import DoclingReader

reader = DoclingReader()
```

To read any file, provide the file path within the `read()` method. If you use `DoclingReader` or `MarkItDownReader`, your files will be automatically parsed to markdown text format. The result of this reader will be a `ReaderOutput` object, a dictionary with the following shape:

```python 
reader_output = reader.read(file_path = "path/to/some/text.txt)
print(reader_output)
```
```bash
{'text': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum sit amet ultricies orci. Nullam et tellus dui.', 
'document_name': 'text.txt', 
'document_path': 'path/to/some/text.txt', 
'document_id': '47ccc8b1-7259-4c83-be03-60e9990aa5cd', 
'conversion_method': 'docling', 
'ocr_method': None, 
'metadata': {}
}
```

### Split text

To split the text, first import the class that implements your desired splitting strategy (e.g., by characters, recursively, by headers, etc.). Then, create an instance of this class and call its `split` method, which is defined in the `BaseSplitter` class.

For example, we will split by characters with a maximum chunk size of 50, with an overlapping betwen chunks of 10 characters:

```python
from splitter_mr.splitter import CharacterSplitter

char_splitter = CharacterSplitter(chunk_size=50, chunk_overlap = 10)
splitter_output = char_splitter.split(reader_output)
print(splitter_output)
```
```bash
{
'chunks': ['Lorem ipsum dolor sit amet, consectetur adipiscing', 'adipiscing elit. Vestibulum sit amet ultricies orc', 'ricies orci. Nullam et tellus dui'], 
'chunk_id': ['357ee998-407a-4476-b09a-ab753a62de76', '487a8b7f-cbb3-4c79-99c8-02f51d5dac71', '257fa636-04f7-4802-9da2-a0841be4e75b'], 
'document_name': 'text.txt', 
'document_path': 'path/to/some/text.txt', 
'document_id': '5a78e5a6-db00-4cfc-8fbd-b37eb7d699ff', 
'conversion_method': 'docling', 
'ocr_method': None, 
'split_method': 'character_splitter', 
'split_params': {'chunk_size': 50, 'chunk_overlap': 10}, 
'metadata': {}
}
```

The returned dictionary is a `SplitterOutput` object, which provides all the information you need to further process your data. You can easily add custom metadata, and you have access to details such as the document name, path, and type. Each chunk is uniquely identified by an UUID, allowing for easy traceability throughout your LLM workflow.

### Compatibility with vision tools for image processing and annotations

> Coming soon!

## Contact

If you want to collaborate, please, send me a mail to the following address: [andresherencia2000@gmail.com](mailto:andresherencia2000@gmail.com).
