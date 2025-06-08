# Reader

## Introduction

The **Reader** component is designed to read files homogeneously which come from many different formats and extensions. All of these readers are implemented sharing the same parent class, `BaseReader`.

### Which Reader should I use for my project?

Each Reader component extracts document text in different ways. Therefore, choosing the most suitable Reader component depends on your use case.

- If you want to preserve the original structure as much as possible, without any kind of markdown parsing, you can use the `VanillaReader` class.
- In case that you have documents which have presented many tables in its structure or with many visual components (such as images), we strongly recommend to use `DoclingReader`. 
- If you are looking to maximize efficiency or make conversions to markdown simpler, we recommend using the `MarkItDownReader` component.

!!! note

    Remember to visit the official repository and guides for these two last reader classes: 

    - **Docling [Developer guide](https://docling-project.github.io/docling/)** 
    - **MarkItDown [GitHub repository](https://github.com/microsoft/markitdown/)**.

Additionally, the file compatibility depending on the Reader class is given by the following table:

| **Reader**         | **Unstructured files & PDFs**    | **MS Office suite files**         | **Tabular data**        | **Files with hierarchical schema**      | **Image files**                  | **Markdown conversion** |
|--------------------|----------------------------------|-----------------------------------|-------------------------|----------------------------------------|----------------------------------|----------------------------------|
| **`VanillaReader`**      | `txt`, `md`                    | `xlsx`                                 | `csv`, `tsv`, `parquet`| `json`, `yaml`, `html`, `xml`          | - | No |----------------------------------| –                                |
| **`MarkItDownReader`**   | `txt`, `md`, `pdf`               | `docx`, `xlsx`, `pptx`            | `csv`, `tsv`                  | `json`, `html`, `xml`                  | `jpg`, `png`, `pneg`             | Yes |
| **`DoclingReader`**      | `txt`, `md`, `pdf`                     | `docx`, `xlsx`, `pptx`            | –                 | `html`, `xhtml`                        | `png`, `jpeg`, `tiff`, `bmp`, `webp` | Yes |

### Output format

`ReaderOutput` is the object get from the `read` method, which is a dictionary with the following attributes:

```python
text: Optional[str] = ""  # The extracted text
document_name: Optional[str] = None  # The base name of the file
document_path: str = ""  # The path to the document
document_id: Optional[str] = None  # The document identifier (given by default by an UUID)
conversion_method: Optional[str] = None  # The format in which the file has been converted (markdown, json, etc.)
reader_method: Optional[str]  # The method used to read the file (markitdown, vanilla, etc.)
ocr_method: Optional[str] = None  # The OCR method or VLM used to analyze images (TBD)
metadata: Optional[List[str]]  # The appended metadata, introduced by the user (TBD)
```

## Readers

### BaseReader

::: splitter_mr.reader.base_reader
    handler: python
    options:
      members_order: source

> 📚 **Note:** file examples are extracted from  the`data` folder in the **GitHub** repository: [**link**](https://github.com/andreshere00/Splitter_MR/tree/main/data).

### VanillaReader

::: splitter_mr.reader.readers.vanilla_reader
    handler: python
    options:
      members_order: source

### DoclingReader

![Docling logo](../assets/docling_logo.png)

::: splitter_mr.reader.readers.docling_reader
    handler: python
    options:
      members_order: source

### MarkItDownReader

![MarkItDown logo](../assets/markitdown_logo.png)

::: splitter_mr.reader.readers.markitdown_reader
    handler: python
    options:
      members_order: source
