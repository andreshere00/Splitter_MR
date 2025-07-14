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

::: splitter_mr.schema.models.ReaderOutput
    handler: python
    options:
      members_order: source

## Readers

### BaseReader

::: splitter_mr.reader.base_reader
    handler: python
    options:
      members_order: source

> 📚 **Note:** file examples are extracted from  the`data` folder in the **GitHub** repository: [**link**](https://github.com/andreshere00/Splitter_MR/tree/main/data).

### VanillaReader

![VanillaReader logo](../assets/vanilla_reader_button.svg#gh-light-mode-only)
![VanillaReader logo](../assets/vanilla_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.vanilla_reader
    handler: python
    options:
      members_order: source

`VanillaReader` uses a helper class to read PDF and use Visual Language Models. This class is `PDFPlumberReader`.

### DoclingReader

![DoclingReader logo](../assets/docling_reader_button.svg#gh-light-mode-only)
![DoclingReader logo](../assets/docling_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.docling_reader
    handler: python
    options:
      members_order: source

To execute pipelines, DoclingReader has a utils class, `DoclingUtils`.

### MarkItDownReader

![MarkItDownReader logo](../assets/markitdown_reader_button.svg#gh-light-mode-only)
![MarkItDownReader logo](../assets/markitdown_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.markitdown_reader
    handler: python
    options:
      members_order: source
