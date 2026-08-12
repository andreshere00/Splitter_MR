# **Reader**

## Introduction

The **Reader** component is designed to read files homogeneously which come from many different formats and extensions. All of these readers are implemented sharing the same parent class, [**`BaseReader`**](#basereader).

### Which Reader should I use for my project?

Each Reader component extracts document text in different ways. Therefore, choosing the most suitable Reader component depends on your use case.

- If you want to preserve the original structure as much as possible, without any kind of markdown parsing, you can use the [**`VanillaReader`**](#vanillareader) class.
- In case that you have documents which have presented many tables in its structure or with many visual components (such as images), we strongly recommend to use [**`DoclingReader`**](#doclingreader). 
- If you are looking to maximize efficiency or make conversions to markdown simpler, we recommend using the [**`MarkItDownReader`**](#markitdownreader) component.
- If you need managed cloud OCR for scanned PDFs, Office files, or images, use [**`TextractReader`**](#textractreader) with AWS Textract.

!!! note

    Remember to visit the official repository and guides for these two last reader classes: 

    - **Docling [Developer guide](https://docling-project.github.io/docling/)** 
    - **MarkItDown [GitHub repository](https://github.com/microsoft/markitdown/)**.

Additionally, the file compatibility depending on the Reader class is given by the following table:

| **Reader**                           | **Unstructured files & PDFs** | **MS Office suite files** | **Tabular data** | **Files with hierarchical schema** | **Image files**                      | **Markdown conversion** |
|-------------------------------------|-------------------------------|---------------------------|------------------|------------------------------------|--------------------------------------|-------------------------|
| [**Vanilla Reader**](#vanillareader)    | `txt`, `md`, `pdf`            | `xlsx`, `docx`, `pptx`    | `csv`, `tsv`, `parquet` | `json`, `yaml`, `html`, `xml`    | `jpg`, `png`, `webp`, `gif`          | Yes                     |
| [**MarkItDown Reader**](#markitdownreader) | `txt`, `md`, `pdf`            | `docx`, `xlsx`, `pptx`    | `csv`, `tsv`     | `json`, `html`, `xml`             | `jpg`, `png`, `pneg`                 | Yes                     |
| [**Docling Reader**](#doclingreader)   | `txt`, `md`, `pdf`            | `docx`, `xlsx`, `pptx`    | –                | `html`, `xhtml`                   | `png`, `jpeg`, `tiff`, `bmp`, `webp` | Yes                     |
| [**Textract Reader**](#textractreader) | `txt`, `md`, `pdf`            | `docx`, `xlsx`, `pptx`    | –                | `json`, `yaml`                    | `png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tif`, `tiff`, `svg` | No |

## Installing optional readers

By default, `pip install splitter-mr` installs **core** features only.  
To use optional readers, install the corresponding **extras**:

> **Python ≥ 3.11** is required.

**MarkItDown:**

```bash
pip install "splitter-mr[markitdown]"
```

**Docling:**

```bash
pip install "splitter-mr[docling]"
```

**Textract:**

```bash
pip install "splitter-mr[textract]"
```

**Multiple readers:**

```bash
pip install "splitter-mr[markitdown,docling,textract]"
```

!!! Note
    For the full matrix of extras and alternative package managers, see the global **How to install** section in the project README:
    [Splitter_MR — How to install](https://github.com/andreshere00/Splitter_MR/#how-to-install)

### Output format

::: splitter_mr.schema.models.ReaderOutput
    handler: python
    options:
      extra:
        members_order: source

## Readers

To see a comparison between reading methods, refer to the following [**example**](../examples/pdf/pdf_with_vlm.md).

### BaseReader

::: splitter_mr.reader.base_reader
    handler: python
    options:
      extra:
        members_order: source

> 📚 **Note:** file examples are extracted from  the`data` folder in the **GitHub** repository: [**link**](https://github.com/andreshere00/Splitter_MR/tree/main/data).

### VanillaReader

![VanillaReader logo](../assets/vanilla_reader_button.svg#gh-light-mode-only)
![VanillaReader logo](../assets/vanilla_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.vanilla_reader
    handler: python
    options:
      extra:
        members_order: source

`VanillaReader` uses a helper class to read PDF and use Visual Language Models. This class is `PDFPlumberReader`.

### DoclingReader

![DoclingReader logo](../assets/docling_reader_button.svg#gh-light-mode-only)
![DoclingReader logo](../assets/docling_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.docling_reader
    handler: python
    options:
      extra:
        members_order: source

To execute pipelines, DoclingReader has a utils class, `DoclingUtils`.

### MarkItDownReader

![MarkItDownReader logo](../assets/markitdown_reader_button.svg#gh-light-mode-only)
![MarkItDownReader logo](../assets/markitdown_reader_button_white.svg#gh-dark-mode-only)

::: splitter_mr.reader.readers.markitdown_reader
    handler: python
    options:
      extra:
        members_order: source

### TextractReader

::: splitter_mr.reader.readers.textract_reader
    handler: python
    options:
      extra:
        members_order: source

`TextractReader` performs synchronous AWS Textract OCR on local files. Text-native
formats (`md`, `json`, `yaml`, `txt`) are delegated to `VanillaReader`, while visual
formats are normalized to PNG pages before calling `detect_document_text`.

**Requirements**

- Install the extra: `pip install "splitter-mr[textract]"`
- Configure AWS credentials through the standard boto3 credential chain
- Install LibreOffice (`soffice`) for Office-to-PDF conversion

**Limits**

- Synchronous Textract requests accept one page per call and up to 10 MB per page
- Multi-page PDFs are rasterized page-by-page before OCR
- Plain text detection does not guarantee layout-aware multi-column reading order

**Example**

```python
from splitter_mr.reader import TextractReader

reader = TextractReader()
output = reader.read("scanned.pdf")
print(output.text)
```
