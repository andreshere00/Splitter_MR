# **Example:** Reading PDF Documents with Images using MarkItDownReader

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button.svg#only-light" alt="MarkItDownReader logo">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button_white.svg#only-dark" alt="MarkItDownReader logo">
</p>

As we have seen in previous examples, reading a PDF is not a simple task. In this case, we will see how to read a PDF using **MarkItDown** framework, and connect this library into Visual Language Models to extract text or get annotations from images.

## How to connect a VLM to MarkItDownReader

For this example, we will use the same document as the (previous tutorial)[https://github.com/andreshere00/Splitter_MR/blob/main/data/sample_pdf.pdf].

To extract image descriptions or perform OCR, instantiate a vision model and pass it to your `MarkItDownReader`. 

Currently, two models are supported: one from **OpenAI** and one from an **Azure** **OpenAI** deployment. After choosing a model, you simply need to instantiate the `BaseModel` class, which implements one of these VLMs.

Before that, you should provide some environment variables (these variables should be saved in a `.env` file in the directory where the Python script will be executed):

<details> <summary>Environment variables definition</summary>
    
    <h3>For <code>OpenAI</code>:</h3>

    ```txt
    OPENAI_API_KEY=<your-api-key>
    ```

    <h3>For <code>AzureOpenAI</code>:</h3>

    ```txt
    AZURE_OPENAI_API_KEY=<your-api-key>
    AZURE_OPENAI_ENDPOINT=<your-endpoint>
    AZURE_OPENAI_API_VERSION=<your-api-version>
    AZURE_OPENAI_DEPLOYMENT=<your-model-name>
    ```
</details>

So, the models can be loaded as follows:

<details> <summary><code>OpenAI</code> and <code>AzureOpenAI</code> implementation example</summary>

    <h3>For <code>OpenAI</code></h3>

    ```python
    import os
    from splitter_mr.model import OpenAIVisionModel

    api_key = os.getenv("OPENAI_API_KEY")

    model = OpenAIVisionModel(api_key=api_key)
    ```

    <h3>For <code>AzureOpenAI</code></h3>

    ```python
    import os
    from splitter_mr.model import AzureOpenAIVisionModel

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    model = AzureOpenAIVisionModel(
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=azure_deployment
    )
    ```
</details>

Alternatively, you can instantiate the model and if the `.env` is present, the variables will be get automatically:

```python
from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import MarkItDownReader

file = "data/sample_pdf.pdf"
model = AzureOpenAIVisionModel()
```

Then, you can simply pass the model that you have instantiated to the Reader class:

```python
reader = MarkItDownReader(model=model)
output = reader.read(file)
```

This returns a `ReaderOutput` object with all document text and extracted image descriptions via the vision model. You can access metadata like `output.conversion_method`, `output.reader_method`, `output.ocr_method`, etc.

To retrieve the text content, you can simply access to the `text` attribute:

```python
print(output.text)
```

```md
<!-- page -->

# Description:
# A sample PDF

Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself. PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.

![Illustración 1. SplitterMR logo.](path/to/image)

## 1. Lack of Structural Information

Unlike formats such as HTML or DOCX, PDFs generally do not store information about the logical structure of the document—such as headings, paragraphs, lists, or tables. Instead, PDFs are often a collection of text blocks, images, and graphical elements placed at specific coordinates on a page. This makes it difficult to accurately infer the intended structure, such as determining what text is a heading versus a regular paragraph.

## 2. Variability in PDF Content

PDF files can contain a wide range of content types: plain text, styled text, images, tables, embedded fonts, and even vector graphics. Some PDFs are generated programmatically and have relatively clean underlying text, while others may be created from scans, resulting in image-based (non-selectable) content that requires OCR (Optical Character Recognition) for extraction. The variability in how PDFs are produced leads to inconsistent results when converting to Markdown.

## An enumerate:

1. One

<!-- page -->

# Description:
# 3. Preservation of Formatting

Markdown is a lightweight markup language that supports basic formatting—such as headings, bold, italics, links, images, and lists. However, it does not support all the visual and layout options available in PDF, such as columns, custom fonts, footnotes, floating images, and complex tables. Deciding how (or whether) to preserve these elements can be difficult, and often requires trade-offs between fidelity and simplicity.

\[ f(x) = x^2, \quad x \in [0,1] \]

## An example list:

- Element 1
- Element 2
- Element 3

# 4. Table and Image Extraction

Tables and images in PDFs present a particular challenge. Tables are often visually represented using lines and spacing, with no underlying indication that a group of text blocks is actually a table. Extracting these and converting them to Markdown tables (which have a much simpler syntax) is error-prone. Similarly, extracting images from a PDF and re-inserting them in a way that makes sense in Markdown requires careful handling.

---

This is a cite.

---

# 5. Multicolumn Layouts and Flowing Text

Many PDFs use complex layouts with multiple columns, headers, footers, or sidebars. Converting these layouts to a single-flowing Markdown document requires decisions about reading order and content hierarchy. It's easy to end up with text in the wrong order or to lose important contextual information.

# 6. Encoding and Character Set Issues

PDFs can use a variety of text encodings, embedded fonts, and even contain non-standard Unicode characters. Extracting text reliably without corruption or data loss is not always straightforward, especially for documents with special symbols or non-Latin scripts.

<!-- page -->

# Description:
| Name         | Role         | Email             |
|--------------|--------------|-------------------|
| Alice Smith  | Developer    | alice@example.com  |
| Bob Johnson   | Designer     | bob@example.com    |
| Carol White   | Project Lead | carol@example.com  |

## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.

![Hummingbird](https://example.com/hummingbird.jpg)
```

With the by-default method, you obtain the text extracted from the PDF as it is shown. This method scan the PDF pages as images and process them using a VLM. The result will be a markdown text with all the images detected in every page. Every page is highlighted with a markdown comment as a placeholder: `<!-- page -->`. 

## Experimenting with some keyword arguments

In case that needed, you can pass use other keyword arguments to process the PDFs.

For example, you can customize how to process the images by the VLM using the parameter prompt. For example, in case that you only need an excerpt or a brief description for every page, you can use the following prompt:

```python
output = reader.read(
    file, 
    scan_pdf_pages = True, 
    prompt = "Return only a short description for these pages"
    )
```

In case that needed, it could be interesting split the PDF pages using another placeholder. You can configure that using the `page_placeholder` parameter:

```python
output = reader.read(
    file, scan_pdf_pages = True, 
    prompt = "Return only a short description for these pages", 
    page_placeholder = "## PAGE"
    )
print(output.text)
```

The result will be the expected markdown string:

```md
## PAGE

# Description:
This document discusses the complexities of converting PDF files to other formats like Markdown due to the inherent design of PDFs. It highlights two major challenges: the lack of structural information, which complicates understanding the document's layout, and the variability in PDF content types, which can result in inconsistent extraction results.

## PAGE

# Description:
3. **Preservation of Formatting**: Discusses the limitations of Markdown in preserving complex PDF formatting, such as custom fonts and intricate layouts, and the trade-offs between fidelity and simplicity.

4. **Table and Image Extraction**: Explains the challenges of extracting tables and images from PDFs, emphasizing the difficulties in converting visual representations into a structured format like Markdown.

5. **Multicolumn Layouts and Flowing Text**: Highlights the issues with converting complex PDF layouts into single-column Markdown, addressing potential problems with reading order and content hierarchy.

6. **Encoding and Character Set Issues**: Covers the difficulties in reliably extracting text from PDFs that use various encodings and non-standard characters, including risks of corruption and data loss.

## PAGE

# Description:
The document discusses the challenges involved in converting PDFs to Markdown format, highlighting the need for sophisticated tools that integrate text extraction and document analysis, as well as machine learning techniques. It emphasizes that achieving perfect conversion is rare, often requiring manual review and cleanup. Additionally, it includes a table with team member details and features an image of a hummingbird.
```

In comparison, `MarkItDownReader` offers a faster conversion than Docling but with less options to be configured. In that sense, we cannot obtain directly the `base64` images from every image detected in our documents, or write image placeholders easily (despite we can do it using a prompt). In addition, you will always get a `# Description` placeholder every time you use a VLM for extraction and captioning in this Reader. 

As conclusion, using this reader with a VLM can be useful for those use cases when we need to efficiently extract the text from a PDF. In case that you need the highest reliability or customization, it is not the most suitable option.

## Complete script

```python
import os

from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import MarkItDownReader

file = "data/sample_pdf.pdf"
model = AzureOpenAIVisionModel()
# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "markitdown_output")
os.makedirs(output_dir, exist_ok=True)

def save_markdown(output, filename_base):
    """
    Saves the ReaderOutput.text attribute to a markdown file in the markitdown_output directory.

    Args:
        output (ReaderOutput): The result object returned from DoclingReader.read().
        filename_base (str): A descriptive base name for the file (e.g., 'vlm', 'scan_pages').
    """
    filename = f"{filename_base}.md"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output.text)
    print(f"Saved: {file_path}")

markitdown_reader = MarkItDownReader(model = model)
markitdown_output = markitdown_reader.read(file)
save_markdown(markitdown_output, "vlm")

markitdown_output = markitdown_reader.read(file, scan_pdf_pages = True, prompt = "Return only a short description for these pages", page_placeholder = "## PAGE")
save_markdown(markitdown_output, "custom_vlm")

markitdown_reader = MarkItDownReader()
markitdown_output = markitdown_reader.read(file)
save_markdown(markitdown_output, "no_vlm")
```

!!! note
    For more on available options, see the [**MarkItDownReader class documentation**](../../api_reference/reader.md#markitdownreader).
