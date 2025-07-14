# **Example:** Read PDF documents with images using Docling Reader

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button.svg#only-light" alt="DoclingReader logo">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button_white.svg#only-dark" alt="DoclingReader logo">
</p>

As we have seen in previous examples, reading a PDF is not a simple task. In this case, we will see how to read a PDF using Docling framework, and connect this library into Visual Language Models to extract text or get annotations from images.

## Connecting to a VLM to extract text and analyze images

For this example, we will use the same document as the (previous tutorial)[https://github.com/andreshere00/Splitter_MR/blob/main/data/sample_pdf.pdf].

To use a VLM to read images and get annotations, you can simply instantiate a model which inherits from a `BaseModel` class and pass that model into the Reader class. 

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
from splitter_mr.reader import DoclingReader

file = "data/sample_pdf.pdf"

model = AzureOpenAIVisionModel()
```

Then, use the `read` method of this object and read a file as always. Once detected that the file is PDF, it will return a ReaderOutput object containing the extracted text.

```python
# 1. Read PDF using a Visual Language Model

print("="* 80 + " DoclingReader with VLM " + "="*80)
docling_reader = DoclingReader(model = model)
docling_output = docling_reader.read(file)

# Get Docling ReaderOutput
print(docling_output)

# Get text attribute from Docling Reader
print(docling_output.text)
```
```python
ReaderOutput(
    text="## A sample PDF\n\nConverting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . ...", 
    document_name='sample_pdf.pdf', 
    document_path='data/sample_pdf.pdf',
    document_id='0c4f4831-2b77-4448-8906-739a6fda8aa9',
    conversion_method='markdown',
    reader_method='docling',
    ocr_method='gpt-4o-mini',
    metadata=None)
```

As we can see, the PDF contents along with some metadata information such as the `conversion_method`, `reader_method` or the `ocr_method` have been retrieved. To get the PDF contents, you can simply access to the `text` attribute as always:

```python
## A sample PDF

Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.

Ilustración 1. SplitterMR logo.

<!-- image -->
*Caption: SplitterMR is a tool designed to segment and prepare textual documents for efficient processing in production-level large language model applications.*

## 1. Lack of Structural Information

Unlike formats such as HTML or DOCX, PDFs generally do not store information about the logical structure of the document -such as headings, paragraphs, lists, or tables. Instead, PDFs are often a collection of text blocks, images, and graphical elements placed at specific coordinates on a page. This makes it difficult to accurately infer the intended structure, such as determining what text is a heading versus a regular paragraph.

## 2. Variability in PDF Content

PDF files can contain a wide range of content types: plain text, styled text, images, tables, embedded fonts, and even vector graphics. Some PDFs are generated programmatically and have relatively clean underlying text, while others may be created from scans, resulting in image-based (nonselectable) content that requires OCR (Optical Character Recognition) for extraction. The variability in how PDFs are produced leads to inconsistent results when converting to Markdown.

An enumerate:

- 1. One
<!-- page -->
- 2. Two
- 3. Three

## 3. Preservation of Formatting

Markdown is a lightweight markup language that supports basic formatting -such as headings, bold, italics, links, images, and lists. However, it does not support all the visual and layout options available in PDF, such as columns, custom fonts, footnotes, floating images, and complex tables. Deciding how (or whether) to preserve these elements can be difficult, and often requires tradeoffs between fidelity and simplicity.

<!-- formula-not-decoded -->

## An example list:

- · Element 1
- · Element 2
- · Element 3

## 4. Table and Image Extraction

Tables and images in PDFs present a particular challenge. Tables are often visually represented using lines and spacing, with no underlying indication that a group of text blocks is actually a table. Extracting these and converting them to Markdown tables (which have a much simpler syntax) is error-prone. Similarly, extracting images from a PDF and re-inserting them in a way that makes sense in Markdown requires careful handling.

This is a cite.

## 5. Multicolumn Layouts and Flowing Text

Many PDFs use complex layouts with multiple columns, headers, footers, or sidebars. Converting these layouts to a single-flowing Markdown document requires decisions about reading order and content hierarchy. It's easy to end up with text in the wrong order or to lose important contextual information.

## 6. Encoding and Character Set Issues

PDFs can use a variety of text encodings, embedded fonts, and even contain nonstandard Unicode characters. Extracting text reliably without corruption or data loss is not always straightforward, especially for documents with special symbols or non-Latin scripts.

<!-- page -->

| Name        | Role         | Email             |
|-------------|--------------|-------------------|
| Alice Smith | Developer    | alice@example.com |
| Bob Johnson | Designer     | bob@example.com   |
| Carol White | Project Lead | carol@example.com |

## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.

<!-- image -->
*Caption: A vibrant hummingbird hovering delicately in front of a bright orange flower, showcasing the beauty of nature and the intricate interplay between pollinators and plants.*
```

As seen, all the images have been described using a caption. 

## Experimenting with some keyword arguments

In case that you have additional requirements to describe these images, you can provide a prompt via a `prompt` argument:

```python
docling_output = docling_reader.read(file, prompt = "Describe the image briefly in Spanish.")

print(docling_output.text)
```
```python
...
## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.


La imagen muestra un colibrí de plumaje verde brillante posado cerca de una flor naranja. El colibrí se encuentra en vuelo, con sus alas extendidas, mientras recoge néctar de la flor. El fondo es difuso, lo que resalta la belleza del ave y la flor.
```

You can read the PDF scanning the pages as images and extracting its content. To do so, enable the option `scan_pdf_pages = True`. In case that you want to change the placeholder, you can do it passing the keyword argument `placeholder = <your desired placeholder>`.

Finally, it could be interesting extract the markdown text with the images as embedded content. In that case, activate the option `show_base64_images`. In that case, it is not necessary to pass the model to the Reader class.

```python
docling_reader = DoclingReader()
docling_output = docling_reader.read(file, show_base64_images = True)

print(docling_output.text)
```
```python
...
## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAasAAAEd...)
```

Of course, remember that the use of a VLM is not mandatory, and you can read the PDF obtaining most of the information.

## Complete script

```python
from splitter_mr.model import AzureOpenAIVisionModel
from splitter_mr.reader import DoclingReader

file = "data/sample_pdf.pdf"

model = AzureOpenAIVisionModel()
docling_reader = DoclingReader(model = model)

# 1. Read PDF using a Visual Language Model

docling_output = docling_reader.read(file)
print(docling_output)  # Get Docling ReaderOutput
print(docling_output.text)  # Get text attribute from Docling Reader

# 2. Describe the images using a custom prompt

docling_output = docling_reader.read(file, prompt = "Describe the image briefly in Spanish.")
print(docling_output.text)

# 3. Scan PDF pages 

docling_output = docling_reader.read(file, scan_pdf_pages = True)
print(docling_output.text)

# 4. Extract images as embedded content

docling_reader = DoclingReader()
docling_output = docling_reader.read(file, show_base64_images = True)
print(docling_output.text)
```

!!! note
    For more on available options, see the [**DoclingReader class documentation**](../../api_reference/reader.md#doclingreader).