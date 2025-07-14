# **Example:** Read PDF documents with images using Vanilla Reader

<p style="text-align:center;">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/vanilla_reader_button.svg#only-light" alt="VanillaReader logo">
<img src="https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/vanilla_reader_button_white.svg#only-dark" alt="VanillaReader logo">
</p>

In this tutorial we will see how to read a PDF using our custom component, which is based on **PDFPlumber**. Then, we will connect this reader component into Visual Language Models to extract text or get annotations from images inside the PDF. In addition, we will explore which options we have to analyze and extract the content of the PDF in a custom, fast and a comprehensive way. Let's dive in.

!!! note
    Remember that you can access to the complete documentation of this Reader Component in the [**Developer Guide**](../../api_reference/reader.md#vanillareader).

## How to connect a VLM to MarkItDownReader

For this tutorial, we will use the same data as the first tutorial. [**Consult reference here**](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf).

Currently, two models are supported, both from OpenAI: the regular client, **OpenAI** and the available deployments from **Azure**. Hence, you can instantiate wherever you want to your project, or create a new one using as reference the [BaseModel abstract class](../../api_reference/model.md#basemodel).

Before instantiating the model, you should provide connection parameters. These connections parameters are loaded from environment variables (you can save them in a `.env` in the root of the project or script that you will execute). Consult these snippets:

<details> <summary><b>Environment variables definition</b></summary>
    
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

After that, you can explicitly declare the connection parameters as follows:

<details> <summary><code>OpenAI</code> and <code>AzureOpenAI</code> <b>implementation example</b></summary>

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

Or, alternatively, if you have saved the environment variables as indicated above, you can simply instantiate the model without explicit parameters. For this tutorial, we will use an `AzureOpenAI` deployment.

```python
from splitter_mr.model import AzureOpenAIVisionModel

model = AzureOpenAIVisionModel()
```

Then, use the Reader component and insert the model as parameter:

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader(model = model)
```

Then, you can read the file. The result will be an object from the type `ReaderOutput`, which is a dictionary containing some metadata about the file. To get the content, you can access to the `text` attribute:

```python
file = "data/sample_pdf.pdf"

output = reader.read(file_path = file)
print(output.text)
```

As observed, all the images have been described by the LLM:

```md
<!-- page -->

A sample PDF
Converting PDF files to other formats, such as Markdown, is a surprisingly
complex task due to the nature of the PDF format itself. PDF (Portable
Document Format) was designed primarily for preserving the visual layout of
documents, making them look the same across different devices and
platforms. However, this design goal introduces several challenges when trying to
extract and convert the underlying content into a more flexible, structured format
like Markdown.

<!-- image -->
*Caption: SplitterMR: A tool designed to efficiently chunk document text for seamless integration into production-ready large language model applications.*

Ilustración 1. SplitterMR logo.
1. Lack of Structural Information
Unlike formats such as HTML or DOCX, PDFs generally do not store
information about the logical structure of the document—such as
headings, paragraphs, lists, or tables. Instead, PDFs are often a collection
of text blocks, images, and graphical elements placed at specific
coordinates on a page. This makes it difficult to accurately infer the
intended structure, such as determining what text is a heading versus a
regular paragraph.
2. Variability in PDF Content
PDF files can contain a wide range of content types: plain text, styled text,
images, tables, embedded fonts, and even vector graphics. Some PDFs
are generated programmatically and have relatively clean underlying text,
while others may be created from scans, resulting in image-based (non-
selectable) content that requires OCR (Optical Character Recognition) for
extraction. The variability in how PDFs are produced leads to inconsistent
results when converting to Markdown.
An enumerate:
1. One

<!-- page -->

2. Two
3. Three
3. Preservation of Formatting
Markdown is a lightweight markup language that supports basic formatting—
such as headings, bold, italics, links, images, and lists. However, it does not
support all the visual and layout options available in PDF, such as columns,
custom fonts, footnotes, floating images, and complex tables. Deciding how (or
whether) to preserve these elements can be difficult, and often requires trade-
offs between fidelity and simplicity.
𝑥2,
𝑓(𝑥)
= 𝑥 ∈ [0,1]
An example list:
• Element 1
• Element 2
• Element 3
4. Table and Image Extraction
Tables and images in PDFs present a particular challenge. Tables are often
visually represented using lines and spacing, with no underlying indication that
a group of text blocks is actually a table. Extracting these and converting them
to Markdown tables (which have a much simpler syntax) is error-prone.
Similarly, extracting images from a PDF and re-inserting them in a way that
makes sense in Markdown requires careful handling.
This is a cite.
5. Multicolumn Layouts and Flowing Text
Many PDFs use complex layouts with multiple columns, headers, footers, or sidebars.
Converting these layouts to a single-flowing Markdown document requires decisions
about reading order and content hierarchy. It's easy to end up with text in the wrong
order or to lose important contextual information.
6. Encoding and Character Set Issues
PDFs can use a variety of text encodings, embedded fonts, and even contain non-
standard Unicode characters. Extracting text reliably without corruption or data loss is
not always straightforward, especially for documents with special symbols or non-Latin
scripts.

<!-- page -->

| Name | Role | Email |
| --- | --- | --- |
| Alice Smith | Developer | alice@example.com |
| Bob Johnson | Designer | bob@example.com |
| Carol White | Project Lead | carol@example.com |

Conclusion
While it may seem simple on the surface, converting PDFs to formats like
Markdown involves a series of technical and interpretive challenges. Effective
conversion tools must blend text extraction, document analysis, and sometimes
machine learning techniques (such as OCR or structure recognition) to produce
usable, readable, and faithful Markdown output. As a result, perfect conversion
is rarely possible, and manual review and cleanup are often required.

<!-- image -->
*Caption: A vibrant hummingbird gracefully hovers in front of a bright yellow flower, showcasing its dazzling plumage and agility as it seeks nectar.*
```

## Experimenting with some keyword arguments

Suppose that you need to simply get the base64 images from the file. Then, you can use the option `show_base64_images` to get those images:

```python
reader = VanillaReader()
output = reader.read(file_path = file, show_base64_images = True)
print(output.text)
```

```md
<!-- page -->

A sample PDF
Converting PDF files to other formats, such as Markdown, is a surprisingly
complex task due to the nature of the PDF format itself. PDF (Portable
Document Format) was designed primarily for preserving the visual layout of
documents, making them look the same across different devices and
platforms. However, this design goal introduces several challenges when trying to
extract and convert the underlying content into a more flexible, structured format
like Markdown.

![Image page 1](data:image/png;base64,iVBORw0KG...=)

...

<!-- page -->

...

<!-- page -->

| Name | Role | Email |
| --- | --- | --- |
| Alice Smith | Developer | alice@example.com |
| Bob Johnson | Designer | bob@example.com |
| Carol White | Project Lead | carol@example.com |

Conclusion
While it may seem simple on the surface, converting PDFs to formats like
Markdown involves a series of technical and interpretive challenges. Effective
conversion tools must blend text extraction, document analysis, and sometimes
machine learning techniques (such as OCR or structure recognition) to produce
usable, readable, and faithful Markdown output. As a result, perfect conversion
is rarely possible, and manual review and cleanup are often required.

![Image page 3](data:image/png;base64,iVBORw0KGgo..)

```

In addition, you can modify how the image and page placeholders are generated with the options `image_placeholder` and `page_placeholder`. *Note that in this case we are not using any VLM*.

```python
reader = VanillaReader()
output = reader.read(file_path = file, image_placeholder = "## Image", page_placeholder = "## Page")
print(output.text)
```

```md
## Page

A sample PDF
Converting PDF files to other formats, such as Markdown, is a surprisingly
complex task due to the nature of the PDF format itself. PDF (Portable
Document Format) was designed primarily for preserving the visual layout of
documents, making them look the same across different devices and
platforms. However, this design goal introduces several challenges when trying to
extract and convert the underlying content into a more flexible, structured format
like Markdown.

## Image

Ilustración 1. SplitterMR logo.
1. Lack of Structural Information
Unlike formats such as HTML or DOCX, PDFs generally do not store
information about the logical structure of the document—such as
headings, paragraphs, lists, or tables. Instead, PDFs are often a collection
of text blocks, images, and graphical elements placed at specific
coordinates on a page. This makes it difficult to accurately infer the
intended structure, such as determining what text is a heading versus a
regular paragraph.
2. Variability in PDF Content
PDF files can contain a wide range of content types: plain text, styled text,
images, tables, embedded fonts, and even vector graphics. Some PDFs
are generated programmatically and have relatively clean underlying text,
while others may be created from scans, resulting in image-based (non-
selectable) content that requires OCR (Optical Character Recognition) for
extraction. The variability in how PDFs are produced leads to inconsistent
results when converting to Markdown.
An enumerate:
1. One

## Page

2. Two
3. Three
3. Preservation of Formatting
Markdown is a lightweight markup language that supports basic formatting—
such as headings, bold, italics, links, images, and lists. However, it does not
support all the visual and layout options available in PDF, such as columns,
custom fonts, footnotes, floating images, and complex tables. Deciding how (or
whether) to preserve these elements can be difficult, and often requires trade-
offs between fidelity and simplicity.
𝑥2,
𝑓(𝑥)
= 𝑥 ∈ [0,1]
An example list:
• Element 1
• Element 2
• Element 3
4. Table and Image Extraction
Tables and images in PDFs present a particular challenge. Tables are often
visually represented using lines and spacing, with no underlying indication that
a group of text blocks is actually a table. Extracting these and converting them
to Markdown tables (which have a much simpler syntax) is error-prone.
Similarly, extracting images from a PDF and re-inserting them in a way that
makes sense in Markdown requires careful handling.
This is a cite.
5. Multicolumn Layouts and Flowing Text
Many PDFs use complex layouts with multiple columns, headers, footers, or sidebars.
Converting these layouts to a single-flowing Markdown document requires decisions
about reading order and content hierarchy. It's easy to end up with text in the wrong
order or to lose important contextual information.
6. Encoding and Character Set Issues
PDFs can use a variety of text encodings, embedded fonts, and even contain non-
standard Unicode characters. Extracting text reliably without corruption or data loss is
not always straightforward, especially for documents with special symbols or non-Latin
scripts.

## Page

| Name | Role | Email |
| --- | --- | --- |
| Alice Smith | Developer | alice@example.com |
| Bob Johnson | Designer | bob@example.com |
| Carol White | Project Lead | carol@example.com |

Conclusion
While it may seem simple on the surface, converting PDFs to formats like
Markdown involves a series of technical and interpretive challenges. Effective
conversion tools must blend text extraction, document analysis, and sometimes
machine learning techniques (such as OCR or structure recognition) to produce
usable, readable, and faithful Markdown output. As a result, perfect conversion
is rarely possible, and manual review and cleanup are often required.

## Image
```

But one of the most important features is to scan the PDF as PageImages, to analyze every page with a VLM to extract the content. In order to do that, you can simply activate the option `scan_pdf_pages`. 

```python
reader = VanillaReader(model = model)
output = reader.read(file_path = file, scan_pdf_pages = True)
print(output.text)
```

```md
<!-- page -->

# A sample PDF

Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself. PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.

![Illustración 1. SplitterMR logo](https://dummyimage.com/600x400/000/fff)

## 1. Lack of Structural Information

Unlike formats such as HTML or DOCX, PDFs generally do not store information about the logical structure of the document—such as headings, paragraphs, lists, or tables. Instead, PDFs are often a collection of text blocks, images, and graphical elements placed at specific coordinates on a page. This makes it difficult to accurately infer the intended structure, such as determining what text is a heading versus a regular paragraph.

## 2. Variability in PDF Content

PDF files can contain a wide range of content types: plain text, styled text, images, tables, embedded fonts, and even vector graphics. Some PDFs are generated programmatically and have relatively clean underlying text, while others may be created from scans, resulting in image-based (non-selectable) content that requires OCR (Optical Character Recognition) for extraction. The variability in how PDFs are produced leads to inconsistent results when converting to Markdown.

### An enumerate:
1. One

---

<!-- page -->

1. Two  
2. Three  

## 3. Preservation of Formatting  
Markdown is a lightweight markup language that supports basic formatting – such as headings, bold, italics, links, images, and lists. However, it does not support all the visual and layout options available in PDF, such as columns, custom fonts, footnotes, floating images, and complex tables. Deciding how (or whether) to preserve these elements can be difficult, and often requires trade-offs between fidelity and simplicity.

$$f(x) = x^2, \quad x \in [0,1]$$

### An example list:
- Element 1
- Element 2
- Element 3  

## 4. Table and Image Extraction  
Tables and images in PDFs present a particular challenge. Tables are often visually represented using lines and spacing, with no underlying indication that a group of text blocks is actually a table. Extracting these and converting them to Markdown tables (which have a much simpler syntax) is error-prone. Similarly, extracting images from a PDF and re-inserting them in a way that makes sense in Markdown requires careful handling.

---

This is a cite.

## 5. Multicolumn Layouts and Flowing Text  
Many PDFs use complex layouts with multiple columns, headers, footers, or sidebars. Converting these layouts to a single-flowing Markdown document requires decisions about reading order and content hierarchy. It's easy to end up with text in the wrong order or to lose important contextual information.  

## 6. Encoding and Character Set Issues  
PDFs can use a variety of text encodings, embedded fonts, and even contain non-standard Unicode characters. Extracting text reliably without corruption or data loss is not always straightforward, especially for documents with special symbols or non-Latin scripts.

---

<!-- page -->

| Name          | Role        | Email                |
|---------------|-------------|----------------------|
| Alice Smith   | Developer   | alice@example.com    |
| Bob Johnson    | Designer    | bob@example.com      |
| Carol White   | Project Lead| carol@example.com    |

## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.

![Hummingbird](<image_url_here>)
```

Remember that you can always customize the prompt to get one or other results using the parameter `prompt`:

```python
reader = VanillaReader(model = model)
output = reader.read(file_path = file, prompt = "Extract the content of this resource in html format")
print(output.text)
```

```html
<!-- page -->

A sample PDF
...

<!-- image -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SplitterMR</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f4;
            padding: 50px;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        h1 {
            color: #333;
            font-size: 24px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <img src="https://www.example.com/path/to/your/logo.png" alt="SplitterMR Logo">
    <h1>SplitterMR</h1>
    <p>Chunk your documents text for production-ready LLM applications.</p>
</body>
</html>

...

<!-- image -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hummingbird</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
            margin: 0;
        }
        img {
            width: 590px;
            height: 886px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
    </style>
</head>
<body>
    <img src="data:image/gif;base64,R0lGODlhBQAIAFIAAAAAAP///4G+tobTzGWShbxPetW2tZasWJiYgA2pQAAOw==" alt="Hummingbird">
</body>
</html>
```

To sum up, we can see that `VanillaReader` is a good option to extract rapidly and efficiently the text content for a PDF file. Remember that you can customize how the extraction is performed. But remember to consult other reading options in the [Developer guide](../../api_reference/reader.md) or [other tutorials](../examples.md).

Thank you so much for reading :).

## Complete script

```python
import os
from splitter_mr.reader import VanillaReader
from splitter_mr.model import AzureOpenAIVisionModel

file = "data/sample_pdf.pdf"
output_dir = "tmp/vanilla_output"
os.makedirs(output_dir, exist_ok=True)

model = AzureOpenAIVisionModel()

# 1. Default with model
reader = VanillaReader(model=model)
output = reader.read(file_path=file)
with open(os.path.join(output_dir, "output_with_model.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 2. Default without model, with base64 images shown
reader = VanillaReader()
output = reader.read(file_path=file, show_base64_images=True)
with open(os.path.join(output_dir, "output_with_base64_images.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 3. Default without model, with placeholders
reader = VanillaReader()
output = reader.read(file_path=file, image_placeholder="## Image", page_placeholder="## Page")
with open(os.path.join(output_dir, "output_with_placeholders.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 4. With model, scan PDF pages
reader = VanillaReader(model=model)
output = reader.read(file_path=file, scan_pdf_pages=True)
with open(os.path.join(output_dir, "output_scan_pdf_pages.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)

# 5. With model, custom prompt
reader = VanillaReader(model=model)
output = reader.read(file_path=file, prompt="Extract the content of this resource in html format")
with open(os.path.join(output_dir, "output_html_prompt.txt"), "w", encoding="utf-8") as f:
    f.write(output.text)
```