# **Example**: Reading a PDF using several Reading methods

Converting a PDF into a readable format is not an easy task. PDF introduces compression, which often results in a complete loss of formatting. As a result, many tools have been developed to convert PDF to text, each of which works differently.

In this example, we will show how to read a PDF file using three readers: `VanillaReader`, `MarkItDownReader`, and `DoclingReader`, and we will observe the differences between each.

!!! note
    A complete description of each of these classes is defined in the [Developer guide](../../api_reference/reader.md).

## 1. Read PDF files using `VanillaReader`

<img src = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/vanilla_reader.svg" alt = "VanillaReader logo" width = 100%>

`VanillaReader` uses open-source libraries to read many file formats, aiming to preserve the text as a string. However, converting a PDF directly to text results in a complete loss of readability. So, to read PDFs, `VanillaReader` uses [PDFPlumber](https://github.com/jsvine/pdfplumber) as the core library. PDFPlumber is a Python library that extracts text, tables, and metadata from PDF files while preserving their layout as much as possible. It is widely used for converting PDF content into readable and structured formats for further processing. Let's see how it works and what results it produces:

First, we instantiate our `VanillaReader` object: 

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
```

To read the file, you simply call to the `read` method:

```python
file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf"
reader_output = reader.read(file)
```

The result will be a `ReaderOutput` object with the following structure:

```python
print(reader_output)
```

```python
ReaderOutput(
    text="\n---\n## Page 1\n---\n\nA sample PDF\nConverting PDF files to other formats, such as Markdown, is a surprisingly\ncomplex tasks ...", 
    document_name='sample_pdf.pdf', 
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf', 
    document_id='2b4a9f04-1b98-40ec-bdae-1d8ddcc652c3', 
    conversion_method='pdf', 
    reader_method='vanilla', 
    ocr_method=None, 
    metadata={}
    )
```

So, we can print the text using this command:

```python
print(reader_output.text)
```

```md
---
## Page 1
---

A sample PDF
Converting PDF files to other formats, such as Markdown, is a surprisingly
complex task due to the nature of the PDF format itself. PDF (Portable
Document Format) was designed primarily for preserving the visual layout of
documents, making them look the same across different devices and
platforms. However, this design goal introduces several challenges when trying to
extract and convert the underlying content into a more flexible, structured format
like Markdown.


--- ![Image]() ---

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


---
## Page 2
---

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


---
## Page 3
---

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


--- ![Image]() ---
```

As we can see from the [original file](https://github.com/andreshere00/Splitter_MR/blob/feature/main/https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf), all the text has been preserved. Bold, italics, etc. are not preserved, nor are text colors, headers, and font type. Despite that, the format is mostly plain text rather than markdown. In addition, we can observe that images are signaled by a `--- ![Image]() ---` placeholder, which can be useful to identify where a image has been placed. The order of the document is preserved.

Now, let's see how well the other readers handle markdown conversion:

## 2. Read PDF files using `MarkItDownReader`

![MarkItDown logo](../../assets/markitdown_logo.png)

The process is analogous to `VanillaReader`. So, we instantiate the `MarkItDownReader` class and we call to the read method:

```python
reader = MarkItDownReader()
reader_output = reader.read(file)

print(reader_output.text)
```

The resulting text is as follows:

```md
A sample PDF

Converting PDF files to other formats, such as Markdown, is a surprisingly
complex task due to the nature of the PDF format itself. PDF (Portable
Document Format) was designed primarily for preserving the visual layout of
documents, making them look the same across different devices and
platforms. However, this design goal introduces several challenges when trying to
extract and convert the underlying content into a more flexible, structured format
like Markdown.

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

1.  One


2.  Two

3.  Three

3. Preservation of Formatting

Markdown is a lightweight markup language that supports basic formatting—
such as headings, bold, italics, links, images, and lists. However, it does not
support all the visual and layout options available in PDF, such as columns,
custom fonts, footnotes, floating images, and complex tables. Deciding how (or
whether) to preserve these elements can be difficult, and often requires trade-
offs between fidelity and simplicity.

𝑓(𝑥) = 𝑥2,

𝑥 ∈ [0,1]

An example list:

•  Element 1
•  Element 2
•  Element 3

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


Role

Name

Email
alice@example.com
Alice Smith  Developer
Bob Johnson  Designer
bob@example.com
Carol White  Project Lead  carol@example.com

Conclusion

While it may seem simple on the surface, converting PDFs to formats like
Markdown involves a series of technical and interpretive challenges. Effective
conversion tools must blend text extraction, document analysis, and sometimes
machine learning techniques (such as OCR or structure recognition) to produce
usable, readable, and faithful Markdown output. As a result, perfect conversion
is rarely possible, and manual review and cleanup are often required.
```

Again, all the text has been preserved. However, we can observe some inconsistencies in line spacing: sometimes there is a single line of separation, while in other cases there are two. Similarly to `VanillaReader`, text formatting has not been preserved: no headers, no italics, no bold... It is simply plain text.

## 3. Read PDF files using `DoclingReader`

![Docling logo](../../assets/docling_logo.png)

`docling` is an open-source Python library designed to analyze and extract structured information from documents, including PDFs. It focuses on preserving the original layout, structure, and semantic elements of documents, making it useful for handling complex formats beyond plain text extraction.

Let's see how it works for this use case:

```md
## A sample PDF

Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.

Ilustración 1. SplitterMR logo.

<!-- image -->

## 1. Lack of Structural Information

Unlike formats such as HTML or DOCX, PDFs generally do not store information about the logical structure of the document -such as headings, paragraphs, lists, or tables. Instead, PDFs are often a collection of text blocks, images, and graphical elements placed at specific coordinates on a page. This makes it difficult to accurately infer the intended structure, such as determining what text is a heading versus a regular paragraph.

## 2. Variability in PDF Content

PDF files can contain a wide range of content types: plain text, styled text, images, tables, embedded fonts, and even vector graphics. Some PDFs are generated programmatically and have relatively clean underlying text, while others may be created from scans, resulting in image-based (nonselectable) content that requires OCR (Optical Character Recognition) for extraction. The variability in how PDFs are produced leads to inconsistent results when converting to Markdown.

An enumerate:

- 1. One
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

| Name        | Role         | Email             |
|-------------|--------------|-------------------|
| Alice Smith | Developer    | alice@example.com |
| Bob Johnson | Designer     | bob@example.com   |
| Carol White | Project Lead | carol@example.com |

## Conclusion

While it may seem simple on the surface, converting PDFs to formats like Markdown involves a series of technical and interpretive challenges. Effective conversion tools must blend text extraction, document analysis, and sometimes machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.

<!-- image -->
```

We can see that the layout is generally better. All the text has been preserved, but markdown format is more present. We can see that headers, tables and lists are markdown formatted, despite bold or italics are not showing. In addition, formulas (`<!-- formula-not-decoded -->`) and images (`<!-- Image -->`) are detected too, despite no description or rendering is provided. Sometimes the line spacing is inconsistent as it was in MarkItDown. However, in general terms, it could be said that it is the method that best formats Markdown.

So, does this mean you should always use this method to parse PDFs? Not exactly. Let's analyze an additional metric: **computation time.**

## 4. Measuring compute time

To measure the compute time for every method, we can encapsulate every reading logic into a function and define a decorator which computes a function execution time. Then, we can compare compute times in relative terms. Then, we can compare compute times in relative terms by executing the following code:

```python
import time

from splitter_mr.reader import DoclingReader, MarkItDownReader, VanillaReader


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Time taken by '{func.__name__}': {elapsed:.4f} seconds\n")
        return result
    return wrapper

@timeit
def get_reader_output(file, reader = VanillaReader()):
    output = reader.read(file)
    print()
    return output.text

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf"

print("*"*20 + " Vanilla Reader " + "*"*20)
vanilla_output = get_reader_output(file, reader = VanillaReader())

print("*"*20 + " MarkItDown Reader " + "*"*20)
markitdown_output = get_reader_output(file, reader = MarkItDownReader())

print("*"*20 + " Docling Reader " + "*"*20)
markitdown_output = get_reader_output(file, reader = DoclingReader())
```

We get the following compute times:

```python
******************** Vanilla Reader ********************

Time taken by 'get_reader_output': 0.1210 seconds

******************** MarkItDown Reader ********************

Time taken by 'get_reader_output': 0.0513 seconds

******************** Docling Reader ********************

Time taken by 'get_reader_output': 6.1602 seconds

```

As we can observe, although DoclingReader offers a really good conversion, it's a resource-intensive method, and therefore takes the longest to return the result. On the other hand, MarkItDownReader, although it preserves the markdown format the least, is the fastest of all. `VanillaReader` offers a balance between computation time and format preservation.

## 5. Comparison between methods

As we've seen, each method has its advantages and disadvantages. Therefore, choosing a reading method depends on the specific needs of the user.

- If you prioritize conversion quality regardless of execution time, `DoclingReader` will be the best option.
- If you want a fast conversion that preserves only the text, `MarkItDownReader` may be your best option.
- If you want a fast conversion but need to detect images and other graphic elements, `VanillaReader` is suitable.

Finally, here we present a comparative table of each method, with the strengths and weaknesses of each one:

| **Feature**                              | `VanillaReader`        | `MarkItDownReader`        | `DoclingReader`            |
| ---------------------------------------- | ---------------------- | ------------------------- | -------------------------- |
| **Header preservation**                  | low                    | mid                       | **high**                   |
| **Text formatting (bold, italic, etc.)** | no                     | no                        | **partial**                |
| **Text color & highlighting**            | no                     | no                        | no                         |
| **Markdown tables**                      | **yes**                | no (txt format)           | **yes**                    |
| **Markdown lists**                       | partial                | no                        | **yes**                    |
| **Images**                               | **With placeholder**   | Without Placeholder       | **With placeholder**       |
| **Formulas**                             | **yes**                | **yes**                   | **yes** (with placeholder) |
| **Pagination**                           | **yes**                | no                        | no                         |
| **Execution time**                       | low                    | **the lowest**            | the highest                |

With this information, we know which method to use. However, there is an element that we have not yet analyzed: the description and annotation of images. Currently, all three methods can describe and annotate images using VLMs. To see how to do this, [jump to the next tutorial](./pdf_with_vlm.md).

**Thanks for Reading!**