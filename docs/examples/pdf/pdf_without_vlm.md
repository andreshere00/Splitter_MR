# **Example**: Reading a PDF using several Reading methods

Converting a PDF into a readable format is not an easy task. PDF introduces compression, which often results in a complete loss of formatting. As a result, many tools have been developed to convert PDF to text, each of which works differently.

In this example, we will show how to read a PDF file using three readers: [**`VanillaReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader), [**`MarkItDownReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#markitdownreader), and [**`DoclingReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#doclingreader), and we will observe the differences between each.

!!! note
    A complete description of each of these classes is defined in the [Developer guide](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/../api_reference/reader.md).

## 1. Read PDF files using `VanillaReader`

![VanillaReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/vanilla_reader_button.svg#gh-light-mode-only)
![VanillaReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/vanilla_reader_button_white.svg#gh-dark-mode-only)

`VanillaReader` uses open-source libraries to read many file formats, aiming to preserve the text as a string. However, converting a PDF directly to text results in a complete loss of readability. So, to read PDFs, `VanillaReader` uses [PDFPlumber](https://github.com/jsvine/pdfplumber) as the core library. PDFPlumber is a Python library that extracts text, tables, and metadata from PDF files while preserving their layout as much as possible. It is widely used for converting PDF content into readable and structured formats for further processing. Let's see how it works and what results it produces:

First, we instantiate our `VanillaReader` object: 


```python
from splitter_mr.reader import VanillaReader

FILE_PATH: str = "data/sample_pdf.pdf"
reader = VanillaReader()
```


To read the file, you simply call to the `read` method:


```python
reader_output = reader.read(FILE_PATH)
```


The result will be a `ReaderOutput` object with the following structure:


```python
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "<!-- page -->\n\nA sample PDF\nConverting PDF files to other formats, such as Markdown, is a surprisingly\ncomplex task due to the nature of the PDF format itself. PDF (Portable\nDocument Format) was designed primarily for preserving the visual layout of\ndocuments, making them look the same across different devices and\nplatforms. However, this design goal introduces several challenges when trying to\nextract and convert the underlying content into a more flexible, structured for
    ...
    structure recognition) to produce\nusable, readable, and faithful Markdown output. As a result, perfect conversion\nis rarely possible, and manual review and cleanup are often required.\n\n<!-- image -->\n",
        "document_name": "sample_pdf.pdf",
        "document_path": "data/sample_pdf.pdf",
        "document_id": "f4c6b28d-2c05-4025-9781-faf019a2176d",
        "conversion_method": "pdf",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": "<!-- page -->",
        "metadata": {}
    }



So, we can print the text using this command:


```python
print(reader_output.text)
```

    <!-- page -->
    
    A sample PDF
    Converting PDF files to other formats, such as Markdown, is a surprisingly
    complex task due to the nature of the PDF format itself. PDF (Portable
    Document Format) was designed primarily for preserving the visual layout of
    documents, making them look the same across different devices and
    platforms. However, this design goal introduces several challenges when trying to
    extract and convert the underlying content into a more flexible, structured format
    like Markdown.
    
    <!-
    ...
    xample.com |
    
    Conclusion
    While it may seem simple on the surface, converting PDFs to formats like
    Markdown involves a series of technical and interpretive challenges. Effective
    conversion tools must blend text extraction, document analysis, and sometimes
    machine learning techniques (such as OCR or structure recognition) to produce
    usable, readable, and faithful Markdown output. As a result, perfect conversion
    is rarely possible, and manual review and cleanup are often required.
    
    <!-- image -->
    



As we can see from the [original file](https://github.com/andreshere00/Splitter_MR/blob/feature/main/https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/sample_pdf.pdf), all the text has been preserved. Bold, italics, etc. are not highlighted, nor are text colors, headers, and font type. Despite that, the format is mostly plain text rather than markdown. In addition, we can observe that images are signaled by a `<!-- image -->` placeholder, which can be useful to identify where a image has been placed. In the same way, pages are marked with another placeholder: `<!-- page -->`. The order of the document is preserved.

Now, let's see how well the other readers handle markdown conversion:

## 2. Read PDF files using `MarkItDownReader`

![MarkItDownReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button.svg#gh-light-mode-only)
![MarkItDownReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/markitdown_reader_button_white.svg#gh-dark-mode-only)

The process is analogous to `VanillaReader`. So, we instantiate the `MarkItDownReader` class and we call to the read method:


```python
from splitter_mr.reader import MarkItDownReader

reader = MarkItDownReader()
reader_output = reader.read(FILE_PATH)

print(reader_output.text)
```

    A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly
    complex task due to the nature of the PDF format itself. PDF (Portable
    Document Format) was designed primarily for preserving the visual layout of
    documents, making them look the same across different devices and
    platforms. However, this design goal introduces several challenges when trying to
    extract and convert the underlying content into a more flexible, structured format
    like Markdown.
    
    Ilustración 1. Sp
    ...
    ct Lead  carol@example.com
    
    Conclusion
    
    While it may seem simple on the surface, converting PDFs to formats like
    Markdown involves a series of technical and interpretive challenges. Effective
    conversion tools must blend text extraction, document analysis, and sometimes
    machine learning techniques (such as OCR or structure recognition) to produce
    usable, readable, and faithful Markdown output. As a result, perfect conversion
    is rarely possible, and manual review and cleanup are often required.
    
    



Again, all the text has been preserved. However, we can observe some inconsistencies in line spacing: sometimes there is a single line of separation, while in other cases there are two. Similarly to `VanillaReader`, text formatting has not been preserved: no headers, no italics, no bold... It is simply plain text.

## 3. Read PDF files using `DoclingReader`

![DoclingReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button.svg#gh-light-mode-only)
![DoclingReader logo](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/docs/assets/docling_reader_button_white.svg#gh-dark-mode-only)

`docling` is an open-source Python library designed to analyze and extract structured information from documents, including PDFs. It focuses on preserving the original layout, structure, and semantic elements of documents, making it useful for handling complex formats beyond plain text extraction.

Let's see how it works for this use case:


```python
from splitter_mr.reader import DoclingReader

reader = DoclingReader()
reader_output = reader.read(file_path=FILE_PATH)
print(reader_output.text)
```

    2025-10-02 22:07:13,178 - INFO - detected formats: [<InputFormat.PDF: 'pdf'>]
    2025-10-02 22:07:13,207 - INFO - Going to convert document batch...
    2025-10-02 22:07:13,208 - INFO - Initializing pipeline for StandardPdfPipeline with options hash e3309ea8218dc3b978b4932281c99b2a
    2025-10-02 22:07:13,215 - INFO - Loading plugin 'docling_defaults'
    2025-10-02 22:07:13,217 - INFO - Registered ocr engines: ['easyocr', 'ocrmac', 'rapidocr', 'tesserocr', 'tesseract']
    2025-10-02 22:07:13,288 - INFO - Accelerator device: 'mps'
    2025-10-02 22:07:15,367 - INFO - Accelerator device: 'mps'
    2025-10-02 22:07:16,659 - INFO - Accelerator device: 'mps'
    2025-10-02 22:07:17,224 - INFO - Loading plugin 'docling_defaults'
    2025-10-02 22:07:17,225 - INFO - Registered picture descriptions: ['vlm', 'api']
    2025-10-02 22:07:17,225 - INFO - Processing document sample_pdf.pdf
    2025-10-02 22:07:19,270 - INFO - Finished converting document sample_pdf.pdf in 6.09 sec.


    ## A sample PDF
    
    Converting PDF files to other formats, such as Markdown, is a surprisingly complex task due to the nature of the PDF format itself . PDF (Portable Document Format) was designed primarily for preserving the visual layout of documents, making them look the same across different devices and platforms. However, this design goal introduces several challenges when trying to extract and convert the underlying content into a more flexible, structured format like Markdown.
    
    <!-- image --
    ...
    machine learning techniques (such as OCR or structure recognition) to produce usable, readable, and faithful Markdown output. As a result, perfect conversion is rarely possible, and manual review and cleanup are often required.
    
    <!-- image -->
    
    | Name        | Role         | Email             |
    |-------------|--------------|-------------------|
    | Alice Smith | Developer    | alice@example.com |
    | Bob Johnson | Designer     | bob@example.com   |
    | Carol White | Project Lead | carol@example.com |



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
def get_reader_output(file, reader=VanillaReader()):
    output = reader.read(file)
    print()
    return output.text


FILE_PATH = "data/sample_pdf.pdf"

print("*" * 20 + " Vanilla Reader " + "*" * 20)
vanilla_output = get_reader_output(FILE_PATH, reader=VanillaReader())

print("*" * 20 + " MarkItDown Reader " + "*" * 20)
markitdown_output = get_reader_output(FILE_PATH, reader=MarkItDownReader())

print("*" * 20 + " Docling Reader " + "*" * 20)
markitdown_output = get_reader_output(FILE_PATH, reader=DoclingReader())
```

    ******************** Vanilla Reader ********************



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 27
         24 FILE_PATH = "data/sample_pdf.pdf"
         26 print("*" * 20 + " Vanilla Reader " + "*" * 20)
    ---> 27 vanilla_output = get_reader_output(file, reader=VanillaReader())
         29 print("*" * 20 + " MarkItDown Reader " + "*" * 20)
         30 markitdown_output = get_reader_output(file, reader=MarkItDownReader())


    NameError: name 'file' is not defined



As we can observe, although DoclingReader offers a really good conversion, it's a resource-intensive method, and therefore takes the longest to return the result. On the other hand, MarkItDownReader, although it preserves the markdown format the least, is the fastest of all. `VanillaReader` offers a balance between computation time and format preservation.

## 5. Comparison between methods

As we've seen, each method has its advantages and disadvantages. Therefore, choosing a reading method depends on the specific needs of the user.

- If you prioritize conversion quality regardless of execution time, `DoclingReader` will be the best option.
- If you want a fast conversion that preserves only the text, `MarkItDownReader` may be your best option.
- If you want a fast conversion but need to detect images and other graphic elements, `VanillaReader` is suitable.

Finally, here we present a comparative table of each method, with the strengths and weaknesses of each one:

| **Feature**                              | `VanillaReader`        | `MarkItDownReader`                | `DoclingReader`            |
| ---------------------------------------- | ---------------------- | --------------------------------- | -------------------------- |
| **Header preservation**                  | low                    | mid                               | **high**                   |
| **Text formatting (bold, italic, etc.)** | no                     | no                                | **partial**                |
| **Text color & highlighting**            | no                     | no                                | no                         |
| **Markdown tables**                      | **yes**                | no (txt format)                   | **yes**                    |
| **Markdown lists**                       | partial                | no                                | **yes**                    |
| **Image placeholders**                   | **yes**                | no                                | **yes**                    |
| **Formulas placeholders**                | no                     | no                                | **yes**                    |
| **Pagination**                           | **yes**                | **yes** (`split_by_pages = True`) | **yes**                    |
| **Execution time**                       | low                    | **the lowest**                    | the highest                |

With this information, we know which method to use. However, there is an element that we have not yet analyzed: the description and annotation of images. Currently, all three methods can describe and annotate images using VLMs. To see how to do this, [jump to the next tutorial](./pdf_with_vlm.md).

**Thanks for Reading!**
