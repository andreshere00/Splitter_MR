# Examples

This section illustrates some use cases with **SplitterMR** to read documents and split them into smaller chunks.

## Text-based splitting

### [How to split recusively](./text/recursive_character_splitter.md)

Divide your text recursively by group of words and sentences, based on the character length as your choice.

### [How to split by characters, words, sentences or paragraphs](./text/fixed_splitter.md)

Divide your text by gramatical groups, with an specific chunk size and with optional chunk overlap.

### [How to split your text by tokens](./text/token_splitter.md)

Divide your text to accomplsih your LLM window context using tokenizers such as `Spacy`, `NLTK` and `Tiktoken`.

## Schema-based splitting

### [How to split HTML documents by tags](./schema/html_tag_splitter.md)

Divide the text by tags conserving the HTML schema.

### [How to split JSON files recusively](./schema/json_splitter.md)

Divide your JSON files into valid smaller serialized objects.

### [How to split by Headers for your Markdown and HTML files](./schema/html_tag_splitter.md)

Divide your HTML or Markdown files hierarchically by headers.

### [How to split your code scripts](./schema/code_splitter.md)

Divide your scripts written in Java, Javascript, Python, Go and many more programming languages by syntax blocks.

### [How to split your tables into smaller tables](./schema/row_column_splitter.md)

Divide your tables by a fixed number of rows and columns preserving the headers and overall structure.

## Reading files

### [How to read a PDF file without image processing](./pdf/pdf_without_vlm.md)

Read your PDF files using three frameworks: `PDFPlumber`, `MarkItDown` and `Docling`. 

### [How to read a PDF file using Vanilla Reader for image processing](./pdf/pdf_vanilla.md)

Read your PDF files and analyze the content using PDFPlumber and Visual Language Models.

### [How to read a PDF file using Docling for image processing](./pdf/pdf_docling.md)

Read your PDF files and analyze the content using Docling and Visual Language Models.

### [How to read a PDF file using MarkItDown for image processing](./pdf/pdf_markitdown.md)

Read your PDF files and analyze the content using MarkItDown and Visual Language Models.

!!! note
    
    👨‍💻 **Work-in-progress...** More examples to come!

## Other examples

### [Deprecated] - [How to read PDFs and analyze its content using several Visual Language Models](./pdf/pdf_with_vlm.md)

Examples about how to read PDF files using PDFPlumber, MarkItDown and Docling frameworks, and read its multimedia content using a visual language model.