# Examples

This section illustrates some use cases with **Splitter_MR** to read documents and split them into smaller chunks.

## Text-based splitting

### [How to split recusively](./text/recursive_character_splitter.md)

Divide your text recursively by group of words and sentences, based on the character length as your choice.

### [How to split by characters, words, sentences or paragraphs]()

Divide your text by gramatical groups, with an specific chunk size and with optional chunk overlap.

### [How to split your text by tokens]()

Divide your text to accomplsih your LLM window context using tokenizers such as `Spacy`, `NLTK` and `Tiktoken`.

## Schema-based splitting

### [How to split HTML documents by tags](./schema/html_tag_splitter.md)

Divide the text by tags conserving the HTML schema.

### [How to split JSON files recusively]()

Divide your JSON files into valid smaller serialized objects.

### [How to split by Headers for your Markdown and HTML files]()

Divide your HTML or Markdown files hierarchically by headers.

### [How to split your code scripts]()

Divide your scripts written in Java, Javascript, Python, Go and many more programming languages by syntax blocks.

### [How to split your tables into smaller tables]()

Divide your tables by a fixed number of rows and columns preserving the headers and overall structure.

## PDF Splitting

### [How to split your PDF files conserving graphical elements]()

Divide your PDF files into markdown chunks with information about images and tables, using optionally a Visual Language Model to process these resources.

!!! note
    
    👨‍💻 **Work-in-progress...** More examples to come!