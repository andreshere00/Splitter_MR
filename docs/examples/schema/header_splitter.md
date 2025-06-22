# **Example**: Splitting Structured Documents by Header Levels with `HeaderSplitter`

Large HTML or Markdown documents often contain multiple sections delineated by headers (`<h1>`, `<h2>`, `#`, `##`, etc.). Chunking these documents by their headers makes them easier to process, search, or send to an LLM. **SplitterMR’s `HeaderSplitter` (or `TagSplitter`) allows you to define *semantic* header levels and split documents accordingly—without manual regex or brittle parsing.**

This Splitter class implements two different Langchain text splitters. See documentation below:

- [HTML Header Text Splitter](https://python.langchain.com/api_reference/text_splitters/html/langchain_text_splitters.html.HTMLHeaderTextSplitter.html)
- [Markdown Header Text Splitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)

## Splitting HTML Files

### Step 1: Read an HTML File

We will use the `VanillaReader` to load a sample HTML file:

```python
from splitter_mr.reader import VanillaReader

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html"
reader = VanillaReader()
reader_output = reader.read(file)

# Print metadata and content
print(reader_output)
print(reader_output.text)
```

**Sample output:**

```python
ReaderOutput(
    text='<!DOCTYPE html> ...',
    document_name='webpage_example.html',
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html',
    document_id='f1773bd6-ec83-4553-a31b-a95c6cd1cbc2',
    conversion_method='html',
    reader_method='vanilla',
    ocr_method=None,
    metadata={}
)
```

The `text` attribute contains the raw HTML, including headers, paragraphs, lists, tables, images, and more:

```html
<!DOCTYPE html>
  <html lang='en'>
  <head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Fancy Example HTML Page</title>
  </head>
  <body>
    <h1>Main Title</h1>
    <p>This is an introductory paragraph with some basic content.</p>
    
    <h2>Section 1: Introduction</h2>
    <p>This section introduces the topic. Below is a list:</p>
    <ul>
      <li>First item</li>
      <li>Second item</li>
      <li>Third item with <strong>bold text</strong> and <a href='#'>a link</a></li>
    </ul>
    
    <h3>Subsection 1.1: Details</h3>
    <p>This subsection provides additional details. Here's a table:</p>
    <table border='1'>
      <thead>
        <tr>
          <th>Header 1</th>
          <th>Header 2</th>
          <th>Header 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Row 1, Cell 1</td>
          <td>Row 1, Cell 2</td>
          <td>Row 1, Cell 3</td>
        </tr>
        <tr>
          <td>Row 2, Cell 1</td>
          <td>Row 2, Cell 2</td>
          <td>Row 2, Cell 3</td>
        </tr>
      </tbody>
    </table>
    
    <h2>Section 2: Media Content</h2>
    <p>This section contains an image and a video:</p>
      <img src='example_image_link.mp4' alt='Example Image'>
      <video controls width='250' src='example_video_link.mp4' type='video/mp4'>
      Your browser does not support the video tag.
    </video>

    <h2>Section 3: Code Example</h2>
    <p>This section contains a code block:</p>
    <pre><code data-lang="html">
    &lt;div&gt;
      &lt;p&gt;This is a paragraph inside a div.&lt;/p&gt;
    &lt;/div&gt;
    </code></pre>

    <h2>Conclusion</h2>
    <p>This is the conclusion of the document.</p>
  </body>
  </html>
```

### Step 2: Split the HTML File by Header Levels

We create a `HeaderSplitter` and specify which semantic headers to split on (e.g., `"Header 1"`, `"Header 2"`, `"Header 3"`). There are up to 6 levels of headers available:

```python
from splitter_mr.splitter import HeaderSplitter

splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2", "Header 3"])
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

Each chunk corresponds to a logical section or sub-section in the HTML, grouped by headers and their associated content.

```python
======================================== Chunk 1 ========================================
Main Title

======================================== Chunk 2 ========================================
This is an introductory paragraph with some basic content.

======================================== Chunk 3 ========================================
Section 1: Introduction

======================================== Chunk 4 ========================================
This section introduces the topic. Below is a list:  
First item  
Second item  
Third item with and  
bold text  
a link
...
```

---

## Splitting Markdown File

The exact same interface works for Markdown files. Just change the path:

```python
print("Markdown file example")

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/markdown_example.md"
reader = VanillaReader()
reader_output = reader.read(file)

print(reader_output)
print(reader_output.text)
```

The original markdown file is:

```
---
__Advertisement :)__

- __[pica](https://nodeca.github.io/pica/demo/)__ - high quality and fast image
  resize in browser.
- __[babelfish](https://github.com/nodeca/babelfish/)__ - developer friendly
  i18n with plurals support and easy syntax.

You will like those projects!

---

# h1 Heading 8-)
## h2 Heading
### h3 Heading
#### h4 Heading
##### h5 Heading
###### h6 Heading


## Horizontal Rules

___

---

***


## Typographic replacements

Enable typographer option to see result.

(c) (C) (r) (R) (tm) (TM) (p) (P) +-

test.. test... test..... test?..... test!....

!!!!!! ???? ,,  -- ---

"Smartypants, double quotes" and 'single quotes'


## Emphasis

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_

~~Strikethrough~~


## Blockquotes


> Blockquotes can also be nested...
>> ...by using additional greater-than signs right next to each other...
> > > ...or with spaces between arrows.

...
```

To split this text by the level 2 headers (`##`), we can use the following instructions:

```python
splitter = HeaderSplitter(headers_to_split_on=["Header 2"])
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

The result will be:

```python

======================================== Chunk 1 ========================================
---
__Advertisement :)__  
- __[pica](https://nodeca.github.io/pica/demo/)__ - high quality and fast image
resize in browser.
- __[babelfish](https://github.com/nodeca/babelfish/)__ - developer friendly
i18n with plurals support and easy syntax.  
You will like those projects!  
---  
# h1 Heading 8-)  
## h2 Heading
### h3 Heading
#### h4 Heading
##### h5 Heading
###### h6 Heading

======================================== Chunk 2 ========================================
## Horizontal Rules  
___  
---  
***

======================================== Chunk 3 ========================================
## Typographic replacements  
Enable typographer option to see result.  
(c) (C) (r) (R) (tm) (TM) (p) (P) +-  
test.. test... test..... test?..... test!....  
!!!!!! ???? ,,  -- ---  
"Smartypants, double quotes" and 'single quotes'

======================================== Chunk 4 ========================================
## Emphasis  
**This is bold text**  
__This is bold text__  
*This is italic text*  
_This is italic text_  
~~Strikethrough~~
```

**And that's it!** Note that `## h2 Heading` since it is not a blankline between `##` and the end of the title. Test with other Headers as your choice!

## Complete Script

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import HeaderSplitter

# Step 1: Read the HTML file
print("HTML file example")
file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html"
reader = VanillaReader()
reader_output = reader.read(file)
print(reader_output)
print(reader_output.text)

splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2", "Header 3"])
splitter_output = splitter.split(reader_output)
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")

# Step 2: Read the Markdown file
print("Markdown file example")
file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/markdown_example.md"
reader = VanillaReader()
reader_output = reader.read(file)
print(reader_output)
print(reader_output.text)

splitter = HeaderSplitter(headers_to_split_on=["Header 2"])
splitter_output = splitter.split(reader_output)
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```