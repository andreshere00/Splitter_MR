# **Example**: Splitting Structured Documents by Header Levels with `HeaderSplitter`

Large HTML or Markdown documents often contain multiple sections delineated by headers (`<h1>`, `<h2>`, `#`, `##`, etc.). Chunking these documents by their headers makes them easier to process, search, or send to an LLM. **SplitterMR’s `HeaderSplitter` (or `TagSplitter`) allows you to define *semantic* header levels and split documents accordingly—without manual regex or brittle parsing.**

This Splitter class implements two different Langchain text splitters. See documentation below:

- [HTML Header Text Splitter](https://python.langchain.com/api_reference/text_splitters/html/langchain_text_splitters.html.HTMLHeaderTextSplitter.html)
- [Markdown Header Text Splitter](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)

## Splitting HTML Files

### Step 1: Read an HTML File

We will use the [**`VanillaReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) to load a sample HTML file:


```python
from splitter_mr.reader import VanillaReader

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html"
reader = VanillaReader()  # you can use the argument html_to_markdown=True
reader_output = reader.read(file)

# Print metadata and content
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "<!DOCTYPE html>\n  <html lang='en'>\n  <head>\n    <meta charset='UTF-8'>\n    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n    <title>Fancy Example HTML Page</title>\n  </head>\n  <body>\n    <h1>Main Title</h1>\n    <p>This is an introductory paragraph with some basic content.</p>\n    \n    <h2>Section 1: Introduction</h2>\n    <p>This section introduces the topic. Below is a list:</p>\n    <ul>\n      <li>First item</li>\n      <li>Second item</li>\n
    ...
    /div&gt;\n    </code></pre>\n\n    <h2>Conclusion</h2>\n    <p>This is the conclusion of the document.</p>\n  </body>\n  </html>",
        "document_name": "webpage_example.html",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html",
        "document_id": "a4b4563e-a1d3-4292-92eb-721a27d5367c",
        "conversion_method": "html",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



The `text` attribute contains the raw HTML, including headers, paragraphs, lists, tables, images, and more:


```python
print(reader_output.text)
```

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
          <li>Third item with <stro
    ...
    link.mp4' alt='Example Image'>
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



### Step 2: Split the HTML File by Header Levels

We create a `HeaderSplitter` and specify which semantic headers to split on (e.g., `"Header 1"`, `"Header 2"`, `"Header 3"`). There are up to 6 levels of headers available:


```python
from splitter_mr.splitter import HeaderSplitter

splitter = HeaderSplitter(headers_to_split_on=["Header 1", "Header 2", "Header 3"])
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    html
    Fancy Example HTML Page
    
    ======================================== Chunk 2 ========================================
    # Main Title  
    This is an introductory paragraph with some basic content.
    
    ======================================== Chunk 3 ========================================
    ## Section 1: Introduction  
    This section introduces the topic. Below is a list:  
    - First item
    - Second item
    - Third item wi
    ...
    is section contains an image and a video:  
    ![Example Image](example_image_link.mp4)
    Your browser does not support the video tag.
    
    ======================================== Chunk 6 ========================================
    ## Section 3: Code Example  
    This section contains a code block:  
    ```
    
    <div>
    <p>This is a paragraph inside a div.</p>
    </div>
    
    ```
    
    ======================================== Chunk 7 ========================================
    ## Conclusion  
    This is the conclusion of the document.
    



Each chunk corresponds to a logical section or sub-section in the HTML, grouped by headers and their associated content.


---

## Splitting Markdown File

The exact same interface works for Markdown files. Just change the path:


```python
print("Markdown file example")

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/markdown_example.md"
reader = VanillaReader()
reader_output = reader.read(file)

print(reader_output.model_dump_json(indent=4))
```

    Markdown file example
    {
        "text": "---\n__Advertisement :)__\n\n- __[pica](https://nodeca.github.io/pica/demo/)__ - high quality and fast image\n  resize in browser.\n- __[babelfish](https://github.com/nodeca/babelfish/)__ - developer friendly\n  i18n with plurals support and easy syntax.\n\nYou will like those projects!\n\n---\n\n# h1 Heading 8-)\n## h2 Heading\n### h3 Heading\n#### h4 Heading\n##### h5 Heading\n###### h6 Heading\n\n\n## Horizontal Rules\n\n___\n\n---\n\n***\n\n\n## Typograph
    ...
     Language\n\n### [Custom containers](https://github.com/markdown-it/markdown-it-container)\n\n::: warning\n*here be dragons*\n:::\n",
        "document_name": "markdown_example.md",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/markdown_example.md",
        "document_id": "0be3de0f-2699-4966-9e6c-efdfaf052bfb",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



```python
print(reader_output.text)
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
    
    (c) (C)
    ...
    some code, part of Definition 2 }
    
        Third paragraph of definition 2.
    
    _Compact style:_
    
    Term 1
      ~ Definition 1
    
    Term 2
      ~ Definition 2a
      ~ Definition 2b
    
    
    ### [Abbreviations](https://github.com/markdown-it/markdown-it-abbr)
    
    This is HTML abbreviation example.
    
    It converts "HTML", but keep intact partial entries like "xxxHTMLyyy" and so on.
    
    *[HTML]: Hyper Text Markup Language
    
    ### [Custom containers](https://github.com/markdown-it/markdown-it-container)
    
    ::: warning
    *here be dragons*
    :::
    


The original markdown file is:

```md
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
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

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
    
    ========================
    ...
     some code, part of Definition 2 }  
    Third paragraph of definition 2.  
    _Compact style:_  
    Term 1
    ~ Definition 1  
    Term 2
    ~ Definition 2a
    ~ Definition 2b  
    ### [Abbreviations](https://github.com/markdown-it/markdown-it-abbr)  
    This is HTML abbreviation example.  
    It converts "HTML", but keep intact partial entries like "xxxHTMLyyy" and so on.  
    *[HTML]: Hyper Text Markup Language  
    ### [Custom containers](https://github.com/markdown-it/markdown-it-container)  
    ::: warning
    *here be dragons*
    :::
    



**And that's it!** 

Note that `## h2 Heading` is not picked as an actual header since there is no blankline between `##` and the end of the title. Test with other Headers as your choice!

## Complete Script

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import HeaderSplitter

# Step 1: Read the HTML file
print("HTML file example")
file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/webpage_example.html"
reader = VanillaReader() # you can use the argument html_to_markdown=True to transform directly to markdown
reader_output = reader.read(file)
print(reader_output.model_dump_json(indent=4))
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
print(reader_output.model_dump_json(indent=4))
print(reader_output.text)

splitter = HeaderSplitter(headers_to_split_on=["Header 2"])
splitter_output = splitter.split(reader_output)
for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```
