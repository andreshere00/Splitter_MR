# Example: Split *Pinocchio* Chapters Using `KeywordSplitter`

![Example of regex patterns](https://miro.medium.com/1*u4NbHgv-TzEE5G6xXnHCbQ.png)

The [**`KeywordSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#keywordsplitter) is a highly versatile text-splitting utility that **divides documents based on custom regular-expression (regex) patterns**. Unlike simpler splitters that break text by fixed lengths (characters, words, or sentences), the KeywordSplitter allows you to identify semantic boundaries—such as headings, section markers, or unique keywords—and chunk your text accordingly.

This splitter is particularly useful when working with structured or semi-structured text where meaningful sections are defined by repeated markers. Examples include:

- Splitting classic books or plays by chapter headings (e.g., "CHAPTER I", "ACT II").
- Parsing logs by timestamps or error codes.
- Extracting sections from Markdown or plain text documents where a specific keyword (e.g., "TODO" or "NOTE") separates ideas.
- Segmenting transcripts or interview notes at speaker identifiers.

This notebook demonstrates splitting *Pinocchio* by chapter markers such as **“CHAPTER I”**, **“CHAPTER II”**, etc., using the `KeywordSplitter`.


## Step 1: Read the text using a Reader component

We will use the `VanillaReader` to fetch the text from Project Gutenberg:



```python
from splitter_mr.reader import VanillaReader

FILE_PATH = "https://www.gutenberg.org/cache/epub/16865/pg16865.txt"

reader = VanillaReader()
reader_output = reader.read(file_path=FILE_PATH)

print(reader_output.model_dump_json(indent=4))
print(reader_output.text[:1000])  # preview first 1000 characters
```

    {
        "text": "﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook or online\r\nat www.gutenberg.org. If you are not located in the United States,\r\nyou will have to check the laws of the country
    ...
    you are located
    before using this eBook.
    
    Title: Pinocchio: The Tale of a Puppet
    
    Author: Carlo Collodi
    
    Illustrator: Alice Carsey
    
    Release date: October 13, 2005 [eBook #16865]
                    Most recently updated: December 12, 2020
    
    Language: English
    
    Credits: Produced by Mark C. Orton, Melissa Er-Raqabi and the Online
            Distributed Proofreading Team at https://www.pgdp.net.
    
    
    *** START OF THE PROJECT GUTENBERG EBOOK PINOCCHIO: THE TALE OF A PUPPET ***
    
    
    
    
    Produced by Mark C. Orton, Me


## Step 2: Split the text using `KeywordSplitter`

We define the regex pattern for Roman numeral chapters and instantiate the splitter:



```python
import re
from splitter_mr.splitter import KeywordSplitter

chapter_pattern = r"CHAPTER\s+[IVXLCDM]+"
splitter = KeywordSplitter(
    patterns={"chapter": chapter_pattern},
    include_delimiters="after",
    flags=re.IGNORECASE,
)

splitter_output = splitter.split(reader_output)
print(splitter_output.model_dump_json(indent=4))
```

    {
        "chunks": [
            "﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook or online\r\nat www.gutenberg.org. If you are not located in the United States,\r\nyou will have to check the laws of
    ...
    
                    ],
                    [
                        203258,
                        203271
                    ],
                    [
                        214484,
                        214496
                    ],
                    [
                        222566,
                        222579
                    ]
                ],
                "include_delimiters": "after",
                "flags": 2,
                "pattern_names": [
                    "chapter"
                ],
                "chunk_size": 100000
            }
        }
    }


As you can see, all the chunks start with `CHAPTER` + a roman number, so the Splitter has divided the text by chapters correctly.

## Step 3: Visualize the chunks

To visualize the chunks, you can execute the following code:


```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40)
    print(chunk[:800] + "\n")
```

    ======================================== Chunk 1 ========================================
    ﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet
        
    This ebook is for the use of anyone anywhere in the United States and
    most other parts of the world at no cost and with almost no restrictions
    whatsoever. You may copy it, give it away or re-use it under the terms
    of the Project Gutenberg License included with this ebook or online
    at www.gutenberg.org. If you are not located in the United 
    ...
    e
    fever.
    
    Was he trembling from cold or from fear. Perhaps a little from both the
    one and the other. But Pinocchio, thinking it was from fear, said, to
    comfort him:
    
    "Courage, papa! In a few minutes we shall be safely on shore."
    
    "But where is this blessed shore?" asked the little old man, becoming
    still more frightened, and screwing up his eyes as tailors do when they
    wish to thread a needle. "I have been looking in every direction and I
    see nothing but the sky and the sea."
    
    "But I see the s
    


## Step 4: Analyze metadata


```python
meta = splitter_output.metadata["keyword_matches"]
print("Chapter matches:", meta["counts"])
print("Spans:", meta["spans"][:5])
print("Splitter parameters:", splitter_output.split_params)
```

    Chapter matches: {'chapter': 36}
    Spans: [(7063, 7072), (10320, 10330), (14085, 14096), (20136, 20146), (23487, 23496)]
    Splitter parameters: {'include_delimiters': 'after', 'flags': re.IGNORECASE, 'chunk_size': 100000, 'pattern_names': ['chapter']}


You can also verify the configuration:


```python
print(splitter_output.split_params)
```

    {'include_delimiters': 'after', 'flags': re.IGNORECASE, 'chunk_size': 100000, 'pattern_names': ['chapter']}


!!! note

    You can experiment with:
    * **`include_delimiters`** = `"after"`, `"both"`, or `"none"` to control delimiter placement.
    * Adjust `chunk_size` for soft wrapping inside chapters.
    * Use multiple patterns (dict or list) to split by different keywords simultaneously.

**And that’s it!** This notebook demonstrates how to segment *Pinocchio* into chapters interactively with [`KeywordSplitter`](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#keywordsplitter).

## Complete script


```python
import re
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import KeywordSplitter

FILE_PATH = "https://www.gutenberg.org/cache/epub/16865/pg16865.txt"

# Step 1: Read Pinocchio
reader = VanillaReader()
reader_output = reader.read(file_path=FILE_PATH)

print(reader_output.model_dump_json(indent=4))  # inspect ReaderOutput

# Step 2: Split by chapter markers
chapter_pattern = r"CHAPTER\s+[IVXLCDM]+"
splitter = KeywordSplitter(
    patterns={"chapter": chapter_pattern},
    include_delimiters="before",
    flags=re.IGNORECASE,
    chunk_size=3000,
)
splitter_output = splitter.split(reader_output)

print(splitter_output.model_dump_json(indent=4))  # inspect SplitterOutput

# Step 3: Visualize chunks
for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + f" Chapter {idx + 1} " + "=" * 40)
    print(chunk[:800] + "\n")

# Step 4: Explore metadata
meta = splitter_output.metadata["keyword_matches"]
print("Counts:", meta["counts"])
print("First spans:", meta["spans"][:5])
print("Splitter parameters:", splitter_output.split_params)
```

    {
        "text": "﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook or online\r\nat www.gutenberg.org. If you are not located in the United States,\r\nyou will have to check the laws of the country
    ...
    resses. Donations are accepted in a number of other
    ways including checks, online payments and credit card donations. To
    donate, please visit: www.gutenberg.org/donate.
    
    Section 5. General Information About Project Gutenberg™ electronic works
    
    Professor 
    
    Counts: {'chapter': 36}
    First spans: [(7063, 7072), (10320, 10330), (14085, 14096), (20136, 20146), (23487, 23496)]
    Splitter parameters: {'include_delimiters': 'before', 'flags': re.IGNORECASE, 'chunk_size': 3000, 'pattern_names': ['chapter']}

