# **Example**: Splitting Files by pages using `PagedSplitter`

![Split by pages illustration](https://www.pdfgear.com/pdf-editor-reader/img/how-to-cut-pdf-pages-in-half-1.png)

For some documents, one of the best splitting strategies can be divide them by pages. To do so, you can use the `PagedSplitter`.

For this example, we will read the file using `VanillaReader. The file can be found on the [GitHub repository](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/attention.pdf), and it consists of a scientific paper (*Attention is All You Need*) with 15 pages. Let's see how to split it.

## Step 1. Read the file

You can read the file using `VanillaReader` or `DoclingReader`. In case that you use `MarkItDownReader`, you should pass the parameter `split_by_pages = True`, since MarkItDown by default does not provide any placeholder to split by pages.

??? example "Show Python examples for all Readers"

    ```python
    from splitter_mr.reader import VanillaReader

    FILE_PATH = "data/attention.pdf"

    reader = VanillaReader()
    reader_output = reader.read(file_path=FILE_PATH)
    ```

    ```python
    from splitter_mr.reader import DoclingReader

    FILE_PATH = "data/attention.pdf"

    reader = DoclingReader()
    reader_output = reader.read(file_path=FILE_PATH)
    ```

    ```python
    from splitter_mr.reader import MarkItDownReader

    FILE_PATH = "data/attention.pdf"

    reader = MarkItDownReader()
    reader_output = reader.read(file_path=FILE_PATH, split_by_pages=True)
    ```

The output will be the following:

```python
text='Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.\n\n## Attention Is All You Need\n\nAshish Vaswani ∗ Google Brain avaswani@google.com\n\nNoam Shazeer ∗ Google Brain noam@google.com\n\nNiki Parmar ∗ Google Research nikip@google.com\n\nJakob Uszkoreit ... We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.\n\n<!-- image -->' document_name='attention.pdf' document_path='data/attention.pdf' document_id='3d0057c0-da49-4d12-bb02-38b0f1116bc3' conversion_method='markdown' reader_method='docling' ocr_method=None page_placeholder='<!-- page -->' metadata={}
```

As you can see, the [`ReaderOutput` object](../../api_reference/reader.md#output-format) has an attribute named `page_placeholder` which allows to identify every page. 

## Split by pages

So, we can simply instantiate the `PageSplitter` object and use the `split` method to get the chunks page-by-page:

```python
splitter = PagedSplitter()
splitter_output = splitter.split(reader_output=reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("\n" + "*"*80 + f" Chunk {idx} " + "*"*80 + "\n")
    print(chunk)
```

```md
******************************************************************************* Chunk 0 ********************************************************************************

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

## Attention Is All You Need

Ashish Vaswani ∗ Google Brain avaswani@google.com

Noam Shazeer ∗ Google Brain noam@google.com

Niki Parmar ∗ Google Research nikip@google.com

Jakob Uszkoreit ∗ Google Research usz@google.com

Llion Jones ∗ Google Research llion@google.com

Aidan N. Gomez ∗ † University of Toronto aidan@cs.toronto.edu

Łukasz Kaiser Google Brain lukaszkaiser@google.com

∗

Illia Polosukhin ∗ ‡

illia.polosukhin@gmail.com

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

...

******************************************************************************** Chunk 13 ********************************************************************************

ewed in color.

<!-- image -->Input-Input Layer5

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word.

<!-- image -->

******************************************************************************** Chunk 14 ********************************************************************************

for this word.

<!-- image -->Input-Input Layer5

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.

```

Indeed, we have obtained a list of chunks with the extracted content, one per page.

### Experimenting with custom parameteres

In case that we want to split by group of many pages (e.g., 3), we can specify that value on the `PageSplitter` object. In addition, we can define an overlap between characters:

```python
splitter = PagedSplitter(chunk_size=3, chunk_overlap = 100)
splitter_output = splitter.split(reader_output=reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("\n" + "*"*80 + f" Chunk {idx} " + "*"*80 + "\n")
    print(chunk)
```

```md
******************************************************************************** Chunk 0 ********************************************************************************

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

## Attention Is All You Need

Ashish Vaswani ∗ Google Brain avaswani@google.com

Noam Shazeer ∗ Google Brain noam@google.com

Niki Parmar ∗ Google Research nikip@google.com

Jakob Uszkoreit ∗ Google Research usz@google.com

Llion Jones ∗ Google Research llion@google.com

Aidan N. Gomez ∗ † University of Toronto aidan@cs.toronto.edu

Łukasz Kaiser Google Brain lukaszkaiser@google.com

∗

Illia Polosukhin ∗ ‡

illia.polosukhin@gmail.com

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

...

******************************************************************************** Chunk 4 ********************************************************************************

apers) , pages 434-443. ACL, August 2013.                                                          |## Attention Visualizations Input-Input Layer5

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shown only for the word 'making'. Different colors represent different heads. Best viewed in color.

<!-- image -->
Input-Input Layer5

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word.

<!-- image -->
Input-Input Layer5

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.

<!-- image -->
```

And that's it! Try to experiment which values are the best option for your use case. Thank you for reading! :)

## Complete script

```python
from splitter_mr.reader import DoclingReader, VanillaReader
from splitter_mr.splitter import PagedSplitter

FILE_PATH = "data/attention.pdf"

reader = DoclingReader()
reader_output = reader.read(file_path=FILE_PATH)

print(reader_output)

splitter = PagedSplitter()
splitter_output = splitter.split(reader_output=reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("\n" + "*"*80 + f" Chunk {idx} " + "*"*80 + "\n")
    print(chunk)

splitter = PagedSplitter(chunk_size=3, chunk_overlap = 100)
splitter_output = splitter.split(reader_output=reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("\n" + "*"*80 + f" Chunk {idx} " + "*"*80 + "\n")
    print(chunk)
```
