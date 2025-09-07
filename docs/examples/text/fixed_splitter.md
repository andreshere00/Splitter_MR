# **Example**: Splitting Text Files with `CharacterSplitter`, `WordSplitter`, `SentenceSplitter`, `ParagraphSplitter`

When processing a plain text file, such as an e-book or an instruction guidebook, for downstream tasks like LLM ingestion, annotation, or search, it is often necessary to divide it into smaller, manageable chunks.

**SplitterMR provides the functionality to segment such files into groups of characters, words, sentences, or paragraphs**. Furthermore, it allows for overlapping chunks to maintain contextual continuity. This example will illustrate the application of each splitter, utilizing the first chapter of "*El Famoso Hidalgo Don Quijote de la Mancha*" (original language) as the sample text.

![El Quijote](https://www.cartv.es/thumbs/990x750r/2021-05/quijote-1-1-.jpg)

## Step 1: Read the Text Document

We will use [**Vanilla Reader**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) to load our text file. The result will be [`ReaderOutput`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#output-format), a Pydantic object with the following fields:


```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
data = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"  # Path to your file
reader_output = reader.read(data)
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\nEn un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. 
    ...
    tural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.",
        "document_name": "quijote_example.txt",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt",
        "document_id": "955e6c5b-3779-4328-a1aa-c3e7a60a4cfd",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



If you want to extract the text, you can access the content via the `text` attribute:


```python
print(reader_output.text)
```

    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della conc
    ...
    quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.



---

## Step 2: Split the Document

We will try four different splitting strategies: by **characters**, **words**, **sentences**, and **paragraphs**. Remember that you can adjust the chunk size as needed.


```python
from splitter_mr.splitter import (
    CharacterSplitter,
    WordSplitter,
    SentenceSplitter,
    ParagraphSplitter,
)
```


---

### 2.1. Split by **Characters**

Firstly, we will test the character-based splitting strategy. To do this, you can instantiate the [**`CharacterSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#charactersplitter) class with the splitting attributes as your choice and pass the reader's output to the split method of this class. Accessing the [**`SplitterOutput`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#splitter_mr.schema.models.SplitterOutput) object's content is then straightforward:


```python
char_splitter = CharacterSplitter(chunk_size=100)
char_splitter_output = char_splitter.split(reader_output)

print(char_splitter_output.model_dump_json(indent=4))
```

    {
        "chunks": [
            "Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\n",
            "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalg",
            "o de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más v",
            "aca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, alg",
            "ún palomino de añadid
    ...
    3f-5343-4cb3-8df5-cb6aa27aaf91"
        ],
        "document_name": "quijote_example.txt",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt",
        "document_id": "955e6c5b-3779-4328-a1aa-c3e7a60a4cfd",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "split_method": "character_splitter",
        "split_params": {
            "chunk_size": 100,
            "chunk_overlap": 0
        },
        "metadata": {}
    }



To visualize each chunk, you can use the following instruction:


```python
for idx, chunk in enumerate(char_splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    
    
    ======================================== Chunk 2 ========================================
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalg
    
    ======================================== Chunk 3 ========================================
    o de los de lanza en astil
    ...
    unk 103 ========================================
    jese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULC
    
    ======================================== Chunk 104 ========================================
    INEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significati
    
    ======================================== Chunk 105 ========================================
    vo, como todos los demás que a él y a sus cosas había puesto.
    



As you can see, the final characters of "hidalgo" are cut by this method. So how can we avoid cutting words? Introducing `WordSplitter`.

---

### 2.2. Split by **Words**

To use the [**`WordSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#wordsplitter), instantiate the class with your desired parameters (you can consult the [Developer guide](../../api_reference/splitter.md) for information on available parameters). Then, split the content using the previous output from the Reader. To visualize the chunks, you need to access the `chunks` attribute in the [`SplitterOutput`](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#splitter_mr.schema.models.SplitterOutput) object:


```python
word_splitter = WordSplitter(chunk_size=20)
word_splitter_output = word_splitter.split(reader_output)

for idx, chunk in enumerate(word_splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha En un lugar
    
    ======================================== Chunk 2 ========================================
    de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de
    
    ======================================== Chunk 3 ========================================
    lanza en astillero
    ...
    de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase
    
    ======================================== Chunk 94 ========================================
    al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su
    
    ======================================== Chunk 95 ========================================
    parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
    



Note that even though words aren't cut, the context isn't adequate because sentences are left incomplete. To avoid this issue, we should split by sentences. Introducing the [**`SentenceSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#sentencesplitter):

---

### 2.3. Split by **Sentences**

Analogously to the previous steps, we can define the [**`SentenceSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#sentencesplitter) object with the number of sentences to split on:


```python
sentence_splitter = SentenceSplitter(chunk_size=5)
sentence_splitter_output = sentence_splitter.split(reader_output)

for idx, chunk in enumerate(sentence_splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomin
    ...
    uien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
    



While the entire context is preserved when splitting by sentences, the varying chunk sizes suggest that chunking by paragraphs might be more beneficial. Introducing [**`ParagraphSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#paragraphsplitter).

---

### 2.4. Split by **Paragraphs**

We can select `3` as the desired number of paragraphs per chunk. The resulting chunks are the following:


```python
paragraph_splitter = ParagraphSplitter(chunk_size=3)
paragraph_splitter_output = paragraph_splitter.split(reader_output)

for idx, chunk in enumerate(paragraph_splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino 
    ...
    uien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
    



---

### 2.5. Add **Overlapping Chunks**

Another strategy you can employ is to preserve some text between chunks. For this use case, you can optionally add *overlap* between chunks. Overlap can be defined as either a fraction (e.g., `chunk_overlap = 0.2` for 20% overlap) or an integer number (e.g., `chunk_overlap = 20`):


```python
char_splitter_with_overlap = CharacterSplitter(chunk_size=100, chunk_overlap=0.2)
char_splitter_output = char_splitter_with_overlap.split(reader_output)

for idx, chunk in enumerate(char_splitter_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    
    
    ======================================== Chunk 2 ========================================
    ijote de la Mancha
    
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo
    
    ======================================== Chunk 3 ========================================
    , no ha mucho tiempo que v
    ...
    unk 129 ========================================
    ncaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del 
    
    ======================================== Chunk 130 ========================================
    que era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los d
    
    ======================================== Chunk 131 ========================================
    vo, como todos los demás que a él y a sus cosas había puesto.
    



---

**And that’s it!** With these splitters, you can flexibly chunk your text data however you need. Remember that you can visit the complete [**Developer Reference**](../../api_reference/splitter.md) to have more information about specific examples, methods, attributes and more of these Splitter classes.

## Complete Example Script

Finally, we provide a full example script for reproducibility purposes:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import (CharacterSplitter, ParagraphSplitter,
                                  SentenceSplitter, WordSplitter)

reader = VanillaReader()

data = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"
reader_output = reader.read(data)

print(reader_output.model_dump_json(indent=4)) # Visualize the ReaderOutput object
print(reader_output.text) # Get the text from the document

# 1. Split by Characters

char_splitter = CharacterSplitter(chunk_size=100)
char_splitter_output = char_splitter.split(reader_output)
print(char_splitter_output) # Visualize Character Splitter output

for idx, chunk in enumerate(char_splitter_output.chunks): # Visualize chunks
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")

# 2. Split by Words

word_splitter = WordSplitter(chunk_size=20)
word_splitter_output = word_splitter.split(reader_output)

for idx, chunk in enumerate(word_splitter_output.chunks):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")

# 3. Split by Sentences

sentence_splitter = SentenceSplitter(chunk_size=5)
sentence_splitter_output = sentence_splitter.split(reader_output)

for idx, chunk in enumerate(sentence_splitter_output.chunks):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")

# 4. Split by Paragraphs

paragraph_splitter = ParagraphSplitter(chunk_size=3)
paragraph_splitter_output = paragraph_splitter.split(reader_output)

for idx, chunk in enumerate(paragraph_splitter_output.chunks):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")

# 5. Add overlapping words between chunks

char_splitter_with_overlap = CharacterSplitter(chunk_size=100, chunk_overlap=0.2)
char_splitter_output = char_splitter_with_overlap.split(reader_output)

for idx, chunk in enumerate(char_splitter_output.chunks):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")
```
