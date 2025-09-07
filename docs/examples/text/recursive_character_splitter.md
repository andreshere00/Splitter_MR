# **Example**: Read a basic document and chunk it with a `RecursiveCharacterSplitter`

As an example, we will use the first chapter of the book "*El ingenioso hidalgo Don Quijote de La Mancha*". The text of reference can be extracted from the [GitHub project](https://github.com/andreshere00/Splitter_MR).

![El Quijote](https://www.cartv.es/thumbs/990x750r/2021-05/quijote-1-1-.jpg)

## Step 1: Read the text using a Reader component

We will use the [**`VanillaReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) class, since there is no need to transform the text into a `markdown` format. 

Firstly, we will create a new Python file and instantiate our class as follows:


```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
```


To read the file, we only need to call the `read` method from this class, which is inherited from the [`BaseReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#basereader) class (see [documentation](../../api_reference/reader.md)).


```python
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"
reader_output = reader.read(file_url=url)
```


The `reader_output` variable now contains a [**`ReaderOutput`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#output-format) object, with the following fields:


```python
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\nEn un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. 
    ...
    tural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.",
        "document_name": "quijote_example.txt",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt",
        "document_id": "b22470b2-e72a-4b25-95fe-f29d092cb9ec",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



The `ReaderOutput` object contains both the document text and useful metadata for ETL pipelines and LLM traceability. In case of using another Reader component, the output will be similar.

To get the text, simply access the `text` attribute:


```python
print(reader_output.text)
```

    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della conc
    ...
    quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.



## Step 2: Split the text using a splitting strategy

Before splitting, you have to choose a splitting strategy depending on your needs. 

In this case, we will use [**`RecursiveCharacterSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#recursivesplitter) since it is suitable for long, unstructured texts with an unknown number of words and stop words.

We will split the chunks to have, at maximum, 1000 characters (`chunk_size = 1000`) with a 10% of overlapping between chunks (`chunk_overlap = 0.1`). Overlapping defines the number or percentage of common words between consecutive chunks.

Instantiate the splitter:


```python
from splitter_mr.splitter import RecursiveCharacterSplitter

splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=0.1)
```


Apply the `split` method to the `reader_output`. This returns a `SplitterOutput` object with:


```python
splitter_output = splitter.split(reader_output)

print(splitter_output.model_dump_json(indent=4))
```

    {
        "chunks": [
            "Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha",
            "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres par
    ...
    72a-4b25-95fe-f29d092cb9ec",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "split_method": "recursive_character_splitter",
        "split_params": {
            "chunk_size": 1000,
            "chunk_overlap": 0.1,
            "separators": [
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "​",
                "，",
                "、",
                "．",
                "。",
                ""
            ]
        },
        "metadata": {}
    }



To visualize every chunk, we can simply perform the following operation:


```python
for idx, chunk in enumerate(splitter_output.chunks):
    print("=" * 40 + " Chunk " + str(idx + 1) + " " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    Capítulo Primero
    
    Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
    
    ======================================== Chunk 2 ========================================
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, sal
    ...
    uien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
    




!!! note

    Remember that in case that we want to use custom separators or define another `chunk_size` or overlapping, we can do it when instantiating the class. 

**And that's it!** This is as simple as shown here.

## Complete script

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import RecursiveCharacterSplitter

reader = VanillaReader()

url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"
reader_output = reader.read(file_url = url)

print(reader_output.model_dump_json(indent=4)) # Visualize the ReaderOutput object
print(reader_output.text) # Get the text from the document

splitter = RecursiveCharacterSplitter(
    chunk_size = 1000,
    chunk_overlap = 100)
splitter_output = splitter.split(reader_output)

print(splitter_output.model_dump_json(indent=4)) # Print the SplitterOutput object

for idx, chunk in enumerate(splitter_output.chunks):
    # Visualize every chunk
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")
```
