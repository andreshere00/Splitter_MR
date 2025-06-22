# **Example**: Splitting Text Files with `CharacterSplitter`, `WordSplitter`, `SentenceSplitter`, `ParagraphSplitter`

When processing a plain text file, such as an e-book or an instruction guidebook, for downstream tasks like LLM ingestion, annotation, or search, it is often necessary to divide it into smaller, manageable chunks.

**SplitterMR provides the functionality to segment such files into groups of characters, words, sentences, or paragraphs**. Furthermore, it allows for overlapping chunks to maintain contextual continuity. This example will illustrate the application of each splitter, utilizing the first chapter of "*El Famoso Hidalgo Don Quijote de la Mancha*" (original language) as the sample text.

![El Quijote](https://www.cartv.es/thumbs/990x750r/2021-05/quijote-1-1-.jpg)

---

## Step 1: Read the Text Document

We will use the `VanillaReader` to load our text file.

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
data = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"  # Path to your file
reader_output = reader.read(data)
```

The `reader_output` is an object containing the raw text and its metadata. You can view its content by simply printing the object:

```python
print(reader_output)
```

```python
ReaderOutput(
    text='Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\nEn un lugar de la Mancha, ...',
    document_name='quijote_example.txt',
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt',
    document_id='34e37be6-743f-4807-80c3-216c76c7798b',
    conversion_method='txt',
    reader_method='vanilla',
    ocr_method=None,
    metadata={}
)
```

If you want to extract the text, you can access the content via the `text` attribute:

```python
print(reader_output.text)
```

---

## Step 2: Split the Document

We will try four different splitting strategies: by **characters**, **words**, **sentences**, and **paragraphs**. Remember that you can adjust the chunk size as needed.

```python
from splitter_mr.splitter import (
    CharacterSplitter,
    WordSplitter,
    SentenceSplitter,
    ParagraphSplitter
)
```

---

### 2.1. Split by **Characters**

Firstly, we will test the character-based splitting strategy. To do this, you can instantiate the `CharacterSplitter` class with the splitting attributes as your choice and pass the reader's output to the split method of this class. Accessing the `SplitterOutput` object's content is then straightforward:

```python
char_splitter = CharacterSplitter(chunk_size=100)
char_splitter_output = char_splitter.split(reader_output)

print(char_splitter_output)  # Visualize Character Splitter output
```

```python
SplitterOutput(
    chunks=['Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\n', 'En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalg', 'o de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más v', 'aca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, alg', 'ún palomino de añadidura los domingos, consumían las tres partes de su hacienda.\n\n...\n\nLimpias, pues', ', sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió ', 'a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballer', 'o andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malo', 's de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario', ' les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuer', 'po, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre ', 'y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gi', 'gante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás com', 'o se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vues', 'tra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro bue', 'n caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fu', 'e, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de', ' quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de', ' ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensami', 'entos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princ', 'esa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su p', 'arecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.'], 
    chunk_id=[
        '72533be1-07f5-47ff-b9bc-275fa14d38c7', 'd8b13e9c-29bf-4f0e-8d73-89e024d93218', '08d4af1b-acae-43ff-8798-70f70a71dbff',
        ..., 
        'c0382a84-693e-4c2f-bb82-2eef1166c0e7', '96a42f50-33bd-4a78-bd99-73542bbd763d', 'e3a0a647-baf9-442c-8737-b0abf0ddf87e', 'ee2b7209-1e1b-4d33-b420-3949385013ad'
        ], 
    document_name='quijote_example.txt', 
    document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt', document_id='34e37be6-743f-4807-80c3-216c76c7798b', conversion_method='txt', 
    reader_method='vanilla', 
    ocr_method=None, 
    split_method='character_splitter', 
    split_params={'chunk_size': 100, 'chunk_overlap': 0}, 
    metadata={}
)
```

To visualize each chunk, you can use the following instruction:

```python
for idx, chunk in enumerate(char_splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

**Sample output:**

```
======================================== Chunk 1 ========================================
Capítulo Primero

Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha

======================================== Chunk 2 ========================================
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalg

...
```

As you can see, the final characters of "hidalgo" are cut by this method. So how can we avoid cutting words? Introducing `WordSplitter`.

---

### 2.2. Split by **Words**

To use the `WordSplitter`, instantiate the class with your desired parameters (you can consult the [Developer guide](../../api_reference/splitter.md) for information on available parameters). Then, split the content using the previous output from the Reader. To visualize the chunks, you need to access the `chunks` attribute of the `SplitterOutput` object:

```python
word_splitter = WordSplitter(chunk_size=20)
word_splitter_output = word_splitter.split(reader_output)

for idx, chunk in enumerate(word_splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

```python
======================================== Chunk 1 ========================================
Capítulo Primero Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha En un lugar

======================================== Chunk 2 ========================================
de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de
...
```

Note that even though words aren't cut, the context isn't adequate because sentences are left incomplete. To avoid this issue, we should split by sentences. Introducing the `SentenceSplitter`:

---

### 2.3. Split by **Sentences**

Analogously to the previous steps, we can define the `SentenceSplitter` object with the number of sentences to split on:

```python
sentence_splitter = SentenceSplitter(chunk_size=5)
sentence_splitter_output = sentence_splitter.split(reader_output)

for idx, chunk in enumerate(sentence_splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

```python
======================================== Chunk 1 ========================================
Capítulo Primero

Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha

En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera.

======================================== Chunk 2 ========================================
Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad. Es, pues, de saber, que este sobredicho hidalgo, los ratos que estaba ocioso (que eran los más del año) se daba a leer libros de caballerías con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura, para comprar libros de caballerías en que leer; y así llevó a su casa todos cuantos pudo haber dellos; y de todos ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva: porque la claridad de su prosa, y aquellas intrincadas razones suyas, le parecían de perlas; y más cuando llegaba a leer aquellos requiebros y cartas de desafío, donde en muchas partes hallaba escrito: la razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura, y también cuando leía: los altos cielos que de vuestra divinidad divinamente con las estrellas se fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza. Con estas y semejantes razones perdía el pobre caballero el juicio, y desvelábase por entenderlas, y desentrañarles el sentido, que no se lo sacara, ni las entendiera el mismo Aristóteles, si resucitara para sólo ello. No estaba muy bien con las heridas que don Belianis daba y recibía, porque se imaginaba que por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales; pero con todo alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma, y darle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran.
```

While the entire context is preserved when splitting by sentences, the varying chunk sizes suggest that chunking by paragraphs might be more beneficial. Introducing `ParagraphSplitter`.

---

### 2.4. Split by **Paragraphs**

We can select 3 as the desired number of paragraphs per chunk. The resulting chunks are the following:

```python
paragraph_splitter = ParagraphSplitter(chunk_size=3)
paragraph_splitter_output = paragraph_splitter.split(reader_output)

for idx, chunk in enumerate(paragraph_splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

```python
======================================== Chunk 1 ========================================
Capítulo Primero
Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad.

======================================== Chunk 2 ========================================
Es, pues, de saber, que este sobredicho hidalgo, los ratos que estaba ocioso (que eran los más del año) se daba a leer libros de caballerías con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura, para comprar libros de caballerías en que leer; y así llevó a su casa todos cuantos pudo haber dellos; y de todos ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva: porque la claridad de su prosa, y aquellas intrincadas razones suyas, le parecían de perlas; y más cuando llegaba a leer aquellos requiebros y cartas de desafío, donde en muchas partes hallaba escrito: la razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura, y también cuando leía: los altos cielos que de vuestra divinidad divinamente con las estrellas se fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza. Con estas y semejantes razones perdía el pobre caballero el juicio, y desvelábase por entenderlas, y desentrañarles el sentido, que no se lo sacara, ni las entendiera el mismo Aristóteles, si resucitara para sólo ello. No estaba muy bien con las heridas que don Belianis daba y recibía, porque se imaginaba que por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales; pero con todo alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma, y darle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran.
Tuvo muchas veces competencia con el cura de su lugar (que era hombre docto graduado en Sigüenza), sobre cuál había sido mejor caballero, Palmerín de Inglaterra o Amadís de Gaula; mas maese Nicolás, barbero del mismo pueblo, decía que ninguno llegaba al caballero del Febo, y que si alguno se le podía comparar, era don Galaor, hermano de Amadís de Gaula, porque tenía muy acomodada condición para todo; que no era caballero melindroso, ni tan llorón como su hermano, y que en lo de la valentía no le iba en zaga.
En resolución, él se enfrascó tanto en su lectura, que se le pasaban las noches leyendo de claro en claro, y los días de turbio en turbio, y así, del poco dormir y del mucho leer, se le secó el cerebro, de manera que vino a perder el juicio. Llenósele la fantasía de todo aquello que leía en los libros, así de encantamientos, como de pendencias, batallas, desafíos, heridas, requiebros, amores, tormentas y disparates imposibles, y asentósele de tal modo en la imaginación que era verdad toda aquella máquina de aquellas soñadas invenciones que leía, que para él no había otra historia más cierta en el mundo.
```

---

### 2.5. Add **Overlapping Chunks**

Another strategy you can employ is to preserve some text between chunks. For this use case, you can optionally add *overlap* between chunks. Overlap can be defined as either a fraction (e.g., `chunk_overlap = 0.2` for 20% overlap) or an integer number (e.g., `chunk_overlap = 20`):

```python
char_splitter_with_overlap = CharacterSplitter(chunk_size=100, chunk_overlap=0.2)
char_splitter_output = char_splitter_with_overlap.split(reader_output)

for idx, chunk in enumerate(char_splitter_output.chunks):
    print("="*40 + f" Chunk {idx + 1} " + "="*40 + "\n" + chunk + "\n")
```

---

**And that’s it!** With these splitters, you can flexibly chunk your text data however you need. Remember that you can visit the complete [Developer Reference](../../api_reference/splitter.md) to have more information about specific examples, methods, attributes and more of these Splitter classes.

## Complete Example Script

Finally, we provide a full example script for reproducibility purposes:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import (CharacterSplitter, ParagraphSplitter,
                                  SentenceSplitter, WordSplitter)

reader = VanillaReader()

data = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/quijote_example.txt"
reader_output = reader.read(data)

print(reader_output) # Visualize the ReaderOutput object
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