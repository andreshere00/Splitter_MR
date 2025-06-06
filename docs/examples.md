# Examples

## Read a basic document and chunk it with a Recursive Character Splitter

We will use as an example the first chapter of the book "El ingenioso hidalgo Don Quijote de La Mancha". The text of reference can be extracted from the [GitHub project](https://github.com/andreshere00/Splitter_MR).

### 1. Read the text using a Reader component.

We will use the `VanillaReader` class, since there is no need to transform the text into a markdown format. 

Firstly, we will create a new Python file and instantiate our class as follows:

```python
from splitter_mr.reader import VanillaReader

reader = VanillaReader()
```

To read the file, we only need to call the `read` method from this class, which is inherited from the `BaseReader` class (see [documentation](./reader.md)).

```python
url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt"
reader_output = reader.read(file_url = url)
```

The result is a `ReaderOutput` object, which has the following structure:

```python
print(reader_output)
```

```json
{
    "text": "Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\nEn un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad...",
    "document_name": "test_1.txt",
    "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt",
    "document_id": "9f57f57c-8a9d-4c02-9155-6eb44002ba0e",
    "conversion_method": "txt",
    "ocr_method": None,
    "metadata": {}
}
```

As we can see, we have obtained an object with not only the text extracted but with information that can be useful to some ETL (Extract, Transform and Load) processes & LLM traceability. In case that we use other Reader components, the output will be similar. 

To extract the text, we can simply print the text field as follows:

```python
print(reader_output['text'])
```

```bash
Capítulo Primero

Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad.

Es, pues, de saber, que este sobredicho hidalgo, los ratos que estaba ocioso (que eran los más del año) se daba a leer libros de caballerías con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura, para comprar libros de caballerías en que leer; y así llevó a su casa todos cuantos pudo haber dellos; y de todos ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva: porque la claridad de su prosa, y aquellas intrincadas razones suyas, le parecían de perlas; y más cuando llegaba a leer aquellos requiebros y cartas de desafío, donde en muchas partes hallaba escrito: la razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura, y también cuando leía: los altos cielos que de vuestra divinidad divinamente con las estrellas se fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza. Con estas y semejantes razones perdía el pobre caballero el juicio, y desvelábase por entenderlas, y desentrañarles el sentido, que no se lo sacara, ni las entendiera el mismo Aristóteles, si resucitara para sólo ello. No estaba muy bien con las heridas que don Belianis daba y recibía, porque se imaginaba que por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales; pero con todo alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma, y darle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran.

...

Limpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
```

### 2. Split the text using a splitting strategy

Prior to chunking, you have to choose a splitting strategy depending on your needs. 

In this case, we will use `RecursiveCharacterSplitter` since it is suitable for long, unstructured texts with an unknown number of words and stop words.

We will split the chunks to have, at maximum, 1000 characters (`chunk_size = 1000`) with a 10% of overlapping between chunks (`chunk_overlap = 0.1`). The overlapping can be defined as the number or percentage of common words between chunks. 

So, we instantiate the class:

```python
from splitter_mr.splitter import RecursiveCharacterSplitter

splitter = RecursiveCharacterSplitter(
    chunk_size = 1000,
    chunk_overlap = 0.1)
```

And we apply the `split` method with the reader_output, we get a `SplitterOutput` object with the following shape:

```python
splitter_output = splitter.split(reader_output)

print(splitter_output)
```
```bash
{'chunks': ['Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha', 'En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que', ..., 'Limpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se', 'ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.'], 'chunk_id': ['b685bd51-73c8-40dd-85dc-6ed7957597c1', 'a18dbc7e-9330-4ddb-affe-039872b49c9c', '6d3c7b53-806a-47ac-9b5d-67df4fc17994', '4f37ef5e-5419-4286-816c-d5c8f7135b29', 'eb576298-3358-4603-bb44-646d0be52416', 'e0bb74ff-3977-458e-95b9-97eb619e8004', '08995389-2b77-427d-9819-20ea31aafdf7', '6ddd81f3-db45-4fcd-a48b-f64953b030d5', '6a1b97b8-f321-4045-952a-a181e3d08929', 'cda247b8-323f-4704-8385-5d3233b8ed15', 'a2c627a7-cd46-4282-9115-5ab29625ce1d', 'de9809e7-ffe0-4218-a4e3-cca16ce758c2', '4e8d3efe-b701-4e7e-a88b-4679a131227a', 'ca8914e2-82cd-47c5-ae62-ef26d308885f', 'd160e16c-6257-4bd2-a10a-e55137d604fd'], 'document_name': 'test_1.txt', 'document_path': 'https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt', 'document_id': '9f57f57c-8a9d-4c02-9155-6eb44002ba0e', 'conversion_method': 'txt', 'ocr_method': None, 'split_method': 'recursive_character_splitter', 'split_params': {'chunk_size': 1000, 'chunk_overlap': 100, 'separators': ['\n\n', '\n', ' ', '.', ',', '\u200b', '，', '、', '．', '。', '']}, 'metadata': {}}
```

To visualize every chunk, we can simply perform the following operation:

```python
for idx, chunk in enumerate(splitter_output["chunks"]):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")
```

```bash
======================================== Chunk 1 ========================================
Capítulo Primero

Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha

======================================== Chunk 2 ========================================
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que

...

======================================== Chunk 14 ========================================
Limpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se

======================================== Chunk 15 ========================================
ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
```

> 💡 **NOTE:** Remember that in case that we want to use custom separators or define another `chunk_size` or overlapping, we can do it when instantiating the class. 

**And that's it!** This is as simple as it is shown in this tutorial.

### Complete script

The complete script for this example is shown below

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import RecursiveCharacterSplitter


reader = VanillaReader()

url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt"
reader_output = reader.read(file_url = url)

print(reader_output) # Visualize the ReaderOutput object
print(reader_output['text']) # Get the text from the document

splitter = RecursiveCharacterSplitter(
    chunk_size = 1000,
    chunk_overlap = 100)
splitter_output = splitter.split(reader_output)

print(splitter_output) # Print the SplitterOutput object

for idx, chunk in enumerate(splitter_output["chunks"]):
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n") # Visualize every chunk
```

> 👨‍💻 **Work-in-progress...** More examples to come!