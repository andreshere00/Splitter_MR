# Read a basic document and chunk it with a Recursive Character Splitter

As an example, we will use the first chapter of the book "*El ingenioso hidalgo Don Quijote de La Mancha*". The text of reference can be extracted from the [GitHub project](https://github.com/andreshere00/Splitter_MR).

## Step 1: Read the text using a Reader component

We will use the `VanillaReader` class, since there is no need to transform the text into a `markdown` format. 

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

The `reader_output` variable now contains a `ReaderOutput` object, with the following fields:

```python
print(reader_output)
```

```python
ReaderOutput(text='Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha\n\nEn un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad.\n\nEs, pues, de saber, que este sobredicho hidalgo, los ratos que estaba ocioso (que eran los más del año) se daba a leer libros de caballerías con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura, para comprar libros de caballerías en que leer; y así llevó a su casa todos cuantos pudo haber dellos; y de todos ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva: porque la claridad de su prosa, y aquellas intrincadas razones suyas, le parecían de perlas; y más cuando llegaba a leer aquellos requiebros y cartas de desafío, donde en muchas partes hallaba escrito: la razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura, y también cuando leía: los altos cielos que de vuestra divinidad divinamente con las estrellas se fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza. Con estas y semejantes razones perdía el pobre caballero el juicio, y desvelábase por entenderlas, y desentrañarles el sentido, que no se lo sacara, ni las entendiera el mismo Aristóteles, si resucitara para sólo ello. No estaba muy bien con las heridas que don Belianis daba y recibía, porque se imaginaba que por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales; pero con todo alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma, y darle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran.\n\nTuvo muchas veces competencia con el cura de su lugar (que era hombre docto graduado en Sigüenza), sobre cuál había sido mejor caballero, Palmerín de Inglaterra o Amadís de Gaula; mas maese Nicolás, barbero del mismo pueblo, decía que ninguno llegaba al caballero del Febo, y que si alguno se le podía comparar, era don Galaor, hermano de Amadís de Gaula, porque tenía muy acomodada condición para todo; que no era caballero melindroso, ni tan llorón como su hermano, y que en lo de la valentía no le iba en zaga.\n\nEn resolución, él se enfrascó tanto en su lectura, que se le pasaban las noches leyendo de claro en claro, y los días de turbio en turbio, y así, del poco dormir y del mucho leer, se le secó el cerebro, de manera que vino a perder el juicio. Llenósele la fantasía de todo aquello que leía en los libros, así de encantamientos, como de pendencias, batallas, desafíos, heridas, requiebros, amores, tormentas y disparates imposibles, y asentósele de tal modo en la imaginación que era verdad toda aquella máquina de aquellas soñadas invenciones que leía, que para él no había otra historia más cierta en el mundo.\n\nDecía él, que el Cid Ruy Díaz había sido muy buen caballero; pero que no tenía que ver con el caballero de la ardiente espada, que de sólo un revés había partido por medio dos fieros y descomunales gigantes. Mejor estaba con Bernardo del Carpio, porque en Roncesvalle había muerto a Roldán el encantado, valiéndose de la industria de Hércules, cuando ahogó a Anteo, el hijo de la Tierra, entre los brazos. Decía mucho bien del gigante Morgante, porque con ser de aquella generación gigantesca, que todos son soberbios y descomedidos, él solo era afable y bien criado; pero sobre todos estaba bien con Reinaldos de Montalbán, y más cuando le veía salir de su castillo y robar cuantos topaba, y cuando en Allende robó aquel ídolo de Mahoma, que era todo de oro, según dice su historia. Diera él, por dar una mano de coces al traidor de Galalón, al ama que tenía y aun a su sobrina de añadidura.\n\nEn efecto, rematado ya su juicio, vino a dar en el más extraño pensamiento que jamás dio loco en el mundo, y fue que le pareció convenible y necesario, así para el aumento de su honra, como para el servicio de su república, hacerse caballero andante, e irse por todo el mundo con sus armas y caballo a buscar las aventuras, y a ejercitarse en todo aquello que él había leído, que los caballeros andantes se ejercitaban, deshaciendo todo género de agravio, y poniéndose en ocasiones y peligros, donde acabándolos, cobrase eterno nombre y fama.\n\nImaginábase el pobre ya coronado por el valor de su brazo por lo menos del imperio de Trapisonda: y así con estos tan agradables pensamientos, llevado del estraño gusto que en ellos sentía, se dió priesa a poner en efecto lo que deseaba. Y lo primero que hizo, fue limpiar unas armas, que habían sido de sus bisabuelos, que, tomadas de orín y llenas de moho, luengos siglos había que estaban puestas y olvidadas en un rincón. Limpiólas y aderezólas lo mejor que pudo; pero vió que tenían una gran falta, y era que no tenía celada de encaje, sino morrión simple; mas a esto suplió su industria, porque de cartones hizo un modo de media celada, que encajada con el morrión, hacía una apariencia de celada entera. Es verdad que para probar si era fuerte, y podía estar al riesgo de una cuchillada, sacó su espada, y le dió dos golpes, y con el primero y en un punto deshizo lo que había hecho en una semana: y no dejó de parecerle mal la facilidad con que la había hecho pedazos, y por asegurarse de este peligro, lo tornó a hacer de nuevo, poniéndole unas barras de hierro por de dentro de tal manera, que él quedó satisfecho de su fortaleza; y, sin querer hacer nueva experiencia de ella, la diputó y tuvo por celada finísima de encaje. Fue luego a ver a su rocín, y aunque tenía más cuartos que un real, y más tachas que el caballo de Gonela, que tantum pellis, et ossa fuit, le pareció que ni el Bucéfalo de Alejandro, ni Babieca el del Cid con él se igualaban. Cuatro días se le pasaron en imaginar qué nombre le podría: porque, según se decía él a sí mismo, no era razón que caballo de caballero tan famoso, y tan bueno él por sí, estuviese sin nombre conocido; y así procuraba acomodársele, de manera que declarase quien había sido, antes que fuese de caballero andante, y lo que era entones: pues estaba muy puesto en razón, que mudando su señor estado, mudase él también el nombre; y le cobrase famoso y de estruendo, como convenía a la nueva orden y al nuevo ejercicio que ya profesaba: y así después de muchos nombres que formó, borró y quitó, añadió, deshizo y tornó a hacer en su memoria e imaginación, al fin le vino a llamar ROCINANTE, nombre a su parecer alto, sonoro y significativo de lo que había sido cuando fue rocín, antes de lo que ahora era, que era antes y primero de todos los rocines del mundo. Puesto nombre y tan a su gusto a su caballo, quiso ponérsele a sí mismo, y en este pensamiento, duró otros ocho días, y al cabo se vino a llamar DON QUIJOTE, de donde como queda dicho, tomaron ocasión los autores de esta tan verdadera historia, que sin duda se debía llamar Quijada, y no Quesada como otros quisieron decir. Pero acordándose que el valeroso Amadís, no sólo se había contentado con llamarse Amadís a secas, sino que añadió el nombre de su reino y patria, por hacerla famosa, y se llamó Amadís de Gaula, así quiso, como buen caballero, añadir al suyo el nombre de la suya, y llamarse DON QUIJOTE DE LA MANCHA, con que a su parecer declaraba muy al vivo su linaje y patria, y la honraba con tomar el sobrenombre della.\n\nLimpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.', document_name='test_1.txt', document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt', document_id='b5fab5c9-a581-46c2-9f7e-ef56a2a14243', conversion_method='txt', reader_method='vanilla', ocr_method=None, metadata={})
```

The `ReaderOutput` object contains both the document text and useful metadata for ETL pipelines and LLM traceability. f you use another Reader component, the output will be similar.

To get the text, simply access the `text` attribute:

```python
print(reader_output.text)
```

```bash
Capítulo Primero

Que trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que deste caso escriben), aunque por conjeturas verosímiles se deja entender que se llama Quijana; pero esto importa poco a nuestro cuento; basta que en la narración dél no se salga un punto de la verdad.

Es, pues, de saber, que este sobredicho hidalgo, los ratos que estaba ocioso (que eran los más del año) se daba a leer libros de caballerías con tanta afición y gusto, que olvidó casi de todo punto el ejercicio de la caza, y aun la administración de su hacienda; y llegó a tanto su curiosidad y desatino en esto, que vendió muchas hanegas de tierra de sembradura, para comprar libros de caballerías en que leer; y así llevó a su casa todos cuantos pudo haber dellos; y de todos ningunos le parecían tan bien como los que compuso el famoso Feliciano de Silva: porque la claridad de su prosa, y aquellas intrincadas razones suyas, le parecían de perlas; y más cuando llegaba a leer aquellos requiebros y cartas de desafío, donde en muchas partes hallaba escrito: la razón de la sinrazón que a mi razón se hace, de tal manera mi razón enflaquece, que con razón me quejo de la vuestra fermosura, y también cuando leía: los altos cielos que de vuestra divinidad divinamente con las estrellas se fortifican, y os hacen merecedora del merecimiento que merece la vuestra grandeza. Con estas y semejantes razones perdía el pobre caballero el juicio, y desvelábase por entenderlas, y desentrañarles el sentido, que no se lo sacara, ni las entendiera el mismo Aristóteles, si resucitara para sólo ello. No estaba muy bien con las heridas que don Belianis daba y recibía, porque se imaginaba que por grandes maestros que le hubiesen curado, no dejaría de tener el rostro y todo el cuerpo lleno de cicatrices y señales; pero con todo alababa en su autor aquel acabar su libro con la promesa de aquella inacabable aventura, y muchas veces le vino deseo de tomar la pluma, y darle fin al pie de la letra como allí se promete; y sin duda alguna lo hiciera, y aun saliera con ello, si otros mayores y continuos pensamientos no se lo estorbaran.

...

Limpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.
```

## Step 2: Split the text using a splitting strategy

Before splitting, you have to choose a splitting strategy depending on your needs. 

In this case, we will use `RecursiveCharacterSplitter` since it is suitable for long, unstructured texts with an unknown number of words and stop words.

We will split the chunks to have, at maximum, 1000 characters (`chunk_size = 1000`) with a 10% of overlapping between chunks (`chunk_overlap = 0.1`). Overlapping defines the number or percentage of common words between consecutive chunks.

Instantiate the splitter:

```python
from splitter_mr.splitter import RecursiveCharacterSplitter

splitter = RecursiveCharacterSplitter(
    chunk_size = 1000,
    chunk_overlap = 0.1)
```

Apply the `split` method to the `reader_output`. This returns a `SplitterOutput` object with:

```python
splitter_output = splitter.split(reader_output)

print(splitter_output)
```
```bash
{'chunks': ['Capítulo Primero\n\nQue trata de la condición y ejercicio del famoso hidalgo D. Quijote de la Mancha', 'En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas con sus pantuflos de lo mismo, los días de entre semana se honraba con su vellori de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años, era de complexión recia, seco de carnes, enjuto de rostro; gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de Quijada o Quesada (que en esto hay alguna diferencia en los autores que', ..., 'Limpias, pues, sus armas, hecho del morrión celada, puesto nombre a su rocín, y confirmándose a sí mismo, se dió a entender que no le faltaba otra cosa, sino buscar una dama de quien enamorarse, porque el caballero andante sin amores, era árbol sin hojas y sin fruto, y cuerpo sin alma. Decíase él: si yo por malos de mis pecados, por por mi buena suerte, me encuentro por ahí con algún gigante, como de ordinario les acontece a los caballeros andantes, y le derribo de un encuentro, o le parto por mitad del cuerpo, o finalmente, le venzo y le rindo, ¿no será bien tener a quién enviarle presentado, y que entre y se hinque de rodillas ante mi dulce señora, y diga con voz humilde y rendida: yo señora, soy el gigante Caraculiambro, señor de la ínsula Malindrania, a quien venció en singular batalla el jamás como se debe alabado caballero D. Quijote de la Mancha, el cual me mandó que me presentase ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se', 'ante la vuestra merced, para que la vuestra grandeza disponga de mí a su talante? ¡Oh, cómo se holgó nuestro buen caballero, cuando hubo hecho este discurso, y más cuando halló a quién dar nombre de su dama! Y fue, a lo que se cree, que en un lugar cerca del suyo había una moza labradora de muy buen parecer, de quien él un tiempo anduvo enamorado, aunque según se entiende, ella jamás lo supo ni se dió cata de ello. Llamábase Aldonza Lorenzo, y a esta le pareció ser bien darle título de señora de sus pensamientos; y buscándole nombre que no desdijese mucho del suyo, y que tirase y se encaminase al de princesa y gran señora, vino a llamarla DULCINEA DEL TOBOSO, porque era natural del Toboso, nombre a su parecer músico y peregrino y significativo, como todos los demás que a él y a sus cosas había puesto.'], 'chunk_id': ['761769b5-7665-4c3a-b519-d3cd5080ba96', '320f03a1-6d38-4083-a10d-03cffdb749e7', 'c055ff94-9619-4a1e-94d9-c9e4268235ec', '7692bb32-dc1a-4c97-94d4-32ae885469ad', 'abb3a662-9d66-442d-8d46-fd299865d047', 'bb298ef1-fc24-4508-bc84-c9d3663085e2', '586a24ca-2bda-47c9-afde-24abbf052908', 'a59b6a58-c0b6-4241-b80d-aec249e2407b', '54bdc851-dad0-4ee3-969e-a12ea4382846', '40214451-e233-4339-80b7-616b2016f81e', 'ff4c3e71-8b09-439f-89ea-2894b7cad423', '8bf75213-4c4d-42f6-a350-3e01c5153ab3', 'fed9271e-515c-410e-a018-a6204fc3a8f1', 'c4a4219b-899a-4faa-af3e-0d46e6c9fcd4', '6df971c3-b533-4a12-a68b-5156e565503c'], 'document_name': 'test_1.txt', 'document_path': 'https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt', 'document_id': 'b5fab5c9-a581-46c2-9f7e-ef56a2a14243', 'conversion_method': 'txt', 'reader_method': None, 'ocr_method': None, 'split_method': 'recursive_character_splitter', 'split_params': {'chunk_size': 1000, 'chunk_overlap': 100, 'separators': ['\n\n', '\n', ' ', '.', ',', '\u200b', '，', '、', '．', '。', '']}, 'metadata': {}}
```

To visualize every chunk, we can simply perform the following operation:

```python
for idx, chunk in enumerate(splitter_output.chunks):
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

**And that's it!** This is as simple as shown here.

## Complete script

Here is the complete script for this example:

```python
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import RecursiveCharacterSplitter

reader = VanillaReader()

url = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/test_1.txt"
reader_output = reader.read(file_url = url)

print(reader_output) # Visualize the ReaderOutput object
print(reader_output.text) # Get the text from the document

splitter = RecursiveCharacterSplitter(
    chunk_size = 1000,
    chunk_overlap = 100)
splitter_output = splitter.split(reader_output)

print(splitter_output) # Print the SplitterOutput object

for idx, chunk in enumerate(splitter_output.chunks):
    # Visualize every chunk
    print("="*40 + " Chunk " + str(idx + 1) + " " + "="*40 + "\n" + chunk + "\n")
```

> 👨‍💻 **Work-in-progress...** More examples to come!