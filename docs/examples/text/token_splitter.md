# **Example**: Split a Document by Tokens with `TokenSplitter` (SpaCy, NLTK, tiktoken)

In this example, we will use several popular NLP libraries to split a text document into token-based chunks. A **token** is the lexical unit in which a text is divided into. Tokenization can be performed in many ways: by words, by characters, by lemmas, etc. One of the most common methods is by sub-words. 

Observe the following example:

![Tokenization illustration](https://miro.medium.com/v2/resize:fit:1400/1*8QoeQNDcgwHjrS4AcX3V8g.png)

Most of Large Language Model uses tokenizers to process a large text into comprehensive lexical units . Hence, split by tokens could be a suitable option to produce chunks of a fixed length compatible with the LLM context window. So, in this tutorial we show how to split the text using three tokenizers: [**SpaCy**](https://spacy.io/api), [**NLTK**](https://www.nltk.org/), and [**tiktoken**](https://github.com/openai/tiktoken) (OpenAI tokenization). Let's see!

---

## Step 1: Read the Text Using a Reader

We will start by reading a text file using the [`MarkItDownReader`](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#markitdownreader). Remember that you can use any other compatible [Reader](https://andreshere00.github.io/Splitter_MR/api_reference/reader.md). Simply, instantiate a Reader object and use the `read` method. Provide as an argument the file to be read, which can be an URL, variable or path


```python
from splitter_mr.reader import MarkItDownReader

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt"
reader = MarkItDownReader()
reader_output = reader.read(file)
```


The output is a `ReaderOutput` object:


```python
print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "My Wonderful Family\nI live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!\n\nMy family is very important to me
    ...
    On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.",
        "document_name": "my_wonderful_family.txt",
        "document_path": "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt",
        "document_id": "5dc57f34-8bd2-4cfb-9442-7324a0b814ec",
        "conversion_method": "markdown",
        "reader_method": "markitdown",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }



To see only the document text, you can access to the `text` attribute of this object:


```python
print(reader_output.text)
```

    My Wonderful Family
    I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!
    
    My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.



---

## Step 2: Split the Document by Tokens

As we have said, the `TokenSplitter` lets you pick the tokenization backend: **SpaCy**, **NLTK**, or **tiktoken**. Use one or another depending on your needs. For every tokenizer, it should be passed:

- A `chunk_size`, the maximum chunk size in characters for the tokenization process. It tries to never cut a sentence in two chunks.
- A `model_name`, the tokenizer model to use. It should always follows this structure: `{tokenizer}/{model_name}`, e.g., `tiktoken/cl100k_base`. 

!!! Note
    For **spaCy** and **tiktoken**, the corresponding models must be installed in your environment.

To see a complete list of available tokenizers, refer to [Available models](#available-models).

### 2.1. Split by Tokens Using **SpaCy**

To split using a spaCy tokenizer model, you firstly need to instantiate the `TokenSplitter` class and select the parameters. Then, call to the `split` method with the path, URL or variable to split on:


```python
from splitter_mr.splitter import TokenSplitter

spacy_splitter = TokenSplitter(
    chunk_size=100,
    model_name="spacy/en_core_web_sm",  # Use the SpaCy model with "spacy/{model_name}" format
)
spacy_output = spacy_splitter.split(reader_output)

print(spacy_output)  # See the SplitterOutput object
```

    chunks=['My Wonderful Family\nI live in a house near the mountains.', 'I have two brothers and one sister, and I was born last.', 'My father teaches mathematics, and my mother is a nurse at a big hospital.', 'My brothers are very smart and work hard in school.', 'My sister is a nervous girl, but she is very kind.\n\nMy grandmother also lives with us.', 'She came from Italy when I was two years old.\n\nShe has grown old, but she is still very strong.', 'She cooks the best food!\n\n\n\nMy family i
    ...
     'a3b56da5-c885-43d5-8846-b0329f232af7', '6cf64509-b3f8-4a60-92f4-5006a4693e8a'] document_name='my_wonderful_family.txt' document_path='https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt' document_id='4798e9f9-4a0e-4453-8924-78748c6402d4' conversion_method='markdown' reader_method='markitdown' ocr_method=None split_method='token_splitter' split_params={'chunk_size': 100, 'model_name': 'spacy/en_core_web_sm', 'language': 'english'} metadata={}


    /Users/aherencia/Documents/Projects/Splitter_MR/.venv/lib/python3.12/site-packages/spacy/pipeline/lemmatizer.py:188: UserWarning: [W108] The rule-based lemmatizer did not find POS annotation for one or more tokens. Check that your pipeline includes components that assign token.pos, typically 'tagger'+'attribute_ruler' or 'morphologizer'.
      warnings.warn(Warnings.W108)



To see the resulting chunks, you can use the following code:


```python
# Visualize each chunk
for idx, chunk in enumerate(spacy_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    My Wonderful Family
    I live in a house near the mountains.
    
    ======================================== Chunk 2 ========================================
    I have two brothers and one sister, and I was born last.
    
    ======================================== Chunk 3 ========================================
    My father teaches mathematics, and my mother is a nurse at a big hospital.
    
    =====================================
    ...
    
    
    ======================================== Chunk 8 ========================================
    My brothers and I like to go on long walks in the mountains.
    
    ======================================== Chunk 9 ========================================
    My sister likes to cook with my grandmother.
    
    On the weekends we all play board games together.
    
    ======================================== Chunk 10 ========================================
    We laugh and always have a good time.
    
    I love my family very much.
    



### 2.2. Split by Tokens Using **NLTK**

Similarly, you can use a NLTK tokenizer. This library will always use `punkt` as the tokenizer, but you can customize the language through this parameter.


```python
nltk_splitter = TokenSplitter(
    chunk_size=100,
    model_name="nltk/punkt",  # Use the NLTK model as "nltk/{model_name}"
    language="english",  # Defaults to English
)
nltk_output = nltk_splitter.split(reader_output)

for idx, chunk in enumerate(nltk_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    My Wonderful Family
    I live in a house near the mountains.
    
    ======================================== Chunk 2 ========================================
    I have two brothers and one sister, and I was born last.
    
    ======================================== Chunk 3 ========================================
    My father teaches mathematics, and my mother is a nurse at a big hospital.
    
    =====================================
    ...
    
    
    ======================================== Chunk 8 ========================================
    My brothers and I like to go on long walks in the mountains.
    
    ======================================== Chunk 9 ========================================
    My sister likes to cook with my grandmother.
    
    On the weekends we all play board games together.
    
    ======================================== Chunk 10 ========================================
    We laugh and always have a good time.
    
    I love my family very much.
    



As you can see, the results are basically the same.

### 2.3. Split by Tokens Using **tiktoken** (OpenAI)

TikToken is one of the most extended tokenizer models. In this case, this tokenizer split by the number of tokens and chunks if `\\n\\n` is detected. Hence, the results are the following:


```python
tiktoken_splitter = TokenSplitter(
    chunk_size=100,
    model_name="tiktoken/cl100k_base",  # Use the tiktoken model as "tiktoken/{model_name}"
    language="english",
)
tiktoken_output = tiktoken_splitter.split(reader_output)

for idx, chunk in enumerate(tiktoken_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    ======================================== Chunk 1 ========================================
    My Wonderful Family
    
    ======================================== Chunk 2 ========================================
    I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives 
    ...
    She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!
    
    ======================================== Chunk 3 ========================================
    My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.
    



## **Extra:** Split by Tokens in Other Languages (e.g., Spanish)

In previous examples, we show you how to split the text by tokens, but these models were adapted to English. But in case that you have texts in other languages, you can use other Tokenizers. Here, there are two examples with SpaCy and NLTK (tiktoken is multilingual by default):


```python
from splitter_mr.reader import DoclingReader

sp_file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/mi_nueva_casa.txt"
sp_reader = DoclingReader()
sp_reader_output = sp_reader.read(sp_file)
print(sp_reader_output.text)
```

    Mi nueva casa
    Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra. Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo.
    
    Mi nueva casa está en una calle ancha que tiene muchos árboles. El piso de arriba de mi casa tiene tres dormitorios y un despacho para trabajar. El piso de abajo tiene una cocina muy grande, un comedor con una mesa y seis sillas, un salón con dos sofás verdes, una televisión y cortinas. Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano.
    
    Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión. Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería.
    
    


    /Users/aherencia/Documents/Projects/Splitter_MR/src/splitter_mr/reader/readers/docling_reader.py:87: UserWarning: Unsupported extension 'txt'. Using VanillaReader.
      warnings.warn(msg)



### Split Spanish by Tokens Using **SpaCy**


```python
spacy_sp_splitter = TokenSplitter(
    chunk_size=100,
    model_name="spacy/es_core_news_sm",  # Use a Spanish SpaCy model
)
spacy_sp_output = spacy_sp_splitter.split(sp_reader_output)

for idx, chunk in enumerate(spacy_sp_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    Created a chunk of size 107, which is longer than the specified 100
    Created a chunk of size 142, which is longer than the specified 100


    ======================================== Chunk 1 ========================================
    Mi nueva casa
    Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra.
    
    ======================================== Chunk 2 ========================================
    Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo.
    
    ======================================== Chunk 3 ========================================
    Mi nueva casa está en una calle ancha
    ...
    ==========================
    Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano.
    
    ======================================== Chunk 7 ========================================
    Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión.
    
    ======================================== Chunk 8 ========================================
    Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería.
    



### Split Spanish by Tokens Using **NLTK**


```python
nltk_sp_splitter = TokenSplitter(
    chunk_size=100, model_name="nltk/punkt", language="spanish"
)
nltk_sp_output = nltk_sp_splitter.split(sp_reader_output)

for idx, chunk in enumerate(nltk_sp_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    Created a chunk of size 107, which is longer than the specified 100
    Created a chunk of size 142, which is longer than the specified 100


    ======================================== Chunk 1 ========================================
    Mi nueva casa
    Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra.
    
    ======================================== Chunk 2 ========================================
    Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo.
    
    ======================================== Chunk 3 ========================================
    Mi nueva casa está en una calle ancha
    ...
    ==========================
    Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano.
    
    ======================================== Chunk 7 ========================================
    Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión.
    
    ======================================== Chunk 8 ========================================
    Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería.
    



---

**And that’s it!**
You can now tokenize and chunk text with precision, using the NLP backend and language that best fits your project.

!!! note
    For best results, make sure to install any SpaCy/NLTK/tiktoken models needed for your language and task.

## **Complete Script**


```python
from splitter_mr.reader import DoclingReader, MarkItDownReader
from splitter_mr.splitter import TokenSplitter

# 1. Read the file using any Reader (e.g., MarkItDownReader)

file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/my_wonderful_family.txt"

reader = MarkItDownReader()
reader_output = reader.read(file)
print(reader_output.text)

# 2. Split by Tokens

## 2.1. Using SpaCy

print("*" * 40 + " spaCy " + "*" * 40)

spacy_splitter = TokenSplitter(
    chunk_size=100,
    model_name="spacy/en_core_web_sm",  # Select a valid model with nomenclature spacy/{model_name}.
)
# Note that it is required to have the model installed in your execution machine.

spacy_output = spacy_splitter.split(reader_output)  # Split the text
print(spacy_output.model_dump_json(indent=4))  # Print the SplitterOutput object

# Visualize each chunk
for idx, chunk in enumerate(spacy_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")

## 2.2. Using NLTK

print("*" * 40 + " NLTK " + "*" * 40)

nltk_splitter = TokenSplitter(
    chunk_size=100,
    model_name="nltk/punkt",  # introduce the model as nltk/{model_name}
    language="english",  # defaults to this language
)

nltk_output = nltk_splitter.split(reader_output)

# Visualize each chunk
for idx, chunk in enumerate(nltk_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")

## 2.3. Using tiktoken

print("*" * 40 + " Tiktoken " + "*" * 40)

tiktoken_splitter = TokenSplitter(
    chunk_size=100,
    model_name="tiktoken/cl100k_base",  # introduce the model as tiktoken/{model_name}
    language="english",
)

tiktoken_output = tiktoken_splitter.split(reader_output)

# Visualize each chunk
for idx, chunk in enumerate(tiktoken_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")

## 2.4. Split by tokens in other languages (e.g., Spanish)

sp_file = "https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/mi_nueva_casa.txt"

sp_reader = DoclingReader()
sp_reader_output = sp_reader.read(sp_file)
print(sp_reader_output.text)  # Visualize the text content

### 2.4.1. Using SpaCy

print("*" * 40 + " Spacy in Spanish " + "*" * 40)

spacy_sp_splitter = TokenSplitter(
    chunk_size=100,
    model_name="spacy/es_core_news_sm",  # Pick another model in Spanish
)
nltk_sp_output = spacy_sp_splitter.split(sp_reader_output)

for idx, chunk in enumerate(nltk_sp_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")

### 2.4.2 Using NLTK

print("*" * 40 + " NLTK in Spanish " + "*" * 40)

nltk_sp_splitter = TokenSplitter(
    chunk_size=100,
    model_name="nltk/punkt",
    language="spanish",  # select `spanish` as language for the tokenizer
)
nltk_sp_output = nltk_sp_splitter.split(sp_reader_output)

for idx, chunk in enumerate(nltk_sp_output.chunks):
    print("=" * 40 + f" Chunk {idx + 1} " + "=" * 40 + "\n" + chunk + "\n")
```

    My Wonderful Family
    I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!
    
    My family is very important to me. We do lots of things together. My brothers and I like to go on long walks in the mountains. My sister likes to cook with my grandmother. On the weekends we all play board games together. We laugh and always have a good time. I love my family very much.
    **************************************** spaCy ****************************************


    /Users/aherencia/Documents/Projects/Splitter_MR/.venv/lib/python3.12/site-packages/spacy/pipeline/lemmatizer.py:188: UserWarning: [W108] The rule-based lemmatizer did not find POS annotation for one or more tokens. Check that your pipeline includes components that assign token.pos, typically 'tagger'+'attribute_ruler' or 'morphologizer'.
      warnings.warn(Warnings.W108)


    chunks=['My Wonderful Family\nI live in a house near the mountains.', 'I have two brothers and one sister, and I was born last.', 'My father teaches mathematics, and my mother is a nurse at a big hospital.', 'My brothers are very smart and work hard in school.', 'My sister is a nervous girl, but she is very kind.\n\nMy grandmother also lives with us.', 'She came from Italy when I was two years old.\n\nShe has grown old, but she is still very strong.', 'She cooks the best food!\n\n\n\nMy family i
    ...
    uy grande, un comedor con una mesa y seis sillas, un salón con dos sofás verdes, una televisión y cortinas. Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano.
    
    Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión. Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería.
    
    
    **************************************** Spacy in Spanish ****************************************


    Created a chunk of size 107, which is longer than the specified 100
    Created a chunk of size 142, which is longer than the specified 100
    Created a chunk of size 107, which is longer than the specified 100
    Created a chunk of size 142, which is longer than the specified 100


    ======================================== Chunk 1 ========================================
    Mi nueva casa
    Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra.
    
    ======================================== Chunk 2 ========================================
    Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo.
    
    ======================================== Chunk 3 ========================================
    Mi nueva casa está en una calle ancha
    ...
    ==========================
    Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano.
    
    ======================================== Chunk 7 ========================================
    Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión.
    
    ======================================== Chunk 8 ========================================
    Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería.
    



## Available models

There are several tokenizer models that you can use to split your text. In the following table is provided a summary of the models that you can currently use, among with some implementation examples:

| **Library**      | **Model identifier/template**                                                                                                      | **How to implement**                      | **Reference Guide**                                            |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- | :------------------------------------------------------------- |
| **NLTK (Punkt)** | `<language>`                                                                                                                       | See [NLTK Example](#nltk-example)         | [NLTK Tokenizers](https://www.nltk.org/api/nltk.tokenize.html) |
| **Tiktoken**     | `<encoder>`                                                                                                                        | See [Tiktoken Example](#tiktoken-example) | [tiktoken](https://github.com/openai/tiktoken)                 |
| **spaCy**        | `{CC}_core_web_sm`,<br>`{CC}_core_web_md`,<br>`{CC}_core_web_lg`,<br>`{CCe}_core_web_trf`,<br>`xx_ent_wiki_sm`,<br>`xx_sent_ud_sm` | See [spaCy Example](#spacy-example)       | [spaCy Models](https://spacy.io/usage/models)                  |

**spaCy Model Suffixes:**
- `sm` (**small**): Fastest, small in size, less accurate; good for prototyping and lightweight use-cases.
- `md` (**medium**): Medium size and accuracy; balances speed and performance.
- `lg` (**large**): Largest and most accurate pipeline with the most vectors; slower and uses more memory.
- `trf` (**transformer**): Uses transformer-based architectures (e.g., BERT, RoBERTa); highest accuracy, slowest, and requires more resources.

### NLTK Example


```python
language = "english"
TokenSplitter(model_name="nltk/punkt", language=language)
```




    <splitter_mr.splitter.splitters.token_splitter.TokenSplitter at 0x3106e42c0>




### Tiktoken Example


```python
encoder = "cl100k_base"
TokenSplitter(model_name=f"tiktoken/{encoder}")
```




    <splitter_mr.splitter.splitters.token_splitter.TokenSplitter at 0x315f1d7c0>




### spaCy Example


```python
CC = "en"
ext = "sm"
encoder = f"{CC}_core_web_{ext}"
TokenSplitter(model_name=f"spacy/{encoder}")
```




    <splitter_mr.splitter.splitters.token_splitter.TokenSplitter at 0x3280d74a0>


