# **Example:** How to create a simple RAG system using SplitterMR

In this example, we will see a use case for creating a GenAI application using **SplitterMR**. Concretely, we will build a RAG system to process Pinocchio's Tale and ask some questions about this book. Let's see how to do it!

![An illustration of Pinocchio Disney's version ](https://www.researchgate.net/publication/336265669/figure/fig2/AS:810368322519040@1570218272217/Pinocchio-cartoon-character-Credit-Walt-Disney.jpg)

## Requirements

1. A **Vector** **Database** => In this case, we will use [Qdrant](), but other options are available.
2. A **generative** **Model** => In this case, we will use [Azure OpenAI](https://azure.microsoft.com/es-es/products/ai-foundry/models/openai/), but other models can be used.
3. An **encoder** **Model** => It should encode text with the same tokenizer as the generative model. So, we will use [Azure OpenAI](ttps://azure.microsoft.com/es-es/products/ai-foundry/models/openai/).
4. The [**SplitterMR**](https://andreshere00.github.io/Splitter_MR/) library (obviously).
5. [**Docker**](https://www.docker.com/) installed. 
6. A set of **data**. In this case, the dataset is provided via the [**Gutenberg project**](https://www.gutenberg.org/ebooks/16865).

## First step. Prepare environment

Firstly, we will import the libraries that we need:


```python
import os
from dotenv import load_dotenv, find_dotenv
from itertools import batched
import re

from qdrant_client import QdrantClient, models
from openai import AzureOpenAI

from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import KeywordSplitter
```

Then, it is necessary to save as constants the environment variables that we will use:


```python
load_dotenv(dotenv_path=find_dotenv())

# ---- Large Language model connection details ---- #

# => Generative model

AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# => Embedding model

AZURE_OPENAI_EMBEDDING: str = os.getenv("AZURE_OPENAI_EMBEDDING")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
)

# ---- Vector Database connection ---- #

QDRANT_URL: str = os.getenv("QDRANT_URL")
QDRANT_API_KEY: str = os.getenv(
    "QDRANT_API_KEY"
)  # No needed, but in case that you want to connect to a production DB
COLLECTION_NAME: str = "pinocchio_demo_chunks"

# ---- Dataset ---- #

FILE_PATH: str = "https://www.gutenberg.org/cache/epub/16865/pg16865.txt"  # URL to Pinocchio's tale raw text
```

## Second step: Process the file

You can use any available Reader that you want. In this case, we will use [**`VanillaReader`**](https://andreshere00.github.io/Splitter_MR/api_reference/reader/#vanillareader) since the text does not need any transformations.


```python
reader = VanillaReader()
reader_output = reader.read(file_path=FILE_PATH)

print(reader_output.model_dump_json(indent=4))
```

    {
        "text": "﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook or online\r\nat www.gutenberg.org. If you are not located in the United States,\r\nyou will have to check the laws of the country
    ...
    tions to the Project Gutenberg Literary\r\nArchive Foundation, how to help produce our new eBooks, and how to\r\nsubscribe to our email newsletter to hear about new eBooks.\r\n\r\n\r\n",
        "document_name": "pg16865.txt",
        "document_path": "https://www.gutenberg.org/cache/epub/16865/pg16865.txt",
        "document_id": "4c00ac02-e833-4f62-a665-ff6c1095e583",
        "conversion_method": "txt",
        "reader_method": "vanilla",
        "ocr_method": null,
        "page_placeholder": null,
        "metadata": {}
    }


Then, a splitting strategy is needed to create smaller pieces of text that can be easily processed by the LLM. In this case, since the Pinocchio's tale is sorted by chapters, we can use the [**`KeywordSplitter`**](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#keywordsplitter). Then, we can define the regex pattern which follows the book to enumerate the chapers to provide the splitting pattern. As we can see, this pattern is composed by `CHAPTER` in uppercase plus the chapter number in roman numbers (e.g., `CHAPTER I`, `CHAPTER V`, etc.). Hence, our pattern will be: 


```python
REGEX_PATTERN: list[str] = [r"CHAPTER\s+[IVXLCDM]+"]
splitter = KeywordSplitter(
    patterns=REGEX_PATTERN,
    include_delimiters="after",
    flags=re.IGNORECASE,
    chunk_size=100000,
)

splitter_output = splitter.split(reader_output)
print(splitter_output.model_dump_json(indent=4))
```

    {
        "chunks": [
            "﻿The Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook or online\r\nat www.gutenberg.org. If you are not located in the United States,\r\nyou will have to check the laws of
    ...
    90122
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
                    "k0"
                ],
                "chunk_size": 100000
            }
        }
    }


The output will be a [`SplitterOutput`](https://andreshere00.github.io/Splitter_MR/api_reference/splitter/#output-format) object, with the following fields:


```python
print(splitter_output.model_json_schema())
```

    {'description': 'Pydantic model defining the output structure for all splitters.\n\nAttributes:\n    chunks: List of text chunks produced by splitting.\n    chunk_id: List of unique IDs corresponding to each chunk.\n    document_name: The name of the document.\n    document_path: The path to the document.\n    document_id: A unique identifier for the document.\n    conversion_method: The method used for document conversion.\n    reader_method: The method used for reading the document.\n    ocr_m
    ...
    t': None, 'title': 'Reader Method'}, 'ocr_method': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Ocr Method'}, 'split_method': {'default': '', 'title': 'Split Method', 'type': 'string'}, 'split_params': {'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}], 'title': 'Split Params'}, 'metadata': {'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}], 'title': 'Metadata'}}, 'title': 'SplitterOutput', 'type': 'object'}



```python
# Inspect counts
print(
    f"Chunks: {len(splitter_output.chunks)} | First chunk_id: {splitter_output.chunk_id[0] if splitter_output.chunk_id else None}"
)
```

    Chunks: 37 | First chunk_id: ea3c39a3-fb32-485f-9910-c44f3e59de8e


As we can see, we have created 37 chunks (which matches with the number of Pinocchio chapters). Trying to process all of these chunks by an LLM could be hard since the model has a limited window context. So, we can apply an attention mechanism to retrieve only those chunks which are related to the query. To do that, we can build a RAG system. Let's see how can do it! 

## Third step: Set up the vector database and upload the content

To build the RAG, it is necessary to embed the documents that we will use. Firstly, we will instantiate our AzureOpenAI class and check for the embedding dimension to ensure that the encoder model is working properly:


```python
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
    api_version="2025-03-01-preview",
)

# Helper to get embedding dimension from a probe call


def _embedding_dimension(model_name: str) -> int:
    vec = (
        client.embeddings.create(model=model_name, input=["dim-probe"])
        .data[0]
        .embedding
    )
    return len(vec)


EMBEDDING_DIM = _embedding_dimension(AZURE_OPENAI_EMBEDDING)
print(f"Embedding dim: {EMBEDDING_DIM}")
```

    Embedding dim: 3072


Now we will instantiate the Qdrant client. Remember that it is necessary to have Docker installed and run a server. Execute the following instructions:


```python
!python qdrant_server.py
```

    Container 'qdrant_db' is already running.
    Waiting for Qdrant health: http://localhost:6333/healthz
    Qdrant is up! REST: http://localhost:6333
    gRPC: localhost: http://localhost:6334


So, we can create (or recreate, for idempotence purposes) our collection in Qdrant to start to work with.


```python
# Instantiate the Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Chcekc if the collection exists. In case that exists, delete it.
if COLLECTION_NAME in [c.name for c in qdrant.get_collections().collections]:
    # optional: recreate from scratch
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

# Create the collection
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIM, distance=models.Distance.DOT
    ),
)

print(f"Collection '{COLLECTION_NAME}' ready.")
```

    Collection 'pinocchio_demo_chunks' ready.


In order to upload successfully to the database, we will create the variables that we need:

- `texts`, representing the chunks.
- `chunk_ids`, representing the unique identifier for each chunk.
- `base_payload`, all the remaining information about how the file and chunks have been processed.


```python
texts: list[str] = splitter_output.chunks
chunk_ids: list[str] = splitter_output.chunk_id

assert len(texts) == len(chunk_ids), "Mismatch: chunks vs chunk_ids length"
assert len(set(chunk_ids)) == len(chunk_ids), "Duplicate chunk_ids found"

base_payload: dict[str, any] = {
    "source": splitter_output.document_name,
    "document_path": splitter_output.document_path,
    "document_id": splitter_output.document_id,
    "conversion_method": splitter_output.conversion_method,
    "reader_method": splitter_output.reader_method,
    "ocr_method": splitter_output.ocr_method,
    "split_method": splitter_output.split_method,
}
```

We process this information as a list of tuples with the previous information. In fact, we are creating the [points](https://qdrant.tech/documentation/concepts/points/) to be uploaded into our Qdrant instance.


```python
all_points: list[tuple[str, str, dict[str, any]]] = []

for i, chunk_text in enumerate(texts):
    payload = dict(base_payload)
    payload.update(
        {
            "chunk_id": chunk_ids[i],
            "chunk_index": i,
            "text": chunk_text,
        }
    )
    all_points.append((chunk_ids[i], chunk_text, payload))
```

Finally, we embed our content and we can upsert it into batches. This can be performed via the following code:


```python
# Embed & upsert in batches
total = 0
for pack in batched(all_points, 64):
    ids = [pid for pid, _, _ in pack]
    inputs = [txt for _, txt, _ in pack]
    payloads = [pl for _, _, pl in pack]

    emb = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT, input=inputs
    )
    vectors = [d.embedding for d in emb.data]
    assert len(vectors) == len(ids), "Embedding count != ids count"

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        wait=True,  # ensure write is persisted before proceeding
    )
    total += len(ids)

print(f"Upserted {total} chunks to Qdrant.")
```

    Upserted 37 chunks to Qdrant.


As we seen, we have uploaded all the 37 chunks into Qdrant. After this, our RAG system is almost ready to be used.

## Fourth step: Create the Retrieval component

Once all the documents have been upserted, we need to create a retrieval function which allow us to compare the user query with the embedded information from the Vector Database. We can use this function:


```python
def retrieve(query: str, k: int = 5) -> list[dict[str, any]]:
    """Use Qdrant's query_points for flexible retrieval capabilities."""
    q_vec = (
        client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT, input=[query]
        )
        .data[0]
        .embedding
    )

    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=k,
        with_payload=True,
    )
    hits = res.points  # Extract the actual list of hits

    results = []
    for h in hits:
        pl = h.payload or {}
        results.append(
            {
                "score": h.score,
                "chunk_id": pl.get("chunk_id", h.id),
                "text": pl.get("text", ""),
                "source": pl.get("source"),
                "document_id": pl.get("document_id"),
                "chunk_index": pl.get("chunk_index"),
            }
        )
    return results
```

## Fifth Step: Prepare input and output format

We will interact with the RAG system via prompts. So, we will firstly create a `SYSTEM_PROMPT` to indicate how to generate the responses and we will process the response to get the answer as we want to. We can use this function:


```python
SYSTEM_PROMPT = "Answer the user's question concisely but precisely using ONLY the provided context. Cite sources as [chunk_id] next to claims."


def answer_with_rag(query: str, k: int = 5) -> dict[str, any]:
    hits = retrieve(query, k=k)
    # Compose context block with per-chunk headers
    context_blocks = []
    for h in hits:
        header = f"[chunk_id: {h['chunk_id']}] source: {h['source']} (idx {h['chunk_index']})"
        # Keep chunks modest to stay in token limits
        chunk_text = h["text"][:2000]
        context_blocks.append(f"{header}\n{chunk_text}")

    context: str = "\n\n".join(context_blocks)

    client = AzureOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-03-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages: list[dict[str, any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
    ]

    resp = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )

    output: dict = {
        "query": query,
        "hits": hits,
        "answer": resp.choices[0].message.content,
    }

    return output
```

With all of these elements, our system is reaedy to be used. Now we only need to build the entrypoint:

## Sixth Step: testing the RAG

To test the RAG, we can launch a set of answers and see how the model is answering. To test if the answers are correct or not, we can create a little dataset with the questions and the ground truth.

| `question`                       | `ground_truth`                                                  | `answer`                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Who is the author of *Pinocchio: The Tale of a Puppet*?**                                                                                             | Carlo Collodi                                                                                                                                                                                              |        |
| **In Chapter V (“The Flying Egg”), what strange thing happens instead of cooking the egg — i.e. what creature emerges from it?**                        | A little chicken                                                                                                                                                                                           |        |
| **In which chapter does Geppetto give Pinocchio new feet, and how does he do this?**                                                                    | Chapter VIII; Geppetto makes new feet out of two pieces of well-seasoned wood, uses glue (melted in an eggshell) to fasten them, then paints/finishes them so the joints are invisible.                    |        |
| **In Chapter XIII (“The Inn of the Red Craw-Fish”), what does Pinocchio dream about, and what does the ghost of the Talking-Cricket advise him to do?** | He dreams of a field full of shrubs bearing gold sovereigns; the Cricket tells him to go back and give his remaining sovereigns to his poor father instead of trusting those who promise riches overnight. |        |
| **In Chapter XXXI, what is Pinocchio’s status or situation, and is there a change from earlier chapters?**                                              | In Chapter XXXI, Pinocchio enjoys five months of happiness; this is a period of good fortune and contentment, contrasting with his many earlier hardships.                                                 |        |


Finally, we ask to the generative model to get the answers:


```python
# ---- 8) EXAMPLE QUERY ----

# Test questions & ground truths
qa_pairs: list[dict[str, str]] = [
    {
        "question": "Who is the author of _Pinocchio: The Tale of a Puppet_?",
        "ground_truth": "Carlo Collodi",
        "answer": "",
    },
    {
        "question": "In Chapter V (“The Flying Egg”), what strange thing happens instead of cooking the egg — i.e. what creature emerges from it?",
        "ground_truth": "A little chicken",
        "answer": "",
    },
    {
        "question": "In which chapter does Geppetto give Pinocchio new feet, and how does he do this?",
        "ground_truth": "Chapter VIII; Geppetto makes new feet out of two pieces of well-seasoned wood, uses glue (melted in an eggshell) to fasten them, then paints/finishes them so the joints are invisible.",
        "answer": "",
    },
    {
        "question": "In Chapter XIII (“The Inn of the Red Craw-Fish”), what does Pinocchio dream about, and what does the ghost of the Talking-Cricket advise him to do?",
        "ground_truth": "He dreams of a field full of shrubs bearing gold sovereigns; the Cricket tells him to go back and give his remaining sovereigns to his poor father instead of trusting those who promise riches overnight.",
        "answer": "",
    },
    {
        "question": "In Chapter XXXI, what is Pinocchio’s status or situation, and is there a change from earlier chapters?",
        "ground_truth": "In Chapter XXXI, Pinocchio enjoys five months of happiness; this is a period of good fortune and contentment, contrasting with his many earlier hardships.",
        "answer": "",
    },
]

# Get answers
answers = []
for qa in qa_pairs:
    out = answer_with_rag(qa["question"], k=5)
    answers.append(out["answer"])
    print(f"Q: {qa['question']}\nA: {out['answer']}\n{'-' * 40}")
```

    Q: Who is the author of _Pinocchio: The Tale of a Puppet_?
    A: The author of _Pinocchio: The Tale of a Puppet_ is Carlo Collodi [ea3c39a3-fb32-485f-9910-c44f3e59de8e].
    ----------------------------------------
    Q: In Chapter V (“The Flying Egg”), what strange thing happens instead of cooking the egg — i.e. what creature emerges from it?
    A: In Chapter V, instead of cooking the egg, a creature emerges from it, specifically a "large dog" [chunk_id: 18dc09bc].
    ----------------------------------------
    Q
    ...
    at his past disobedience has led to his troubles. This marks a significant change from earlier chapters where he often acted impulsively and selfishly, such as when he stole grapes and faced dire consequences for his actions. His newfound resolve to improve himself indicates a growth in character and a shift towards responsibility and gratitude, particularly towards his father and the Fairy who helped him [chunk_id: 986b6dee-064f-4759-b5f1-1b699a9e3c21].
    ----------------------------------------


We can see which chunks have been retrieved to see if the retrieved content is the expected one:


```python
for i, qa in enumerate(qa_pairs):
    hits = retrieve(qa["question"], k=5)
    print(f"\nQ{i + 1}: {qa['question']}")
    for h in hits:
        print(
            f"chunk_id: {h['chunk_id']}, score: {h['score']:.4f}, text sample: {h['text']!r}"
        )
```

    
    Q1: Who is the author of _Pinocchio: The Tale of a Puppet_?
    chunk_id: ea3c39a3-fb32-485f-9910-c44f3e59de8e, score: 0.5768, text sample: '\ufeffThe Project Gutenberg eBook of Pinocchio: The Tale of a Puppet\r\n    \r\nThis ebook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions\r\nwhatsoever. You may copy it, give it away or re-use it under the terms\r\nof the Project Gutenberg License included with this ebook o
    ...
    ter\r\nwaiting and waiting, he found that no one came, absolutely no one, then\r\nhe remembered his poor father, and, thinking he was dying, he stammered\r\nout:\r\n\r\n"Oh, papa! papa! if only you were here!"\r\n\r\n[Illustration: Four Rabbits as Black as Ink Entered Carrying a Little\r\nBier]\r\n\r\nHis breath failed him and he could say no more. He shut his eyes, opened\r\nhis mouth, stretched his legs, gave a long shudder, and hung stiff and\r\ninsensible.\r\n\r\n\r\n\r\n\r\n[Illustration]'


Now, let's compare the expected outputs with the actual outputs, and see if the generative model have answered correctly according to the provided context.

| **`question`**                                                                                                                                            | **`ground_truth`**                                                                                                                                                                                              | **`answer`**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **`score`** |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Who is the author of *Pinocchio: The Tale of a Puppet*?**                                                                                             | Carlo Collodi                                                                                                                                                                                              | The author of *Pinocchio: The Tale of a Puppet* is Carlo Collodi `[ea3c39a3-fb32-485f-9910-c44f3e59de8e]`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 1 |
| **In Chapter V (“The Flying Egg”), what strange thing happens instead of cooking the egg — i.e. what creature emerges from it?**                        | A little chicken                                                                                                                                                                                           | In Chapter V, instead of cooking the egg, a creature emerges from it, specifically a "large dog" `[chunk_id: 18dc09bc]`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 0 |
| **In which chapter does Geppetto give Pinocchio new feet, and how does he do this?**                                                                    | Chapter VIII; Geppetto makes new feet out of two pieces of well-seasoned wood, uses glue (melted in an eggshell) to fasten them, then paints/finishes them so the joints are invisible.                    | Geppetto gives Pinocchio new feet in **Chapter VIII**. After Pinocchio expresses his desire for new feet, Geppetto initially refuses, wanting to punish him for his naughtiness. However, seeing Pinocchio's pitiable state, Geppetto decides to make the feet. He works diligently with two small pieces of well-seasoned wood and, in less than an hour, finishes the new feet. He then uses melted glue to attach them to Pinocchio while the puppet pretends to sleep, ensuring that the joining is seamless and unnoticeable `[chunk_id: 8aee64ab-1b08-43b1-bbe6-78b8662bac55]`.                                                                                                 | 1 |
| **In Chapter XIII (“The Inn of the Red Craw-Fish”), what does Pinocchio dream about, and what does the ghost of the Talking-Cricket advise him to do?** | He dreams of a field full of shrubs bearing gold sovereigns; the Cricket tells him to go back and give his remaining sovereigns to his poor father instead of trusting those who promise riches overnight. | In Chapter XIII, Pinocchio dreams about the Field of Miracles, where he imagines that he will find great wealth. The ghost of the Talking-Cricket advises him to be cautious and to remember the lessons he has learned about obedience and the consequences of his actions `[chunk_id: 46907011-ef61-44d3-973c-6862e3faf075]`.                                                                                                                                                                                                                                                                                                                                                       | 1 |
| **In Chapter XXXI, what is Pinocchio’s status or situation, and is there a change from earlier chapters?**                                              | In Chapter XXXI, Pinocchio enjoys five months of happiness; this is a period of good fortune and contentment, contrasting with his many earlier hardships.                                                 | In Chapter XXXI, Pinocchio is in a state of reflection and determination to change his ways after experiencing numerous misfortunes. He expresses a desire to become orderly and obedient, recognizing that his past disobedience has led to his troubles. This marks a significant change from earlier chapters where he often acted impulsively and selfishly, such as when he stole grapes and faced dire consequences for his actions. His newfound resolve to improve himself indicates a growth in character and a shift towards responsibility and gratitude, particularly towards his father and the Fairy who helped him `[chunk_id: 986b6dee-064f-4759-b5f1-1b699a9e3c21]`. | 1 |



Despite that one question has been answered incorrectly, 4 out of 5 answers were correct. Logically, these results can be improved by optimizing the RAG workflow or using other chunk techniques. But this is not the main idea for this tutorial. 

**And that's it!** In this tutorial, we have seen how to create a RAG system using SplitterMR. And as you can check, the document processing part was the easiest part! 
