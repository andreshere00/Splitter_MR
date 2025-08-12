# **Example**: Splitting Files by Semantic Similarity using `SemanticSplitter`

![Semantic splitting illustration](https://arxiv.org/html/2410.13070v1/extracted/5932831/images/chunkers.png)

For some documents, the best splitting strategy is to break them at **semantic boundaries** rather than fixed lengths or pages. This is exactly what the [`SemanticSplitter`](../../api_reference/splitter.md#semanticsplitter) does, which uses **cosine similarity** to detect topic or meaning shifts between sentences. This allows to produce semantically coherent chunks.

In this example, we will read the famous **Pinocchio** tale ([`pinocchio_example.md`](https://raw.githubusercontent.com/andreshere00/Splitter_MR/refs/heads/main/data/pinocchio_example.md)) using `VanillaReader`, and then split it into chunks using `SemanticSplitter`.

## Step 1. Read the file

You can read the file using `VanillaReader` or any other `Reader` that outputs a [`ReaderOutput` object](../../api_reference/reader.md#output-format).

```python
from splitter_mr.reader import VanillaReader

FILE_PATH = "data/pinocchio_example.md"

reader = VanillaReader()
reader_output = reader.read(file_path=FILE_PATH)
```

The output will look like:

```python
{
    "text": "# Pinocchio by Carlo Colodi (*The Tale of a Puppet*)\n\n## Chapter 1\n\n### THE PIECE OF WOOD THAT LAUGHED...",
    "document_name": "pinocchio_example.md",
    "document_path": "data/pinocchio_example.md",
    "document_id": "3603aabf-191a-48ba-9cf1-2e49372aa19a",
    "conversion_method": "md",
    "reader_method": "vanilla",
    "ocr_method": null,
    "metadata": {}
}
```

Note that this is a Pydantic object, so you can represent it as a JSON dictionary with the `.model_dump_json()` instruction.

## Step 2. Split by semantic similarity

To split semantically, instantiate the `SemanticSplitter` object with an [embedding backend](../../api_reference/embedding.md#embedders) and call to the `.split()` method:

```python
from splitter_mr.embedding import AzureOpenAIEmbedding # can be any other Embedding model.
from splitter_mr.splitter import SemanticSplitter

embedding = AzureOpenAIEmbedding() # can be any other Embedding model.

splitter = SemanticSplitter(embedding)
splitter_output = splitter.split(reader_output)

for idx, chunk in enumerate(splitter_output.chunks):
    print("\n" + "*"*80 + f" Chunk {idx} " + "*"*80 + "\n")
    print(chunk)
```

Example output:

```md
******************************************************************************** Chunk 0 ********************************************************************************
# Pinocchio by Carlo Colodi (*The Tale of a Puppet*)

## Chapter 1

### THE PIECE OF WOOD THAT LAUGHED AND CRIED LIKE A CHILD

There was once upon a time a piece of wood in the shop of an old carpenter named Master Antonio. Everybody, however, called him Master Cherry, on account of the end of his nose, which was always as red and polished as a ripe cherry. No sooner had Master Cherry set eyes on the piece of wood than his face beamed with delight, and, rubbing his hands together with satisfaction, he said softly to himself:

...

************************************************************* Chunk 4 ********************************************************************************
"Yes!" And, becoming more and more angry, from words they came to blows, and, flying at each other, they bit and fought, and scratched. When the fight was over Master Antonio was in possession of Geppetto's yellow wig, and Geppetto discovered that the grey wig belonging to the carpenter remained between his teeth. "Give me back my wig," screamed Master Antonio. ..
```

## How it works

The **SemanticSplitter** algorithm:

1. **Sentence Splitting** — Breaks text into individual sentences using `SentenceSplitter`.
2. **Sliding Window Context** — Combines each sentence with `buffer_size` neighbors before and after for better semantic representation.
3. **Embedding Generation** — Uses the provided `BaseEmbedding` model to get vector representations of each combined sentence.
4. **Distance Calculation** — Computes **cosine distances** between consecutive embeddings.
5. **Breakpoint Detection** — Finds points where the distance exceeds a threshold (based on percentile, standard deviation, interquartile range, or gradient).
6. **Chunk Assembly** — Merges sentences between breakpoints into chunks, ensuring each meets the minimum `chunk_size`.

To consult information more-in-depth about how this algorithm works, you can check the following [section](#annex-semantic-splitting-algorithm).

## Customizing Parameters

You can adjust how chunks are detected and their minimum length:

```python
splitter = SemanticSplitter(
    embedding=embedding,
    buffer_size=1,                     # number of neighbor sentences to include
    breakpoint_threshold_type="percentile",  # method for determining breakpoints
    breakpoint_threshold_amount=80.0,  # threshold value (percentile here)
    chunk_size=1000                    # minimum characters per chunk
)
splitter_output = splitter.split(reader_output)
```

In this case, more chunks are extracted since the splitter becomes more sensitive to smaller semantic changes.

Other available `breakpoint_threshold_type` values are:

- `"percentile"` — Split at distances above a given percentile.
- `"standard_deviation"` — Split at mean + (amount × std deviation).
- `"interquartile"` — Split at mean + (amount × IQR).
- `"gradient"` — Split at steep changes in distance.

Alternatively, you can also directly control the **number of chunks** by setting `number_of_chunks`.

---

## Complete Script

```python
import json
from splitter_mr.embedding import AzureOpenAIEmbedding
from splitter_mr.reader import VanillaReader
from splitter_mr.splitter import SemanticSplitter

FILE_PATH = "data/pinocchio_example.md"

embedding = AzureOpenAIEmbedding()
reader = VanillaReader()
reader_output = reader.read(file_path=FILE_PATH)

print("*"*40 + "\n Output from Reader: \n" + "*"*40)
print(json.dumps(reader_output.model_dump(), indent=4))

splitter = SemanticSplitter(embedding)
splitter_output = splitter.split(reader_output)

print("*"*40 + "\n Output from Splitter: \n" + "*"*40)
print(json.dumps(splitter_output.model_dump(), indent=4))

for idx, chunk in enumerate(splitter_output.chunks):
    print("="*40 + f" Chunk {idx+1} " + "="*40)
    print(chunk)
```

---

## Annex: Semantic Splitting algorithm

The Semantic Splitter detects **semantic shift points** between adjacent sentence windows by:

1. Splitting the text into sentences.
2. Creating a **sliding window** around each sentence (`buffer_size` neighbors on both sides).
3. Embedding each window with the configured `BaseEmbedding`.
4. Computing **cosine distances** between consecutive windows.
5. Selecting **breakpoints** where the signal exceeds a threshold (or meets a desired number of chunks).
6. Emitting chunks between breakpoints, honoring a **minimum** `chunk_size` (in characters).

Below, the full process will be detailed.

### Notation

- Let the document be split into sentences $ S = [s_0, s_1, \ldots, s_{n-1}] $.
- For each index $ i $, define a **window** $ w_i $ as the concatenation of sentences
  $$
  w_i = \text{concat}\big(s_{\max(0,\, i-b)}, \ldots, s_i, \ldots, s_{\min(n-1,\, i+b)}\big),
  $$
  where $ b = \texttt{buffer\_size} $.
- Let $ \mathbf{e}_i = \text{embed}(w_i) \in \mathbb{R}^d $ be the embedding vector.
- Define **cosine similarity** and **distance**:
  $$
  \text{cos\_sim}(\mathbf{e}_i, \mathbf{e}_{i+1}) = 
  \frac{\mathbf{e}_i \cdot \mathbf{e}_{i+1}}{\|\mathbf{e}_i\|\,\|\mathbf{e}_{i+1}\| + \varepsilon}, 
  \qquad
  d_i = 1 - \text{cos\_sim}(\mathbf{e}_i, \mathbf{e}_{i+1}),
  $$
  with a small $ \varepsilon $ for numerical stability.

We obtain a distance vector $ D = [d_0, d_1, \ldots, d_{n-2}] $ (length $ n-1 $).


### Breakpoint Selection

Let $ R $ be the **reference array** used to select cut indices:
- For most strategies, $ R = D $.
- For `"gradient"`, compute $ G = \nabla D $ and set $ R = G $.

We then choose a **threshold** $ T $ and mark **breakpoints** at all indices $ i $ where $ R[i] > T $.

#### Threshold Strategies

1. **Percentile** (default):
   - $ T = \text{percentile}(D, p) $, where $ p \in [0, 100] $.
   - Intuition: cut at the largest $ (100 - p)\% $ jumps.

2. **Standard Deviation**:
   - $ \mu = \text{mean}(D),\ \sigma = \text{std}(D) $
   - $ T = \mu + \alpha \sigma $ (e.g., $ \alpha=3 $).
   - Intuition: cut on statistical outliers.

3. **Interquartile (IQR)**:
   - $ Q_1, Q_3 = \text{percentile}(D, 25), \text{percentile}(D, 75) $
   - $ \text{IQR} = Q_3 - Q_1,\ \mu = \text{mean}(D) $
   - $ T = \mu + \beta \cdot \text{IQR} $ (e.g., $ \beta=1.5 $).
   - Intuition: robust outlier detection vs. heavy tails.

4. **Gradient**:
   - Compute $ G = \nabla D $ and threshold by percentile on $ G $:
     $ T = \text{percentile}(G, p) $.
   - Intuition: cut at steep **changes** in the distance signal (useful when distances drift).

> **Note on units**: For `"percentile"` and `"gradient"`, the class accepts `breakpoint_threshold_amount` in either $[0, 1]$ (interpreted as a fraction, converted internally to $[0,100]$) or $[0,100]$ (as a percentile). For `"standard_deviation"` and `"interquartile"`, the amount is a **multiplier** ($\alpha$ or $\beta$).

### Targeting a Desired Number of Chunks

If `number_of_chunks` is set, the splitter **maps** the requested count to an **inverse percentile** over $ D $:

- Let $ m = |D| $. Requested chunks $ k $ are mapped to a percentile $ y \in [0,100] $ such that:
  - $ k=1 \Rightarrow y \approx 100 $ (almost no cuts),
  - $ k=m \Rightarrow y \approx 0 $ (many cuts).
- The threshold is then $ T = \text{percentile}(D, y) $.

This provides an approximate, monotonic control over chunk count without explicit clustering.

### Chunk Assembly and `chunk_size` Enforcement

After computing indices $ \mathcal{B} = \{i \mid R[i] > T\} $, we sweep left-to-right:

1. Maintain a `start_idx` (initially 0).
2. For each $ i \in \mathcal{B} $ in ascending order, propose a cut **after** sentence $ i $ (i.e., up to index $ i $ inclusive).
3. Build the candidate chunk $ C = \text{concat}(s_{\text{start}}, \ldots, s_{i}) $.
4. If `len(C) >= chunk_size`, **emit** the chunk and set `start_idx = i + 1`.
5. After the sweep, **emit the tail** (if non-empty).

If no candidate passes `chunk_size`, the splitter **falls back** to a single chunk (all sentences concatenated).

> `chunk_size` is a **minimum** only. Chunks can be **larger** depending on where valid cut points occur.