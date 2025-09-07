from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from ...embedding import BaseEmbedding
from ...schema import (
    DEFAULT_BREAKPOINTS,
    BreakpointThresholdType,
    ReaderOutput,
    SplitterOutput,
)
from ...splitter import BaseSplitter
from .sentence_splitter import SentenceSplitter


def _cosine_similaritynp(a: List[float], b: List[float], eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors using NumPy for speed.

    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.
        eps (float): Numerical stability epsilon.

    Returns:
        float: Cosine similarity in [-1, 1].
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.maximum(np.linalg.norm(va) * np.linalg.norm(vb), eps))
    return float(np.dot(va, vb) / denom)


def _combine_sentences(
    sentences: List[Dict[str, Any]], buffer_size: int
) -> List[Dict[str, Any]]:
    """Create a sliding window string around each sentence using NumPy helpers.

    For each sentence i, concatenates up to `buffer_size` neighbors on both sides.

    Args:
        sentences (List[Dict[str, Any]]): Items with {"sentence": str, "index": int}.
        buffer_size (int): Number of neighbors on each side.

    Returns:
        List[Dict[str, Any]]: In-place augmented with "combined_sentence".
    """
    n = len(sentences)
    for i in range(n):
        left = int(np.maximum(0, i - buffer_size))
        right = int(np.minimum(n, i + 1 + buffer_size))
        parts = [sentences[j]["sentence"] for j in range(left, right)]
        sentences[i]["combined_sentence"] = " ".join(parts).strip()
    return sentences


class SemanticSplitter(BaseSplitter):
    """
    Split text into semantically coherent chunks using embedding similarity.

    **Pipeline:**

    - Split text into sentences via `SentenceSplitter` (one sentence chunks).
    - Build a sliding window around each sentence (`buffer_size`).
    - Embed each window with `BaseEmbedding` (batched).
    - Compute cosine *distances* between consecutive windows (1 - cosine_sim).
    - Pick breakpoints using a thresholding strategy, or aim for `number_of_chunks`.
    - Join sentences between breakpoints; enforce minimum size via `chunk_size`.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        *,
        buffer_size: int = 1,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> None:
        """Initialize the semantic splitter.

        Args:
            embedding (BaseEmbedding): Embedding backend.
            buffer_size (int): Neighbor window size around each sentence.
            breakpoint_threshold_type (BreakpointThresholdType): Threshold strategy:
                "percentile" | "standard_deviation" | "interquartile" | "gradient".
            breakpoint_threshold_amount (Optional[float]): Threshold parameter. If None,
                uses sensible defaults per strategy (e.g., 95th percentile).
            number_of_chunks (Optional[int]): If set, pick a threshold that
                approximately yields this number of chunks (inverse percentile).
            chunk_size (int): **Minimum** characters required to emit a chunk.
        """
        super().__init__(chunk_size=chunk_size)
        self.embedding = embedding
        self.buffer_size = int(buffer_size)
        self.breakpoint_threshold_type = cast(
            BreakpointThresholdType, breakpoint_threshold_type
        )
        self.breakpoint_threshold_amount = (
            DEFAULT_BREAKPOINTS[self.breakpoint_threshold_type]
            if breakpoint_threshold_amount is None
            else float(breakpoint_threshold_amount)
        )
        self.number_of_chunks = number_of_chunks
        self._sentence_splitter = SentenceSplitter(
            chunk_size=1, chunk_overlap=0, separators=[".", "!", "?"]
        )

    # ---------- Helpers ----------

    def _split_into_sentences(self, reader_output: ReaderOutput) -> List[str]:
        """Split the input text into sentences using `SentenceSplitter` (no overlap).

        Args:
            reader_output (ReaderOutput): The document to split.

        Returns:
            List[str]: List of sentences preserving punctuation.
        """
        sent_out = self._sentence_splitter.split(reader_output)
        return sent_out.chunks

    def _calculate_sentence_distances(
        self, single_sentences: List[str]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Embed sentence windows (batch) and compute consecutive cosine distances.

        Args:
            single_sentences (List[str]): Sentences in order.

        Returns:
            Tuple[List[float], List[Dict[str, Any]]]:
                - distances between consecutive windows (len = n-1)
                - sentence dicts enriched with combined text and embeddings
        """
        # Prepare sentence dicts and combine with buffer
        sentences = [
            {"sentence": s, "index": i} for i, s in enumerate(single_sentences)
        ]
        sentences = _combine_sentences(sentences, self.buffer_size)

        # Batch embed all combined sentences
        windows = [item["combined_sentence"] for item in sentences]
        embeddings = self.embedding.embed_documents(windows)

        for item, emb in zip(sentences, embeddings):
            item["combined_sentence_embedding"] = emb

        # Distances (1 - cosine similarity) between consecutive windows
        n = len(sentences)
        if n <= 1:
            return [], sentences

        distances: List[float] = []
        for i in range(n - 1):
            sim = _cosine_similaritynp(
                sentences[i]["combined_sentence_embedding"],
                sentences[i + 1]["combined_sentence_embedding"],
            )
            dist = 1.0 - sim
            distances.append(dist)
            sentences[i]["distance_to_next"] = dist

        return distances, sentences

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """Estimate a percentile threshold to reach `number_of_chunks`.

        Maps desired chunks x∈[1, len(distances)] to percentile y∈[100, 0].

        Args:
            distances (List[float]): Consecutive distances.

        Returns:
            float: Threshold value as a percentile over `distances`.
        """
        assert self.number_of_chunks is not None
        x1, y1 = float(len(distances)), 0.0
        x2, y2 = 1.0, 100.0
        x = max(min(float(self.number_of_chunks), x1), x2)
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1) if x2 != x1 else y2
        y = float(np.clip(y, 0.0, 100.0))
        return float(np.percentile(distances, y)) if distances else 0.0

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        """Compute the breakpoint threshold and reference array per selected strategy.

        Args:
            distances (List[float]): Consecutive distances between windows.

        Returns:
            Tuple[float, List[float]]: (threshold, reference_array)
                If strategy == "gradient", reference_array is the gradient;
                otherwise it's `distances`.
        """
        if not distances:
            return 0.0, distances

        if self.breakpoint_threshold_type == "percentile":
            return (
                float(np.percentile(distances, self.breakpoint_threshold_amount)),
                distances,
            )

        if self.breakpoint_threshold_type == "standard_deviation":
            mu = float(np.mean(distances))
            sd = float(np.std(distances))
            return mu + self.breakpoint_threshold_amount * sd, distances

        if self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25.0, 75.0])
            iqr = float(q3 - q1)
            mu = float(np.mean(distances))
            return mu + self.breakpoint_threshold_amount * iqr, distances

        if self.breakpoint_threshold_type == "gradient":
            grads = np.gradient(np.asarray(distances, dtype=np.float64)).tolist()
            thr = float(np.percentile(grads, self.breakpoint_threshold_amount))
            return thr, grads  # use gradient array as the reference

        raise ValueError(
            f"Unexpected breakpoint_threshold_type: {self.breakpoint_threshold_type}"
        )

    # ---------- Public API ----------

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """Split the document text into semantically coherent chunks.

        Args:
            reader_output (ReaderOutput): The document text & metadata.

        Returns:
            SplitterOutput: Chunks, IDs, metadata, and splitter configuration.

        Notes:
            - With 1 sentence (or 2 in gradient mode), returns the text/sentences as-is.
            - Chunks shorter than `chunk_size` (minimum) are skipped and merged forward.
            - `chunk_size` behaves as the *minimum* chunk size in this splitter.
        """
        text = reader_output.text
        if text == "" or text is None:
            raise ValueError("No text has been provided")

        amt = self.breakpoint_threshold_amount
        if (
            self.breakpoint_threshold_type in ("percentile", "gradient")
            and 0.0 < amt <= 1.0  # noqa: W503
        ):
            self.breakpoint_threshold_amount = amt * 100.0

        sentences = self._split_into_sentences(reader_output)

        # Edge cases where thresholds aren't meaningful
        if len(sentences) <= 1:
            chunks = sentences if sentences else [text]
        elif self.breakpoint_threshold_type == "gradient" and len(sentences) == 2:
            chunks = sentences
        else:
            distances, sentence_dicts = self._calculate_sentence_distances(sentences)

            if self.number_of_chunks is not None and distances:
                # Pick top (k-1) distances as breakpoints
                k = int(self.number_of_chunks)
                m = max(0, min(k - 1, len(distances)))  # number of cuts to make
                if m == 0:
                    indices_above = []  # single chunk
                else:
                    # indices of the m largest distances (breaks), sorted in ascending order
                    idxs = np.argsort(np.asarray(distances))[-m:]
                    indices_above = sorted(int(i) for i in idxs.tolist())
            else:
                threshold, ref_array = self._calculate_breakpoint_threshold(distances)
                indices_above = [
                    i for i, val in enumerate(ref_array) if val > threshold
                ]

            chunks: List[str] = []
            start_idx = 0

            for idx in indices_above:
                end = idx + 1  # inclusive slice end
                candidate = " ".join(
                    d["sentence"] for d in sentence_dicts[start_idx:end]
                ).strip()
                if len(candidate) < self.chunk_size:
                    # too small: keep accumulating (do NOT move start_idx)
                    continue
                chunks.append(candidate)
                start_idx = end

            # Tail (always emit whatever remains)
            if start_idx < len(sentence_dicts):
                tail = " ".join(
                    d["sentence"] for d in sentence_dicts[start_idx:]
                ).strip()
                if tail:
                    chunks.append(tail)

            if not chunks:
                chunks = [" ".join(sentences).strip() or (reader_output.text or "")]

        # IDs & metadata
        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()
        model_name = getattr(self.embedding, "model_name", None)

        return SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="semantic_splitter",
            split_params={
                "buffer_size": self.buffer_size,
                "breakpoint_threshold_type": self.breakpoint_threshold_type,
                "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
                "number_of_chunks": self.number_of_chunks,
                "chunk_size": self.chunk_size,
                "model_name": model_name,
            },
            metadata=metadata,
        )
