from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np
import torch

from ..base_embedding import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    """
    Encoder provider using Hugging Face `sentence-transformers` models.

    This class wraps a local (or HF Hub) SentenceTransformer model to produce
    dense embeddings for text. It provides a consistent interface with your
    `BaseEmbedding` and convenient options for device selection and optional
    input-length validation. This class is available only if
    `splitter-mr[multimodal]` is installed.

    Example:
        ```python
        from splitter_mr.embedding.models.huggingface_embedding import HuggingFaceEmbedding

        # Any sentence-transformers checkpoint works (local path or HF Hub id)
        embedder = HuggingFaceEmbedding(
            model_name="ibm-granite/granite-embedding-english-r2",
            device="cpu",            # or "cuda", "mps", etc.
            normalize=True,          # L2-normalize outputs
            enforce_max_length=True  # raise if text exceeds model max seq length
        )

        vector = embedder.embed_text("hello world")
        print(vector)
        ```
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = "cpu",
        normalize: bool = True,
        enforce_max_length: bool = False,
    ) -> None:
        """
        Initialize the sentence-transformers embeddings provider.

        Args:
            model_name:
                SentenceTransformer model id or local path. Examples:
                - `"ibm-granite/granite-embedding-english-r2"`
                - `"sentence-transformers/all-MiniLM-L6-v2"`
                - `"/path/to/local/model"`
            device:
                Optional device spec (e.g., `"cpu"`, `"cuda"`, `"mps"` or a
                `torch.device`). If omitted, sentence-transformers chooses.
            normalize:
                If True, return L2-normalized embeddings (sets
                `normalize_embeddings=True` in `encode`).
            enforce_max_length:
                If True, attempt to count tokens and raise `ValueError` when
                input exceeds the model's configured max sequence length.
                (If the model/tokenizer does not expose this reliably, the
                check is skipped gracefully.)

        Raises:
            ValueError: If the model cannot be loaded.
        """

        from sentence_transformers import SentenceTransformer

        st_device = str(device) if device is not None else None
        try:
            self.model = SentenceTransformer(model_name, device=st_device)
        except Exception as e:
            raise ValueError(
                f"Failed to load SentenceTransformer '{model_name}': {e}"
            ) from e

        self.model_name = model_name
        self.normalize = normalize
        self.enforce_max_length = enforce_max_length

    def get_client(self) -> "SentenceTransformer":
        """Return the underlying `SentenceTransformer` instance."""
        return self.model

    def _max_seq_length(self) -> Optional[int]:
        """Best-effort retrieval of model's max sequence length."""
        try:
            # sentence-transformers exposes this on the model
            return int(self.model.get_max_seq_length())
        except Exception:
            try:
                # Fallback: some versions have `max_seq_length` attribute
                return int(getattr(self.model, "max_seq_length", None))
            except Exception:
                return None

    def _count_tokens(self, text: str) -> Optional[int]:
        """
        Best-effort token counting via model.tokenize; returns None if unavailable.
        """
        try:
            features = self.model.tokenize([text])  # dict with "input_ids"
            input_ids = features["input_ids"]
            # input_ids is usually a list/array/tensor of shape [batch, seq]
            if isinstance(input_ids, list):
                first = input_ids[0]
                return len(first)
            if torch is not None and torch.is_tensor(input_ids):
                return int(input_ids.shape[1])
            if isinstance(input_ids, np.ndarray):
                return int(input_ids.shape[1])
        except Exception:
            pass
        return None

    def _validate_length_if_needed(self, text: str) -> None:
        """Raise ValueError if enforce_max_length=True and text is too long."""
        if not self.enforce_max_length:
            return
        max_len = self._max_seq_length()
        tok_count = self._count_tokens(text)
        if max_len is not None and tok_count is not None and tok_count > max_len:
            raise ValueError(
                f"Input exceeds model max sequence length ({tok_count} > {max_len} tokens)."
            )

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
        """
        Compute an embedding vector for a single text string.

        Args:
            text:
                The text to embed. Must be non-empty. If `enforce_max_length`
                is True, a ValueError is raised when it exceeds the model limit.
            **parameters:
                Extra keyword arguments forwarded to `SentenceTransformer.encode`.
                Common options include:
                  - `batch_size` (int)
                  - `show_progress_bar` (bool)
                  - `convert_to_tensor` (bool)  # will be forced False here
                  - `device` (str)
                  - `normalize_embeddings` (bool)

        Returns:
            List[float]: The computed embedding vector.

        Raises:
            ValueError: If `text` is empty or exceeds length constraints (when enforced).
            RuntimeError: If the embedding call fails unexpectedly.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("`text` must be a non-empty string.")

        self._validate_length_if_needed(text)

        # Ensure Python list output
        parameters = dict(parameters)  # shallow copy
        parameters["convert_to_tensor"] = False
        parameters.setdefault("normalize_embeddings", self.normalize)

        try:
            # `encode` accepts a single string and returns a 1D array-like
            vec = self.model.encode(text, **parameters)
        except Exception as e:
            raise RuntimeError(f"Embedding call failed: {e}") from e

        # Normalize output to List[float]
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32, copy=False).tolist()
        if torch is not None and hasattr(vec, "detach"):
            return vec.detach().cpu().float().tolist()
        if isinstance(vec, (list, tuple)):
            return [float(x) for x in vec]
        # Anything else: try to coerce
        try:
            return list(map(float, vec))  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(f"Unexpected embedding output type: {type(vec)}") from e

    def embed_documents(self, texts: List[str], **parameters: Any) -> List[List[float]]:
        """
        Compute embeddings for multiple texts efficiently using `encode`.

        Args:
            texts:
                List of input strings to embed. Must be non-empty and contain
                only non-empty strings. Length enforcement is applied per item
                if `enforce_max_length=True`.
            **parameters:
                Extra keyword arguments forwarded to `SentenceTransformer.encode`.
                Common options:
                  - `batch_size` (int)
                  - `show_progress_bar` (bool)
                  - `convert_to_tensor` (bool)  # will be forced False here
                  - `device` (str)
                  - `normalize_embeddings` (bool)

        Returns:
            List[List[float]]: One embedding per input string.

        Raises:
            ValueError: If `texts` is empty or any element is empty/non-string.
            RuntimeError: If the embedding call fails unexpectedly.
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        if any((not isinstance(t, str) or not t) for t in texts):
            raise ValueError("All items in `texts` must be non-empty strings.")

        if self.enforce_max_length:
            for t in texts:
                self._validate_length_if_needed(t)

        parameters = dict(parameters)
        parameters["convert_to_tensor"] = False
        parameters.setdefault("normalize_embeddings", self.normalize)

        try:
            # Returns ndarray (n, d) or list-of-lists
            mat = self.model.encode(texts, **parameters)
        except Exception as e:
            raise RuntimeError(f"Batch embedding call failed: {e}") from e

        if isinstance(mat, np.ndarray):
            return mat.astype(np.float32, copy=False).tolist()
        if torch is not None and hasattr(mat, "detach"):
            return mat.detach().cpu().float().tolist()
        if (
            isinstance(mat, list)
            and mat  # noqa: W503
            and isinstance(mat[0], (list, tuple, float, int))  # noqa: W503
        ):
            # Already python lists (ST often returns this when convert_to_tensor=False)
            if mat and isinstance(mat[0], (float, int)):  # single vector in a flat list
                return [list(map(float, mat))]
            return [list(map(float, row)) for row in mat]  # type: ignore[arg-type]

        raise RuntimeError(f"Unexpected batch embedding output type: {type(mat)}")
