import os
from typing import Any, List, Optional

import voyageai

from ..base_embedding import BaseEmbedding


class AnthropicEmbedding(BaseEmbedding):
    """
    Embedding provider aligned with Anthropic's guidance, implemented via Voyage AI.

    Anthropic does not offer a native embeddings API; their docs recommend using
    third-party providers such as **Voyage AI** for high-quality, domain-specific,
    and multimodal embeddings. This class wraps Voyage's Python SDK to provide a
    consistent interface that matches `BaseEmbedding`.

    Example:
        ```python
        from splitter_mr.embedding import AnthropicEmbeddings

        embedder = AnthropicEmbeddings(model_name="voyage-3.5")
        vec = embedder.embed_text("hello world", input_type="document")
        print(len(vec))
        ```
    """

    def __init__(
        self,
        model_name: str = "voyage-3.5",
        api_key: Optional[str] = None,
        default_input_type: Optional[str] = "document",
    ) -> None:
        """
        Initialize the Voyage embeddings provider.

        Args:
            model_name:
                Voyage embedding model name (e.g., "voyage-3.5", "voyage-3-large",
                "voyage-code-3", "voyage-finance-2", "voyage-law-2").
            api_key:
                Voyage API key. If not provided, reads from the `VOYAGE_API_KEY`
                environment variable.
            default_input_type:
                Default for Voyage's `input_type` parameter ("document" | "query").

        Raises:
            ImportError: If the `multimodal` extra (with `voyageai`) is not installed.
            ValueError: If no API key is provided or found in the environment.
        """

        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Voyage API key not provided and 'VOYAGE_API_KEY' environment variable is not set."
                )

        self.client = voyageai.Client(api_key=api_key)
        self.model_name = model_name
        self.default_input_type = default_input_type

    def get_client(self) -> Any:
        """Return the underlying Voyage client."""
        return self.client

    def _ensure_input_type(self, parameters: dict) -> dict:
        """Default `input_type` to self.default_input_type if not set."""
        params = dict(parameters) if parameters else {}
        if "input_type" not in params and self.default_input_type:
            params["input_type"] = self.default_input_type
        return params

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
        """Compute an embedding vector for a single text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")

        params = self._ensure_input_type(parameters)
        result = self.client.embed([text], model=self.model_name, **params)

        if not hasattr(result, "embeddings") or not result.embeddings:
            raise RuntimeError(
                "Voyage returned an empty or malformed embeddings response."
            )

        embedding = result.embeddings[0]
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Voyage returned an invalid embedding vector.")

        return embedding

    def embed_documents(self, texts: List[str], **parameters: Any) -> List[List[float]]:
        """Compute embeddings for multiple texts in one API call."""
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("All items in `texts` must be non-empty strings.")

        params = self._ensure_input_type(parameters)
        result = self.client.embed(texts, model=self.model_name, **params)

        if not hasattr(result, "embeddings") or not result.embeddings:
            raise RuntimeError(
                "Voyage returned an empty or malformed embeddings response."
            )

        if len(result.embeddings) != len(texts):
            raise RuntimeError(
                f"Voyage returned {len(result.embeddings)} embeddings for {len(texts)} inputs."
            )

        embeddings = result.embeddings

        return embeddings
