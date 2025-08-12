from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEmbedding(ABC):
    """
    Abstract base for text embedding providers.

    Implementations wrap specific backends (e.g., OpenAI, Azure OpenAI, local
    models) and expose a consistent interface to convert text into numeric
    vectors suitable for similarity search, clustering, and retrieval-augmented
    generation.
    """

    @abstractmethod
    def __init__(self, model_name: str) -> Any:
        """Initialize the embedding backend.

        Args:
            model_name (str): Identifier of the embedding model (e.g.,
                ``"text-embedding-3-large"`` or a local model alias/path).

        Raises:
            ValueError: If required configuration or credentials are missing.
        """

    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client or handle.

        Returns:
            Any: A client/handle used to perform embedding calls (e.g., an SDK
                client instance, session object, or local runner). May be ``None``
                for pure-local implementations that do not require a client.
        """

    @abstractmethod
    def embed_text(
        self,
        text: str,
        **parameters: Dict[str, Any],
    ) -> List[float]:
        """
        Compute an embedding vector for the given text.

        Args:
            text (str): Input text to embed. Implementations may apply
                normalization or truncation according to model limits.
            **parameters (Dict[str, Any]): Additional backend-specific options
                forwarded to the implementation (e.g., user tags, request IDs).

        Returns:
            A single embedding vector representing ``text``.

        Raises:
            ValueError: If ``text`` is empty or exceeds backend constraints.
            RuntimeError: If the embedding call fails or returns an unexpected
                response shape.
        """

    def embed_documents(
        self,
        texts: List[str],
        **parameters: Dict[str, Any],
    ) -> List[List[float]]:
        """Compute embeddings for multiple texts (default loops over `embed_text`).

        Implementations are encouraged to override for true batch performance.

        Args:
            texts: List of input strings to embed.
            **parameters: Backend-specific options.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            ValueError: If `texts` is empty or any element is empty.
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")
        return [self.embed_text(t, **parameters) for t in texts]
