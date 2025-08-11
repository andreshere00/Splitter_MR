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
        pass

    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying client or handle.

        Returns:
            Any: A client/handle used to perform embedding calls (e.g., an SDK
            client instance, session object, or local runner). May be ``None``
            for pure-local implementations that do not require a client.
        """
        pass

    @abstractmethod
    def embed_text(
        self,
        text: str,
        **parameters: Dict[str, Any],
    ) -> List[float]:
        """Compute an embedding vector for the given text.

        Args:
            text (str): Input text to embed. Implementations may apply
                normalization or truncation according to model limits.
            **parameters (Dict[str, Any]): Additional backend-specific options
                forwarded to the implementation (e.g., user tags, request IDs).

        Returns:
            List[float]: A single embedding vector representing ``text``.

        Raises:
            ValueError: If ``text`` is empty or exceeds backend constraints.
            RuntimeError: If the embedding call fails or returns an unexpected
                response shape.
        """
        pass
