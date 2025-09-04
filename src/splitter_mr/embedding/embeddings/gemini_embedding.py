import os
from typing import Any, List, Optional

from google import genai

from ..base_embedding import BaseEmbedding


class GeminiEmbedding(BaseEmbedding):
    """
    Embedding provider using Google Gemini's embedding API.

    This class wraps the Gemini API for generating embeddings from text or documents.
    Requires the `google-genai` package and a valid Gemini API key. This class
    is available only if `splitter-mr[multimodal]` is installed.

    Typical usage example:
        ```python
        from splitter_mr.embedding.models.gemini_embedding import GeminiEmbedding
        embedder = GeminiEmbedding(api_key="your-api-key")
        vector = embedder.embed_text("Hello, world!")
        print(vector)
        ```
    """

    def __init__(
        self,
        model_name: str = "models/embedding-001",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Gemini embedding provider.

        Args:
            model_name (str): The Gemini model identifier to use for embedding. Defaults to "models/embedding-001".
            api_key (Optional[str]): The Gemini API key. If not provided, reads from the 'GEMINI_API_KEY' environment variable.

        Raises:
            ImportError: If the `google-genai` package is not installed.
            ValueError: If no API key is provided or found in the environment.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not provided and 'GEMINI_API_KEY' environment variable not set."
            )
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.models = self.client.models

    def get_client(self) -> "genai.Client":
        """
        Return the underlying Gemini API client.

        Returns:
            The loaded Gemini API module (`google.genai`).
        """
        return self.client

    def embed_text(self, text: str, **parameters: Any) -> List[float]:
        """
        Generate an embedding for a single text string using Gemini.

        Args:
            text (str): The input text to embed.
            **parameters (Any): Additional parameters for the Gemini API.

        Returns:
            List[float]: The generated embedding vector.

        Raises:
            ValueError: If the input text is not a non-empty string.
            RuntimeError: If the embedding call fails or returns an invalid response.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")

        try:
            result = self.models.embed_content(
                model=self.model_name, contents=text, **parameters
            )
            embedding = getattr(result, "embedding", None)
            if embedding is None:
                raise RuntimeError(
                    "Gemini embedding call succeeded but no 'embedding' field was returned."
                )
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding from Gemini: {e}") from e

    def embed_documents(self, texts: List[str], **parameters: Any) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings using Gemini.

        Args:
            texts (List[str]): A list of input text strings.
            **parameters (Any): Additional parameters for the Gemini API.

        Returns:
            List[List[float]]: The generated embedding vectors, one per input.

        Raises:
            ValueError: If the input is not a non-empty list of non-empty strings.
            RuntimeError: If the embedding call fails or returns an invalid response.
        """
        if (
            not isinstance(texts, list)
            or not texts  # noqa: W503
            or any(not isinstance(t, str) or not t.strip() for t in texts)  # noqa: W503
        ):
            raise ValueError("`texts` must be a non-empty list of non-empty strings.")

        try:
            result = self.models.embed_content(
                model=self.model_name, contents=texts, **parameters
            )
            # The Gemini API returns a list of embeddings under .embeddings
            embeddings = getattr(result, "embeddings", None)
            if embeddings is None:
                raise RuntimeError(
                    "Gemini embedding call succeeded but no 'embeddings' field was returned."
                )
            return embeddings

        except Exception as e:
            raise RuntimeError(
                f"Failed to get document embeddings from Gemini: {e}"
            ) from e
