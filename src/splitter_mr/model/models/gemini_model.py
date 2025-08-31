import base64
import importlib
import mimetypes
import os
from typing import Any, Optional

from ...model import BaseVisionModel


def _require_extra(extra: str, import_name: Optional[str] = None) -> None:
    """Raise ImportError with install instructions if a required extra is missing."""
    mod = import_name or extra
    try:
        __import__(mod)
    except ImportError as e:
        raise ImportError(
            f"This feature requires the '{extra}' extra.\n"
            f"Install it with:\n\n"
            f"    pip install splitter-mr[{extra}]\n"
        ) from e


class GeminiVisionModel(BaseVisionModel):
    """Implementation of `BaseVisionModel` using Google's Gemini Image Understanding API."""

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
    ) -> None:
        """
        Initialize the GeminiVisionModel.

        Args:
            api_key: Gemini API key. If not provided, uses 'GEMINI_API_KEY' env var.
            model_name: Vision-capable Gemini model name.

        Raises:
            ImportError: If `google-generativeai` is not installed.
            ValueError: If no API key is provided or 'GEMINI_API_KEY' not set.
        """
        _require_extra("multimodal", "google.genai")

        # Lazy import so test monkeypatching works
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Gemini API key not provided or 'GEMINI_API_KEY' not set."
            )

        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        self.model = self.client.models
        self._types = types  # keep handle for analyze_content

    def get_client(self) -> Any:
        """Return the underlying Gemini SDK client."""
        return self.client

    def analyze_content(
        self,
        prompt: str,
        file: Optional[bytes],
        file_ext: Optional[str] = None,
        **parameters: Any,
    ) -> str:
        """Extract text from an image using Gemini's image understanding API."""
        if file is None:
            raise ValueError("No image file provided for extraction.")

        ext = (file_ext or "jpg").lower()
        mime_type = mimetypes.types_map.get(f".{ext}", "image/jpeg")

        img_b64 = file.decode("utf-8") if isinstance(file, (bytes, bytearray)) else file
        try:
            img_bytes = base64.b64decode(img_b64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}")

        # Build Gemini-compatible parts (using lazy-imported types)
        image_part = self._types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        text_part = prompt
        contents = [image_part, text_part]

        try:
            response = self.model.generate_content(
                model=self.model_name,
                contents=contents,
                **parameters,
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini model inference failed: {e}")
