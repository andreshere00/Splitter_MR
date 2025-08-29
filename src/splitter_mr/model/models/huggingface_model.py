import mimetypes
from typing import Any, Dict, List, Optional, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForPreTraining,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ...model import BaseVisionModel
from ...schema import HFChatImageContent, HFChatMessage, HFChatTextContent


class HuggingFaceVisionModel(BaseVisionModel):
    """
    Implementation of BaseVisionModel using Hugging Face Transformers vision-language models.

    Loads a local or Hugging Face Hub model that supports image-to-text or multimodal tasks.
    Accepts a prompt and an image as base64 (without the data URI header), returning only the model's text output.
    Message construction and validation are performed using Pydantic schema models for safety.

    Example:
        ```python
        import requests
        import base64
        from splitter_mr.model.models.huggingface_model import HuggingFaceVisionModel

        # Fetch an image and encode as base64 string (without prefix)
        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
        img_bytes = requests.get(img_url).content
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Initialize the model (first call may download weights)
        model = HuggingFaceVisionModel("ds4sd/SmolDocling-256M-preview")

        # Call extract_text
        prompt = "What animal is on the candy?"
        result = model.extract_text(prompt, file=img_b64, file_ext="jpg")
        print(result)
        ```
        ```python
        A small green thing.
        ```
    """

    DEFAULT_EXT: str = "jpg"
    FALLBACKS: List[Tuple] = [
        ("AutoModelForVision2Seq", AutoModelForVision2Seq),
        ("AutoModelForImageTextToText", AutoModelForImageTextToText),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModelForPreTraining", AutoModelForPreTraining),
        ("AutoModel", AutoModel),
    ]

    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview") -> None:
        """
        Initialize a HuggingFaceVisionModel.

        Args:
            model_name (str): Model repo ID or path (e.g., "ds4sd/SmolDocling-256M-preview").
                Can be a Hugging Face Hub model or a local path.

        Raises:
            RuntimeError: If the model or processor cannot be loaded.
        """
        self.model_id = model_name
        self.model = None
        self.processor = None

        # Load processor (robust fallback)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception:
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            except Exception as e:
                raise RuntimeError("All processor loading attempts failed.") from e

        # Load model (robust fallback)
        config = AutoConfig.from_pretrained(self.model_id)
        errors = []

        try:
            arch_name = config.architectures[0]
            ModelClass = getattr(transformers, arch_name)
            self.model = ModelClass.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception as e:
            errors.append(f"[AutoModel by architecture] {e}")

        if self.model is None:
            for name, cls in self.FALLBACKS:
                try:
                    self.model = cls.from_pretrained(
                        self.model_id, trust_remote_code=True
                    )
                    break
                except Exception as e:
                    errors.append(f"[{name}] {e}")

        if self.model is None:
            raise RuntimeError(
                "All model loading attempts failed:\n" + "\n".join(errors)
            )

    def get_client(self) -> Any:
        """
        Returns the underlying HuggingFace model object.

        Returns:
            Any: The instantiated HuggingFace model object.
        """
        return self.model

    def extract_text(
        self,
        prompt: str,
        file: Optional[bytes],
        file_ext: Optional[str] = None,
        **parameters: Dict[str, Any],
    ) -> str:
        """
        Extract text from an image using a Hugging Face vision-language model.

        Encodes the image as a data URI with the appropriate MIME type (derived from file extension),
        builds a Pydantic-validated chat template, and calls the loaded model to generate a textual response.

        Args:
            prompt (str): The instruction or caption for the image (e.g., "Describe this image.").
            file (bytes or str): The image, as a base64-encoded string (no prefix, just the base64 body).
            file_ext (str, optional): The image file extension (e.g., "png", "jpg"). Defaults to "jpg".
            **parameters (Any): Additional keyword arguments passed directly to
                the model's ``generate()`` method (e.g., ``max_new_tokens``, ``temperature``, etc.).

        Returns:
            str: The extracted or generated text returned by the vision-language model.

        Raises:
            ValueError: If ``file`` is None.
            RuntimeError: If input preparation or inference fails.
        """
        if file is None:
            raise ValueError("No image file provided for extraction.")

        ext = (file_ext or self.DEFAULT_EXT).lower()
        mime_type = mimetypes.types_map.get(f".{ext}", "image/jpeg")
        img_b64 = file if isinstance(file, str) else file.decode("utf-8")
        img_data_uri = f"data:{mime_type};base64,{img_b64}"

        # Build validated Pydantic chat message
        text_content = HFChatTextContent(type="text", text=prompt)
        image_content = HFChatImageContent(type="image", image=img_data_uri)
        chat_msg = HFChatMessage(role="user", content=[image_content, text_content])

        # Convert to dict for HuggingFace template
        messages = [chat_msg.model_dump(exclude_none=True)]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
            ).to(self.model.device)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare input: {e}")

        try:
            max_new_tokens = parameters.pop("max_new_tokens", 40)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **parameters
            )
            output_text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            return output_text
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
