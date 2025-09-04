import importlib
import mimetypes
from typing import Any, Dict, List, Optional, Tuple

from ...model import BaseVisionModel
from ...schema import HFChatImageContent, HFChatMessage, HFChatTextContent


class HuggingFaceVisionModel(BaseVisionModel):
    """
    Vision-language model wrapper using Hugging Face Transformers.

    This implementation loads a local or Hugging Face Hub model that supports
    image-to-text or multimodal tasks. It accepts a prompt and an image as
    base64 (without the data URI header) and returns the model's generated text.
    Pydantic schema models are used for message validation.

    Example:
        ```python
        import base64, requests
        from splitter_mr.model.models.huggingface_model import HuggingFaceVisionModel

        # Encode an image as base64
        img_bytes = requests.get(
            "https://huggingface.co/datasets/huggingface/documentation-images/"
            "resolve/main/p-blog/candy.JPG"
        ).content
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        model = HuggingFaceVisionModel("ds4sd/SmolDocling-256M-preview")
        result = model.analyze_content("What animal is on the candy?", file=img_b64)
        print(result)  # e.g., "A small green thing."
        ```
    """

    DEFAULT_EXT: str = "jpg"
    FALLBACKS: List[Tuple[str, Optional[Any]]] = [
        ("AutoModelForVision2Seq", None),
        ("AutoModelForImageTextToText", None),
        ("AutoModelForCausalLM", None),
        ("AutoModelForPreTraining", None),
        ("AutoModel", None),
    ]

    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview") -> None:
        """
        Initialize a HuggingFaceVisionModel.

        Args:
            model_name (str, optional): Model repo ID or local path
                (e.g., ``"ds4sd/SmolDocling-256M-preview"``).

        Raises:
            ImportError: If the 'multimodal' extra (transformers) is not installed.
            RuntimeError: If processor or model loading fails after all attempts.
        """

        transformers = importlib.import_module("transformers")

        AutoProcessor = transformers.AutoProcessor
        AutoImageProcessor = transformers.AutoImageProcessor
        AutoConfig = transformers.AutoConfig

        self.model_id = model_name
        self.model = None
        self.processor = None

        # Load processor
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

        # Load model
        config = AutoConfig.from_pretrained(self.model_id)
        errors: List[str] = []

        try:
            arch_name = config.architectures[0]
            ModelClass = getattr(transformers, arch_name)
            self.model = ModelClass.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception as e:
            errors.append(f"[AutoModel by architecture] {e}")

        if self.model is None:
            resolved: List[Tuple[str, Any]] = []
            for name, cls in self.FALLBACKS:
                resolved.append((name, cls or getattr(transformers, name)))
            for name, cls in resolved:
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
        """Return the underlying HuggingFace model instance.

        Returns:
            Any: The instantiated HuggingFace model object.
        """
        return self.model

    def analyze_content(
        self,
        prompt: str,
        file: Optional[bytes],
        file_ext: Optional[str] = None,
        **parameters: Dict[str, Any],
    ) -> str:
        """
        Extract text from an image using the vision-language model.

        This method encodes an image as a data URI, builds a validated
        message using schema models, prepares inputs, and calls the model
        to generate a textual response.

        Args:
            prompt (str): Instruction or question for the model
                (e.g., ``"Describe this image."``).
            file (Optional[bytes]): Image as a base64-encoded string (without prefix).
            file_ext (Optional[str], optional): File extension (e.g., ``"jpg"`` or ``"png"``).
                Defaults to ``"jpg"`` if not provided.
            **parameters (Dict[str, Any]): Extra keyword arguments passed directly
                to the model's ``generate()`` method (e.g., ``max_new_tokens``,
                ``temperature``).

        Returns:
            str: The extracted or generated text.

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

        text_content = HFChatTextContent(type="text", text=prompt)
        image_content = HFChatImageContent(type="image", image=img_data_uri)
        chat_msg = HFChatMessage(role="user", content=[image_content, text_content])
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
