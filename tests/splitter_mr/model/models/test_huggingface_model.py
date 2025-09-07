# tests/splitter_mr/model/models/test_huggingface_model.py
import base64
import builtins
from typing import Iterable, Tuple

import pytest

from splitter_mr.model.models import huggingface_model as hf_mod
from splitter_mr.model.models.huggingface_model import HuggingFaceVisionModel

MODEL_ID = "ds4sd/SmolDocling-256M-preview"

# ---- Fixutes, helpers and patchs ---- #


class FakeTokenList(list):
    @property
    def shape(self) -> Tuple[int, int]:
        return (1, len(self))


class DummyInputs(dict):
    def to(self, _device):
        return self


class DummyProcessor:
    def apply_chat_template(self, *_a, **_k):
        # Minimal output the model expects later
        return DummyInputs({"input_ids": FakeTokenList([1, 2, 3, 4, 5])})

    def decode(self, *_a, **_k) -> str:
        return "This is a fake vision model answer."


class DummyModel:
    device = "cpu"

    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def generate(self, *_a, **_k) -> Iterable[FakeTokenList]:
        # 5 prompt tokens + 3 new tokens
        return [FakeTokenList([1, 2, 3, 4, 5, 99, 100, 101])]


class DummyConfig:
    architectures = ["DummyModelClass"]


@pytest.fixture(autouse=True)
def patch_schema_models(monkeypatch):
    """
    Ensure the objects that go into apply_chat_template always have the expected
    dict shape:
      - {"type": "image", "image": "<data-uri>"}
      - {"type": "text", "text": "<prompt>"}
    and the outer message:
      - {"role": "user", "content": [ ... ]}
    """
    import splitter_mr.schema as schema

    class DummyImageContent:
        def __init__(self, type="image", image=None, **_):
            self.type = type or "image"
            self.image = image

        def model_dump(self, **_):
            return {"type": "image", "image": self.image}

    class DummyTextContent:
        def __init__(self, type="text", text=None, **_):
            self.type = type or "text"
            self.text = text

        def model_dump(self, **_):
            return {"type": "text", "text": self.text}

    class DummyMessage:
        def __init__(self, role, content, **_):
            self.role = role
            self.content = content

        def model_dump(self, **_):
            return {
                "role": self.role,
                "content": [
                    (c.model_dump() if hasattr(c, "model_dump") else c)
                    for c in self.content
                ],
            }

    # Patch in both the schema package and the already-imported symbols inside the module under test
    for mod in (schema, hf_mod):
        monkeypatch.setattr(mod, "HFChatImageContent", DummyImageContent, raising=True)
        monkeypatch.setattr(mod, "HFChatTextContent", DummyTextContent, raising=True)
        monkeypatch.setattr(mod, "HFChatMessage", DummyMessage, raising=True)


@pytest.fixture(autouse=True)
def patch_huggingface(monkeypatch):
    """
    Replace all Hugging Face load points with dummy implementations so tests
    never download weights or call the real processors.
    """
    import transformers

    def patch_symbol(name: str, obj):
        # Patch symbol on transformers, and mirror into the already-imported hf_mod namespace if present
        monkeypatch.setattr(transformers, name, obj, raising=False)
        if hasattr(hf_mod, name):
            monkeypatch.setattr(hf_mod, name, obj, raising=True)

    # Keep class objects, but override their from_pretrained to return dummies
    patch_symbol("AutoProcessor", transformers.AutoProcessor)
    patch_symbol("AutoImageProcessor", transformers.AutoImageProcessor)
    patch_symbol("AutoConfig", transformers.AutoConfig)

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *_a, **_k: DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *_a, **_k: DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: DummyConfig()
    )

    # Mirror those patched classes into the module under test
    hf_mod.AutoProcessor = transformers.AutoProcessor
    hf_mod.AutoImageProcessor = transformers.AutoImageProcessor
    hf_mod.AutoConfig = transformers.AutoConfig

    # Every fallback model & dynamic architecture -> DummyModel
    for sym in (
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
        "AutoModelForCausalLM",
        "AutoModelForPreTraining",
        "AutoModel",
        "DummyModelClass",
    ):
        patch_symbol(sym, DummyModel)

    # Override the FALLBACKS list on the class (value doesn't matter, we return DummyModel)
    hf_mod.HuggingFaceVisionModel.FALLBACKS = [(sym, DummyModel) for sym in range(5)]

    # If code does getattr(transformers, "DummyModelClass"), return DummyModel
    real_getattr = builtins.getattr

    def patched_getattr(obj, name, *args):
        if obj is transformers and name == "DummyModelClass":
            return DummyModel
        return real_getattr(obj, name, *args)

    monkeypatch.setattr(builtins, "getattr", patched_getattr)


@pytest.fixture(scope="module")
def model() -> HuggingFaceVisionModel:
    m = HuggingFaceVisionModel(MODEL_ID)
    # Force dummies on the instance to be 100% sure no real processor runs
    m.processor = DummyProcessor()
    m.model = DummyModel()
    return m


@pytest.fixture
def fake_img_b64() -> str:
    return base64.b64encode(b"fakeimagebytes").decode("utf-8")


# ---- Test cases ---- #


def test_get_client(model):
    assert model.get_client() is model.model


def test_model_loads_with_default_model():
    assert HuggingFaceVisionModel().model


def test_model_loads_with_custom_model():
    assert HuggingFaceVisionModel(MODEL_ID).model


def test_model_fallbacks_on_error(monkeypatch):
    import transformers

    class AlwaysFailModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise Exception("model load fail")

    class DummyConfig2:
        architectures = ["AlwaysFailModel"]

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *a, **k: DummyProcessor()
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *a, **k: DummyProcessor(),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: DummyConfig2()
    )
    monkeypatch.setattr(transformers, "AlwaysFailModel", AlwaysFailModel, raising=False)
    monkeypatch.setattr(transformers, "DummyModel", DummyModel, raising=False)

    # Patch the class fallback list to use the dummies
    hf_mod.HuggingFaceVisionModel.FALLBACKS = [
        ("AlwaysFailModel", AlwaysFailModel),
        ("DummyModel", DummyModel),
    ]

    m = HuggingFaceVisionModel("some/model")
    assert isinstance(m.model, DummyModel)


@pytest.mark.parametrize("prompt", ["What animal is on the candy?", "Describe colours"])
def test_analyze_content_success(model, fake_img_b64, prompt):
    assert "fake vision model answer" in model.analyze_content(prompt, fake_img_b64)


def test_analyze_content_png(model, fake_img_b64):
    assert model.analyze_content("Describe", fake_img_b64, file_ext="png")


def test_analyze_content_missing_image(model):
    with pytest.raises(ValueError):
        model.analyze_content("Prompt", None)


def test_analyze_content_unknown_extension(model, fake_img_b64):
    assert model.analyze_content("Prompt", fake_img_b64, file_ext="zzz")


def test_analyze_content_with_bytes(model, fake_img_b64):
    # Pass bytes, not str (simulate an actual base64-encoded bytes object)
    fake_bytes = fake_img_b64.encode("utf-8")
    assert "fake vision model answer" in model.analyze_content("Prompt", fake_bytes)


def test_prepare_input_failure(model, fake_img_b64, monkeypatch):
    monkeypatch.setattr(
        model.processor,
        "apply_chat_template",
        lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")),
    )
    with pytest.raises(RuntimeError, match="Failed to prepare input"):
        model.analyze_content("Prompt", fake_img_b64)


def test_inference_failure(model, fake_img_b64, monkeypatch):
    # Make sure input preparation succeeds so we hit the generate() path
    monkeypatch.setattr(
        model.processor,
        "apply_chat_template",
        lambda *_a, **_k: DummyInputs({"input_ids": FakeTokenList([1, 2, 3, 4, 5])}),
    )
    monkeypatch.setattr(
        model.model,
        "generate",
        lambda *_a, **_k: (_ for _ in ()).throw(Exception("boom")),
    )
    with pytest.raises(RuntimeError, match="Model inference failed"):
        model.analyze_content("Prompt", fake_img_b64)


def test_model_load_fail(monkeypatch):
    import transformers

    def always_fail(*_a, **_k):
        raise Exception("fail!")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", always_fail)
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", always_fail)
    with pytest.raises(RuntimeError, match="All processor loading attempts failed"):
        HuggingFaceVisionModel("bad/model")
