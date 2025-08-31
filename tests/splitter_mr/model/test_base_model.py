import pytest

from splitter_mr.model.base_model import BaseVisionModel

# Helpers


class DummyModel(BaseVisionModel):
    def __init__(self, model_name="dummy-model"):
        self.model_name = model_name

    def get_client(self):
        return "dummy-client"

    def analyze_content(self, prompt: str, file: bytes = None, **parameters) -> str:
        # just echo the input for testing
        return f"extract:{prompt}"


# Test cases
def test_BaseVisionModel_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseVisionModel("foo")


def test_dummy_model_instantiable_and_methods_work():
    dummy = DummyModel()
    assert dummy.get_client() == "dummy-client"
    assert dummy.analyze_content("PROMPT") == "extract:PROMPT"
